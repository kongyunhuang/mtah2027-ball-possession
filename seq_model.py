"""
seq_model, train a non-causal temporal convolutional network (TCN) on the same per frame features as mtah_v3, leave-one-match-out, and write held-out probabilities in the same format as mtah_v3
============================
The per frame gradient boosted tree model only sees the features of one frame (which already include window statistics over a
few seconds of past and future). Here a model learns the temporal context end to end. The input is the per frame feature
sequence of build_features (cut into halves), eight dilated convolution layers give a receptive field of about plus or minus
20 seconds, and the output is the H/A/D probability of every frame. Data, degradation and label conventions are reused from
build_sources of mtah_v3, so the held-out probabilities can be scored and ensembled directly with rescore_oof.py,
spell_decode.py and ensemble_oof.py. Training uses public sources only and never reads BAR-ATH. The submitted networks use
--no_inner_val --ema 0.998, so no validation match is kept and the exponential moving average of the weights after the last step
is saved. Without --no_inner_val one more training match is set aside in every fold for internal validation and the checkpoint is
chosen by validation cross entropy (--select_by acc chooses by validation per frame accuracy); --label_smoothing adds label
smoothing to the cross entropy.

Inputs  --sources metrica,idsse and so on (same as mtah_v3), --extra_eval 3:15,0:0,4:21 for extra benchmarks; the model and training
        hyperparameters are described in the argument help
Outputs  under $MTAH_RUNS/<name>/, loso_oof_proba.npz, loso_oof_proba_eval_<tag>.npz, model.joblib (metadata only, the weights
         are saved separately as seq_<match>.pt), metrics.json and versions.json; only summary numbers and progress on screen

Usage
  python -u seq_model.py --name seq_s0 --seed 0 --sources metrica,idsse --realistic --noise 0.01 --touch_set clean --touch_feats --gap_feats \
      --idsse_min_spell 25 --channels 128 --blocks 8 --window 4096 --batch 8 --augment swap --no_inner_val --steps 2500 --trans_weight 2 \
      --trans_win 12 --ema 0.998 --rand_degrade 3 --rand_bench 3 --rand_main --extra_eval 3:15,0:0,4:21 --lazy_norm --slim_m --aux_win 6 \
      --save_boundary --aux_boundary 1 --setp_clip --shape_feats --comove_feats
  python -u seq_model.py --name seq_smoke --sources metrica --realistic --noise 0.01 --touch_feats --gap_feats --smoke

Last modified 2026-11-15
"""
import argparse
import json
import math
import os
import platform
import sys
import time

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtah_v2 import to_hadx, viterbi3_tv  # noqa: E402
from mtah_v3 import build_sources  # noqa: E402
from possession.mtah_features import build_features  # noqa: E402
from possession.spell_metrics import codes_from_hadx, evaluate  # noqa: E402

T0 = time.time()
OUT = {}


def rep(k, v):
    OUT[k] = v
    print("%s = %s" % (k, v if not isinstance(v, float) else "%.4f" % v), flush=True)


def pick_device(name):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_model(n_in, channels, n_blocks, kernel, dropout, arch="tcn", rnn_layers=2, n_out=3, xf_layers=4, xf_heads=4, xf_stride=8, pset=None):
    """arch tcn is the dilated convolution network; gru is a bidirectional GRU (rnn_layers layers, channels//2 units per direction) with the same input and output as tcn;
    xf is a convolution stem plus a Transformer encoder (xf_layers layers, xf_heads heads, downsampled into tokens by xf_stride), again with the same input and output"""
    import torch.nn as nn
    import torch.nn.functional as F_
    n_raw, n_ps = n_in, 0
    if pset:                                                # --pset_pool. The last K*f columns are per-player features; the backbone sees the other columns plus E+1 channels for each team
        K_, f_ = int(pset[0]), int(pset[1]); n_ps = K_ * f_; E_ = 16
        ps_mode = str(pset[2]) if len(pset) > 2 else "pool"   # a third item "point" selects the per-player scoring output structure (--pset_point); without it the per-team pooling is used
        n_in = n_raw - n_ps + 2 * (E_ + 1) if ps_mode == "pool" else n_raw - n_ps
        n_user, Hc_ = n_out, 64
        if ps_mode == "point":
            n_out = Hc_                                     # the backbone does not output classes directly but Hc_ context channels; the classes below read this value when they are built

    class ResBlock(nn.Module):
        def __init__(self, c, k, d, p):
            super().__init__()
            self.conv1 = nn.Conv1d(c, c, k, dilation=d, padding=d * (k - 1) // 2)
            self.norm = nn.GroupNorm(1, c)
            self.conv2 = nn.Conv1d(c, c, 1)
            self.drop = nn.Dropout(p)
            self.act = nn.GELU()

        def forward(self, x):
            return x + self.drop(self.conv2(self.act(self.norm(self.conv1(x)))))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_in, channels, 1)
            self.blocks = nn.ModuleList([ResBlock(channels, kernel, 2 ** i, dropout) for i in range(n_blocks)])
            self.norm = nn.GroupNorm(1, channels)
            self.out = nn.Conv1d(channels, n_out, 1)       # with n_out 4 the 4th channel is the auxiliary change-point head (--aux_boundary)

        def forward(self, x):          # x [B, n_in, T] → logits [B, 3, T]
            h = self.inp(x)
            for b in self.blocks:
                h = b(h)
            return self.out(self.norm(h))

    class BiGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_in, channels, 1)
            self.rnn = nn.GRU(channels, channels // 2, num_layers=rnn_layers, batch_first=True, bidirectional=True,
                              dropout=dropout if rnn_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.norm = nn.GroupNorm(1, channels)
            self.out = nn.Conv1d(channels, n_out, 1)       # with n_out 4 the 4th channel is the auxiliary change-point head (--aux_boundary)

        def forward(self, x):          # x [B, n_in, T] → logits [B, 3, T]
            h = self.inp(x).transpose(1, 2)            # [B, T, C]
            h, _ = self.rnn(h)
            h = self.drop(h).transpose(1, 2)           # [B, C, T]
            return self.out(self.norm(h))

    class ConvXF(nn.Module):
        """Convolution stem plus Transformer. First 3 local convolution blocks with dilation 1, 2, 4, then average pooling by xf_stride into tokens, plus a wav2vec style depthwise convolution positional encoding
        (kernel 15; it encodes relative position only, so any length works at inference), then xf_layers pre-norm encoder layers (xf_heads heads, feed-forward width twice the channels), linear interpolation back to frames added to the local features,
        and 2 more local convolution blocks before the head. A training window of 4096 frames at stride 8 gives 512 tokens; an inference chunk of 8192 plus 2048 frames of overlap gives 1280"""
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_in, channels, 1)
            self.pre = nn.ModuleList([ResBlock(channels, kernel, 2 ** i, dropout) for i in range(3)])
            self.pool = nn.AvgPool1d(xf_stride, stride=xf_stride, ceil_mode=True)
            self.pos = nn.Conv1d(channels, channels, 15, padding=7, groups=channels)
            layer = nn.TransformerEncoderLayer(channels, xf_heads, dim_feedforward=2 * channels, dropout=dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, xf_layers, enable_nested_tensor=False)
            self.mix = nn.Conv1d(channels, channels, 1)
            self.post = nn.ModuleList([ResBlock(channels, kernel, 2 ** i, dropout) for i in range(2)])
            self.norm = nn.GroupNorm(1, channels)
            self.out = nn.Conv1d(channels, n_out, 1)       # with n_out 4 the 4th channel is the auxiliary change-point head (--aux_boundary)

        def forward(self, x):          # x [B, n_in, T] → logits [B, 3, T]
            h = self.inp(x)
            for b in self.pre:
                h = b(h)
            t = self.pool(h)                                   # [B, C, ceil(T / s)]
            t = t + self.pos(t)
            t = self.enc(t.transpose(1, 2)).transpose(1, 2)    # [B, C, ceil(T / s)]
            t = F_.interpolate(t, size=h.shape[2], mode="linear", align_corners=False)
            h = h + self.mix(t)
            for b in self.post:
                h = b(h)
            return self.out(self.norm(h))

    core = {"gru": BiGRU, "xf": ConvXF}.get(arch, TCN)()
    if not pset:
        return core
    import torch

    class PSetWrap(nn.Module):
        """Each of the K players goes through the same small network; inside each team the outputs are summed with attention weights, the highest attention score of each team is appended, and the result is concatenated after the other columns and fed to the backbone.
        Column 0 of every player is the team (home 1, away -1, missing 0); it is only used to split the teams and does not enter the small network, so exchanging home and away exactly exchanges the channels of the two teams"""
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv1d(f_ - 1, 32, 1), nn.GELU(), nn.Conv1d(32, E_, 1), nn.GELU())
            self.att = nn.Conv1d(E_, 1, 1)
            self.core = core

        def forward(self, x):          # x [B, n_raw, T]
            B, _, T = x.shape
            xm = x[:, :n_raw - n_ps]
            xp = x[:, n_raw - n_ps:].reshape(B, K_, f_, T)
            sign = xp[:, :, 0]                                              # [B, K, T]
            h = self.enc(xp[:, :, 1:].reshape(B * K_, f_ - 1, T))           # [B*K, E, T]
            sc = self.att(h).reshape(B, K_, T)
            h = h.reshape(B, K_, E_, T)
            outs = [xm]
            for sgn in (1.0, -1.0):
                m = (sign * sgn) > 0.5
                am = sc.masked_fill(~m, -1e4)
                w = torch.softmax(am, dim=1) * m
                outs.append((w.unsqueeze(2) * h).sum(1))                    # [B, E, T]
                outs.append((am.max(dim=1, keepdim=True).values) * m.any(dim=1, keepdim=True))   # [B, 1, T]
            return self.core(torch.cat(outs, dim=1))

    class PSetPoint(nn.Module):
        """Per-player scoring output structure (--pset_point). The backbone only sees the other columns and outputs Hc_ context channels; each of the K players nearest to the ball goes through the same small network,
        and its output is concatenated with the context to give one score; the context also gives three more scores (home team token, away team token, dead ball). A softmax runs over the K players and the three tokens together; the home team probability is the sum over the home players and the home team token,
        and likewise for the away team. The three output channels are the logs of these three sums (not normalised; after a softmax they are exactly the three class probabilities), so the training and inference code needs no change; the auxiliary heads are read linearly from the context.
        Column 0 of every player is the team (home 1, away -1, missing 0); it is only used to split the teams and does not enter the small network"""
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv1d(f_ - 1, 32, 1), nn.GELU(), nn.Conv1d(32, E_, 1), nn.GELU())
            self.core = core
            self.act = nn.GELU()
            self.score = nn.Sequential(nn.Conv1d(E_ + Hc_, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1))
            self.tok = nn.Conv1d(Hc_, 3, 1)
            self.extra = nn.Conv1d(Hc_, n_user - 3, 1) if n_user > 3 else None

        def forward(self, x):          # x [B, n_raw, T] → [B, n_user, T]
            B, _, T = x.shape
            xp = x[:, n_raw - n_ps:].reshape(B, K_, f_, T)
            sign = xp[:, :, 0]                                              # [B, K, T]
            hg = self.act(self.core(x[:, :n_raw - n_ps]))                   # [B, Hc, T]
            e = self.enc(xp[:, :, 1:].reshape(B * K_, f_ - 1, T))           # [B*K, E, T]
            hr = hg.unsqueeze(1).expand(B, K_, Hc_, T).reshape(B * K_, Hc_, T)
            sc = self.score(torch.cat([e, hr], dim=1)).reshape(B, K_, T)
            tok = self.tok(hg)                                              # [B, 3, T], home team token, away team token, dead ball
            lh = torch.logsumexp(torch.cat([sc.masked_fill(~(sign > 0.5), -1e4), tok[:, 0:1]], dim=1), dim=1)
            la = torch.logsumexp(torch.cat([sc.masked_fill(~(sign < -0.5), -1e4), tok[:, 1:2]], dim=1), dim=1)
            out = torch.stack([lh, la, tok[:, 2]], dim=1)
            return out if self.extra is None else torch.cat([out, self.extra(hg)], dim=1)

    return PSetPoint() if ps_mode == "point" else PSetWrap()


def drop_cols(X, names, pattern):
    """Drop the feature columns whose name matches the regular expression pattern (--drop_feats); the input is returned unchanged when pattern is empty"""
    if not pattern:
        return X, names
    import re
    keep = [i for i, n in enumerate(names) if not re.search(pattern, n)]
    return X[:, keep], [names[i] for i in keep]


def standardize(X, mu, sd):
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(Z, -10.0, 10.0).astype(np.float32)


def block_ranges(M):
    """Return [(s, e)], the frame range of each half, e exclusive. The blocks of the adapters are (first frame, last frame) with both ends included, so one is added to the last frame"""
    return [(int(M["blocks"][k][0]), int(M["blocks"][k][1]) + 1) for k in sorted(M["blocks"])]


def predict_game(model, Z, M, device, chunk, aux_store=None):
    """Inference on whole blocks; frames outside the blocks get uniform probabilities. Blocks that are too long are split into overlapping chunks and stitched. When aux_store is an array of length T, the sigmoid of the 4th channel is written into it"""
    import torch
    T = len(Z)
    P = np.full((T, 3), 1.0 / 3, np.float32)
    model.eval()
    with torch.no_grad():
        for s, e in block_ranges(M):
            seg = Z[s:e]
            n = len(seg)
            if n == 0:
                continue
            out = np.zeros((n, 3), np.float32)
            pos = 0
            ov = 1024
            while pos < n:
                a = max(0, pos - ov)
                b = min(n, pos + chunk + ov)
                x = torch.from_numpy(np.ascontiguousarray(seg[a:b].T))[None].to(device)
                raw = model(x)[0].float().cpu().numpy().T      # [b-a, 3 or 4]
                lg = raw[:, :3]                                # with auxiliary heads only the first 3 channels are used
                lo, hi = pos - a, min(b, pos + chunk) - a
                out[pos:pos + (hi - lo)] = lg[lo:hi]
                if aux_store is not None:
                    aux_store[s + pos:s + pos + (hi - lo)] = 1.0 / (1.0 + np.exp(-raw[lo:hi, 3]))
                pos += chunk
            m = out.max(axis=1, keepdims=True)
            ex = np.exp(out - m)
            P[s:e] = ex / ex.sum(axis=1, keepdims=True)
    return P


def _win(Z, s, e, nz):
    """Take the window [s, e). When nz is None, Z is already standardised over the whole match; when nz=(mu, sd), Z holds the raw features and only the window is standardised (--lazy_norm).
    standardize works element by element, so standardising the window gives bit-identical results to slicing the standardised match; what is saved is the memory of one full copy of Z per fold"""
    w = Z[s:e]
    return w if nz is None else standardize(w, nz[0], nz[1])


def sample_batch(rng, items, window, batch, with_ball=False):
    """items is [(Z, y, (s, e), w, nz)]. Windows are drawn in proportion to the block length times the weight w (1, or --pseudo_w for the pseudo-labelled copy); short blocks are taken whole and zero padded.
    For nz see _win. Returns X [B, F, W] and y [B, W] (-1 is excluded from the loss). With with_ball, items carry a 6th element (T, 2) with the clean ball position, and B [B, 2, W] is also returned, NaN where the ball is missing"""
    F_ = items[0][0].shape[1]
    X = np.zeros((batch, F_, window), np.float32)
    Y = np.full((batch, window), -1, np.int64)
    B = np.full((batch, 2, window), np.nan, np.float32) if with_ball else None
    lens = np.array([(it[2][1] - it[2][0]) * it[3] for it in items], float)
    pr = lens / lens.sum()
    for i in range(batch):
        it = items[rng.choice(len(items), p=pr)]
        Z, y, (s, e), _, nz = it[:5]
        n = e - s
        if n <= window:
            X[i, :, :n] = _win(Z, s, e, nz).T
            Y[i, :n] = y[s:e]
            if with_ball:
                B[i, :, :n] = it[5][s:e].T
        else:
            a = s + int(rng.integers(0, n - window + 1))
            X[i] = _win(Z, a, a + window, nz).T
            Y[i] = y[a:a + window]
            if with_ball:
                B[i] = it[5][a:a + window].T
    return (X, Y, B) if with_ball else (X, Y)


def train_fold(train_sets, val_set, n_in, a, device, seed, log_prefix):
    """Returns (model, training record). The elements of train_sets are {"Z", "y", "M"}; val_set may be None"""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(n_in, a.channels, a.blocks, a.kernel, a.dropout, a.arch, a.rnn_layers, 3 + (a.aux_boundary > 0) + (a.aux_next > 0) + 2 * (a.aux_ball > 0), a.xf_layers, a.xf_heads, a.xf_stride, pset=a.pset_arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    items = []
    for st in train_sets:
        for s, e in block_ranges(st["M"]):
            if e - s > 50:
                items.append((st["Z"], st["y"], (s, e), float(st.get("w", 1.0)), st.get("norm")) + ((st["ball"],) if a.aux_ball > 0 else ()))
    warm = min(100, a.steps // 10)
    best = (float("inf"), None, 0)
    hist = []

    def val_ce():
        if val_set is None:
            return float("nan"), float("nan")
        P = predict_game(model, val_set["Z"], val_set["M"], device, a.chunk)
        m = val_set["y"] >= 0
        p = np.clip(P[m, :][np.arange(m.sum()), val_set["y"][m]], 1e-6, 1)
        acc = float((P[m].argmax(1) == val_set["y"][m]).mean())
        return float(-np.log(p).mean()), acc

    t_fold = time.time()
    ema_sd = {k: v.detach().clone() for k, v in model.state_dict().items()} if a.ema > 0 else None   # --ema
    model.train()
    for step in range(1, a.steps + 1):
        lr = a.lr * (step / warm if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, a.steps - warm))))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        if a.aux_ball > 0:
            X, Y, Bb = sample_batch(rng, items, a.window, a.batch, True)
        else:
            X, Y = sample_batch(rng, items, a.window, a.batch)
        if a.feat_drop > 0:                                  # --feat_drop. For every sample whole feature columns are set to 0 at random (the mean after standardisation), without rescaling
            X = X * (rng.random((X.shape[0], X.shape[1], 1)) >= a.feat_drop).astype(X.dtype)
        if a.time_mask:                                      # --time_mask n:L. Every sample gets n masked spans of 1 to L frames each; all features inside a span are set to 0 and the labels are untouched
            tm_n, tm_l = (int(v) for v in a.time_mask.split(":"))
            for i_ in range(X.shape[0]):
                for _ in range(tm_n):
                    L_ = int(rng.integers(1, tm_l + 1)); a_ = int(rng.integers(0, max(1, X.shape[2] - L_)))
                    X[i_, :, a_:a_ + L_] = 0
        xb = torch.from_numpy(X).to(device)
        yb = torch.from_numpy(Y).to(device)
        logits = model(xb)                                   # [B, 3, W] ([B, 4, W] with an auxiliary head)
        if a.aux_boundary > 0:                               # --aux_boundary. The 4th channel predicts whether there is a change point within --aux_win frames; its binary loss times this coefficient is added to the total loss
            ch2 = (Y[:, 1:] != Y[:, :-1]) & (Y[:, 1:] >= 0) & (Y[:, :-1] >= 0)
            near2 = np.zeros(Y.shape, bool)
            ib2, it2 = np.nonzero(ch2)
            for off in range(-a.aux_win, a.aux_win + 1):
                near2[ib2, np.clip(it2 + 1 + off, 0, Y.shape[1] - 1)] = True
            tgt2 = torch.from_numpy(near2.astype(np.float32)).to(device)
            val2 = (yb >= 0).float()
            aux_loss = (F.binary_cross_entropy_with_logits(logits[:, 3], tgt2, reduction="none") * val2).sum() / torch.clamp(val2.sum(), min=1.0)
        if a.aux_next > 0:                                   # --aux_next. The last channel predicts whether the next possession frame from this frame on belongs to the away team; frames with no later possession frame in the window are excluded
            nW = Y.shape[1]; pos_ = np.arange(nW)
            is_p = (Y == 0) | (Y == 1)
            nx_i = np.minimum.accumulate(np.where(is_p, pos_[None, :], nW)[:, ::-1], axis=1)[:, ::-1]
            has_ = (nx_i < nW) & (Y >= 0)
            tgt3 = np.zeros(Y.shape, np.float32)
            ib3 = np.arange(Y.shape[0])[:, None].repeat(nW, axis=1)
            tgt3[has_] = (Y[ib3[has_], nx_i[has_]] == 1).astype(np.float32)
            val3 = torch.from_numpy(has_.astype(np.float32)).to(device)
            next_loss = (F.binary_cross_entropy_with_logits(logits[:, 3 + (a.aux_boundary > 0)], torch.from_numpy(tgt3).to(device), reduction="none") * val3).sum() / torch.clamp(val3.sum(), min=1.0)
        if a.aux_ball > 0:                                   # --aux_ball. The last two channels predict the clean ball position (x along the attacking direction of the home team, and y, in units of 10 m), only on graded frames where the clean data has the ball
            i_b = 3 + (a.aux_boundary > 0) + (a.aux_next > 0)
            ok_b = np.isfinite(Bb).all(axis=1) & (Y >= 0)
            tb = torch.from_numpy(np.nan_to_num(Bb, nan=0.0)).to(device)
            vb = torch.from_numpy(ok_b.astype(np.float32)).to(device)
            ball_loss = (F.smooth_l1_loss(logits[:, i_b:i_b + 2], tb, reduction="none").sum(1) * vb).sum() / torch.clamp(vb.sum(), min=1.0)
        if a.aux_boundary > 0 or a.aux_next > 0 or a.aux_ball > 0:
            logits = logits[:, :3]
        if a.trans_weight > 0:                               # --trans_weight. The loss weight within --trans_win frames of a change point is multiplied by 1 plus this value
            ch = (Y[:, 1:] != Y[:, :-1]) & (Y[:, 1:] >= 0) & (Y[:, :-1] >= 0)
            near = np.zeros(Y.shape, bool)
            idx_b, idx_t = np.nonzero(ch)
            for off in range(-a.trans_win, a.trans_win + 1):
                tt_ = np.clip(idx_t + 1 + off, 0, Y.shape[1] - 1)
                near[idx_b, tt_] = True
            wt = torch.from_numpy((1.0 + a.trans_weight * near).astype(np.float32)).to(device)
            ce = F.cross_entropy(logits, yb, ignore_index=-1, label_smoothing=a.label_smoothing, reduction="none")
            valid = (yb >= 0).float()
            loss = (ce * wt * valid).sum() / torch.clamp((wt * valid).sum(), min=1.0)
        else:
            loss = F.cross_entropy(logits, yb, ignore_index=-1, label_smoothing=a.label_smoothing)
        if a.aux_boundary > 0:
            loss = loss + a.aux_boundary * aux_loss
        if a.aux_next > 0:
            loss = loss + a.aux_next * next_loss
        if a.aux_ball > 0:
            loss = loss + a.aux_ball * ball_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema_sd is not None:
            with torch.no_grad():
                for k_, v_ in model.state_dict().items():
                    if v_.dtype.is_floating_point:
                        ema_sd[k_].mul_(a.ema).add_(v_.detach(), alpha=1 - a.ema)
                    else:
                        ema_sd[k_].copy_(v_)
        if step % a.eval_every == 0 or step == a.steps:
            ce, acc = val_ce()
            model.train()
            hist.append({"step": step, "train_loss": float(loss.item()), "val_ce": ce, "val_acc": acc, "elapsed_s": time.time() - t_fold})
            print("%s step %d/%d  train loss %.4f  validation cross entropy %.4f  validation per frame accuracy %.4f  time %.0fs" % (
                log_prefix, step, a.steps, loss.item(), ce, acc, time.time() - t_fold), flush=True)
            crit = ce if a.select_by == "ce" else -acc      # checkpoint criterion, lowest validation cross entropy or highest validation per frame accuracy (--select_by acc)
            if val_set is None or crit < best[0]:
                best = (crit, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, step)
    if ema_sd is not None:                                  # use the exponential moving average of the weights instead of the weights of the last step (only with --no_inner_val)
        best = (best[0], {k: v.detach().cpu().clone() for k, v in ema_sd.items()}, a.steps)
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, {"hist": hist, "best_step": best[2], "best_val_ce": best[0], "fold_s": time.time() - t_fold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, the output directory under $MTAH_RUNS")
    ap.add_argument("--sources", default="metrica,idsse")
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--alive_missing", type=float, default=0.30)
    ap.add_argument("--realistic", action="store_true")
    ap.add_argument("--idsse_min_spell", type=int, default=25)
    ap.add_argument("--touch_set", default="clean")
    ap.add_argument("--touch_feats", action="store_true")
    ap.add_argument("--gap_feats", action="store_true")
    ap.add_argument("--kin_feats", action="store_true", help="add 11 player-ball kinematic features, same as the argument of the same name in mtah_v3")
    ap.add_argument("--setp_feats", action="store_true", help="add 49 set piece positioning feature columns (possession/mtah_features.py setp_features; off by default)")
    ap.add_argument("--shape_feats", action="store_true", help="add 53 team shape behaviour feature columns (possession/mtah_features.py shape_features; off by default)")
    ap.add_argument("--comove_feats", action="store_true", help="add 9 player-ball co-movement features (possession/mtah_features.comove_features; off by default)")
    ap.add_argument("--pset_feats", type=int, default=0, help="add ball-centred per-player features, 11 columns for each of the K players nearest to the ball (possession/mtah_features.pset_features; default 0, off)")
    ap.add_argument("--pset_pool", action="store_true", help="use with --pset_feats; the per-player features are not standardised, the K players go through the same small network and are pooled per team before the backbone (pset in make_model; off by default)")
    ap.add_argument("--pset_point", action="store_true", help="use with --pset_feats; per-player scoring output structure, a softmax over the K players plus three tokens (home team, away team, dead ball), where the probability of a team is the sum over its players and its token (PSetPoint in make_model; off by default, not to be combined with --pset_pool)")
    ap.add_argument("--comove_v2", action="store_true", help="use with --comove_feats; 9 more player-ball co-movement feature columns (multi-scale; off by default)")
    ap.add_argument("--drop_feats", default="", help="drop the feature columns whose name matches this regular expression before they enter the model; empty means off")
    ap.add_argument("--dir_feats", action="store_true", help="add 32 movement direction feature columns (possession/mtah_features.py dir_features; off by default)")
    ap.add_argument("--press_feats", action="store_true", help="add 52 feature columns on the pressing focus and positioning relations (possession/mtah_features.py press_features; off by default)")
    ap.add_argument("--roster_feats", action="store_true", help="add 110 per-player feature columns ordered by position (possession/mtah_features.py roster_features; off by default)")
    ap.add_argument("--setp_clip", action="store_true", help="clip the set piece positioning features to the physical range before they enter the model, and switch these features on (off by default)")
    ap.add_argument("--save_boundary", action="store_true", help="save the sigmoid of the 4th channel (the change-point output of --aux_boundary) as loso_oof_boundary.npz and loso_oof_boundary_eval_<tag>.npz, to serve as the change-point head of the segment decoding (off by default)")
    ap.add_argument("--tta_feats", action="store_true", help="add 4 time-to-arrival features (possession/mtah_features.py tta_features; off by default)")
    ap.add_argument("--lazy_norm", action="store_true", help="standardise the training windows when they are sampled instead of keeping a standardised copy of every match (bit-identical results, saves memory; off by default)")
    ap.add_argument("--slim_m", action="store_true", help="once the features are built, discard the large arrays such as the coordinates in every M and keep only small fields such as blocks and graded (off by default)")
    ap.add_argument("--trans_weight", type=float, default=0.0, help="extra loss weight near change points; the loss weight within --trans_win frames of a change point (two adjacent frames with different labels) is multiplied by 1 plus this value; 0 is off")
    ap.add_argument("--trans_win", type=int, default=12, help="half window of --trans_weight, in frames")
    ap.add_argument("--aux_boundary", type=float, default=0.0, help="add a 4th output channel that predicts whether there is a change point within --aux_win frames; its binary loss is multiplied by this coefficient; 0 is off")
    ap.add_argument("--aux_next", type=float, default=0.0, help="add one more output channel that predicts whether the next possession frame from this frame on belongs to the away team; its binary loss is multiplied by this coefficient; 0 is off")
    ap.add_argument("--aux_ball", type=float, default=0.0, help="add two more output channels that predict the clean ball position as an auxiliary task; the Smooth L1 loss is multiplied by this coefficient; 0 is off")
    ap.add_argument("--aux_win", type=int, default=12, help="half window of --aux_boundary, in frames")
    ap.add_argument("--time_mask", default="", help="during training every window sample gets n masked time spans of 1 to L frames each, with all features inside set to 0; written as n:L (for example 3:75); empty means off")
    ap.add_argument("--feat_drop", type=float, default=0.0, help="during training every feature column of every sample is set to 0 with this probability (0 is the mean after standardisation); 0 is off")
    ap.add_argument("--ema", type=float, default=0.0, help="keep an exponential moving average of the weights during training (decay factor, for example 0.995) and use it instead of the weights of the last step; 0 is off; only with --no_inner_val")
    ap.add_argument("--pseudo_run", default="", help="name of the run that provides pseudo labels (relative to runs/); its loso_oof_proba.npz is read, and in every fold the frames of the held-out match with confidence >= --pseudo_thr enter training with their argmax as label")
    ap.add_argument("--pseudo_thr", type=float, default=0.9)
    ap.add_argument("--pseudo_w", type=float, default=1.0, help="sampling weight of the pseudo-labelled copy (multiplied into the block length)")
    ap.add_argument("--norm", default="train", choices=("train", "per_game"), help="feature standardisation; train uses the mean and standard deviation of the training matches (the default), per_game uses those of the graded frames of each match itself (no labels are used)")
    ap.add_argument("--train_extra_degrade", default="", help="other degradation bench settings that also enter training, a comma separated list of static_wander:ball_smooth[:missing ball share[:position noise[:degradation seed]]], such as 0:0,4:21 or 3:15:0.45 or 3:15:0.3:0.01:1 (training only; the held-out matches still use the main degradation bench)")
    ap.add_argument("--rand_degrade", type=int, default=0, help="randomised degradation; the training copies of Metrica and IDSSE use the randomised degradation of degrade_random, this many copies per match with one parameter set drawn for each (the --augment copies follow their own copy); --noise, --static_wander, --ball_smooth and --alive_missing then have no effect on training; 0 is off")
    ap.add_argument("--rand_bench", type=int, default=0, help="also build this many randomised benchmarks rnd0, rnd1 and so on (degrade_random.draw_bench; the held-out probabilities are saved as loso_oof_proba_eval_rnd{k}.npz); 0 builds none")
    ap.add_argument("--rand_main", action="store_true", help="replace the main benchmark (loso_oof_proba.npz) by the randomised benchmark rnd0; requires --rand_bench")
    ap.add_argument("--static_wander", type=float, default=0.0)
    ap.add_argument("--ball_smooth", type=int, default=0)
    ap.add_argument("--metrica_jitter", type=float, default=0.0)
    ap.add_argument("--impute_sigma", type=float, default=0.0)
    ap.add_argument("--sc_matches", type=int, default=20)
    ap.add_argument("--train_only", default="", help="names of the sources used for training only (comma separated, such as skillcorner; do not list them in --sources as well); these matches enter training in every fold, are never held out and get no benchmark; the seed 0 copy of the fixed degradation bench is used (frames where the ball was already missing stay missing), with its --augment copies (empty by default)")
    ap.add_argument("--drop_train", default="", help="these matches (comma separated match names, such as idsse_J03WMX,metrica1) are left out of training in every fold, while their own fold is still held out and scored as usual; used to measure the loss from having fewer training matches (empty by default)")
    ap.add_argument("--extra_eval", default="", help="extra benchmarks, such as 0:0,4:21 (static_wander:ball_smooth); every fold is trained once and predicts all of them")
    ap.add_argument("--augment", default="", help="training set augmentation, a comma separated list of swap (home and away exchanged), mirror (mirrored along the long axis) and both (mtah_v3.augment_sets); the copies are used for training only")
    ap.add_argument("--arch", default="tcn", choices=("tcn", "gru", "xf"), help="network architecture; tcn is the dilated convolution network, gru a bidirectional GRU, xf a convolution stem plus a Transformer encoder")
    ap.add_argument("--rnn_layers", type=int, default=2, help="number of layers for --arch gru")
    ap.add_argument("--xf_layers", type=int, default=4, help="number of encoder layers for --arch xf")
    ap.add_argument("--xf_heads", type=int, default=4, help="number of attention heads for --arch xf; must divide --channels")
    ap.add_argument("--xf_stride", type=int, default=8, help="downsampling stride for --arch xf (frames per token); a training window of 4096 at stride 8 gives 512 tokens")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8, help="number of dilated convolution blocks, dilation 1,2,4 and so on; 8 blocks with kernel 5 have a receptive field of about plus or minus 510 frames")
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="label smoothing of the cross entropy; 0 is off")
    ap.add_argument("--select_by", default="ce", choices=("ce", "acc"), help="criterion for choosing the checkpoint on the internal validation match; ce is the lowest validation cross entropy (the default), acc the highest validation per frame accuracy")
    ap.add_argument("--chunk", type=int, default=32768, help="number of frames per chunk at inference (plus an overlap of 1024 frames on each side)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_inner_val", action="store_true", help="keep no internal validation match and always train for the full number of steps")
    ap.add_argument("--costs", default="8,32,24", help="Viterbi costs for the quick score of every fold printed by this script; rescore_oof.py rescores the held-out probabilities with the Viterbi decoder")
    ap.add_argument("--smoke", action="store_true", help="smoke test, few steps and a small window")
    ap.add_argument("--check_args", action="store_true")
    a = ap.parse_args()
    if a.ema > 0 and not a.no_inner_val:
        ap.error("--ema requires --no_inner_val")
    if a.rand_main and a.rand_bench < 1:
        ap.error("--rand_main needs --rand_bench")
    if a.check_args:
        print("argument check passed", a.name)
        return
    if a.smoke:
        a.steps, a.window, a.batch, a.eval_every, a.channels, a.blocks = 20, 1024, 4, 10, 32, 4
    import torch
    from possession.mtah_features import PSET_F
    assert not a.pset_pool or a.pset_feats > 0, "--pset_pool requires --pset_feats"
    assert not a.pset_point or (a.pset_feats > 0 and not a.pset_pool), "--pset_point requires --pset_feats and cannot be combined with --pset_pool"
    a.pset_arch = (a.pset_feats, PSET_F) if a.pset_pool else ((a.pset_feats, PSET_F, "point") if a.pset_point else None)
    device = pick_device(a.device)
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "10")))
    rep("device", str(device))
    out_dir = os.path.join(os.path.expanduser(os.environ.get("MTAH_RUNS", "~/Data/more_than_a_hack_2027/runs")), a.name)
    os.makedirs(out_dir, exist_ok=True)

    names_ = a.sources.split(",")
    sets = build_sources(names_, max(1, a.rand_degrade), a.noise, a.alive_missing, a.idsse_min_spell, a.realistic, a.sc_matches, a.metrica_jitter,
                         a.impute_sigma, a.static_wander, a.ball_smooth, augment=a.augment, rand_mode="train" if a.rand_degrade else "")
    if a.rand_degrade:                                     # copies 1 and above are used for training only, like the augmented copies (their seed is not 0), and are held out together with their match
        print("[progress] randomised degradation, %d copies per match, %d sets in total with augmentation" % (a.rand_degrade, len(sets)), flush=True)
    # --train_extra_degrade. Degraded copies from other degradation bench settings also enter training; like the augmented copies they are used for training only and are held out together with their match
    for k, p in enumerate([p for p in a.train_extra_degrade.split(",") if p]):
        f_ = p.split(":")
        xsw, xbs = float(f_[0]), int(f_[1])
        xam = float(f_[2]) if len(f_) > 2 else a.alive_missing     # third item, the share of live ball frames with the ball missing in this copy; --alive_missing when not given
        xno = float(f_[3]) if len(f_) > 3 else a.noise             # fourth item, the position noise of this copy; --noise when not given
        xsd = int(f_[4]) if len(f_) > 4 else 0                     # fifth item, the random seed of the degradation of this copy (sampling of ball gaps, noise and static wander); 0 when not given, the same as the main degradation bench
        more = build_sources(names_, xsd + 1, xno, xam, a.idsse_min_spell, a.realistic, a.sc_matches, a.metrica_jitter,
                             a.impute_sigma, xsw, xbs, augment=a.augment)
        more = [st for st in more if st["seed"] % 1000 == xsd]      # keep only the copy with this seed and its augmented copies (the seed of an augmented copy has a multiple of 1000 added)
        for st in more:
            st["seed"] = st["seed"] + 10000 * (k + 1)
        sets += more
        print("[progress] degraded copies added to training, sw%g bs%d missing ball %g noise %g degradation seed %d, %d sets" % (xsw, xbs, xam, xno, xsd, len(more)), flush=True)
    to_sets = []                                           # --train_only
    if a.train_only:
        to_names = [p for p in a.train_only.split(",") if p]
        assert not (set(to_names) & set(names_)), "the sources in --train_only must not be listed in --sources as well"
        assert a.aux_ball == 0 and not a.pseudo_run, "--train_only cannot be combined with --aux_ball or --pseudo_run"
        to_sets = build_sources(to_names, 1, a.noise, a.alive_missing, a.idsse_min_spell, a.realistic, a.sc_matches, a.metrica_jitter,
                                a.impute_sigma, a.static_wander, a.ball_smooth, augment=a.augment)
        print("[progress] training-only sources %s, %d matches, %d sets in total with augmentation" % (a.train_only, len({st["game"] for st in to_sets}), len(to_sets)), flush=True)
    feat_names = None
    clean_cache = {}

    def clean_ball(g):
        """Clean ball position (T, 2), the training target of --aux_ball; public sources only"""
        if g not in clean_cache:
            if g.startswith("metrica"):
                from possession.adapters.metrica_sample import load_game
                Mc, _ = load_game("~/Data/alt_training/metrica_sample", int(g[-1]))
            elif g.startswith("idsse_"):
                from mtah_v3 import IDSSE_ROOT
                from possession.adapters.idsse import load_match
                Mc, _ = load_match(IDSSE_ROOT, g[len("idsse_"):], min_spell=0)
            else:
                raise ValueError("--aux_ball only supports metrica and idsse, got %s" % g)
            clean_cache[g] = np.asarray(Mc["ball_xy"], np.float32)
        return clean_cache[g]
    for st in sets + to_sets:
        if a.aux_ball > 0:                                     # x along the attacking direction of the home team, and y, in units of 10 m; in copies with home and away exchanged the attacking direction flips as well
            from possession.mtah_features import _attack_dir
            cb = clean_ball(st["game"])
            assert len(cb) == len(st["M"]["t_ms"]), st["game"]
            dirs_ = _attack_dir(st["M"], st["M"]["blocks"])
            hd_ = np.array([dirs_.get(int(x_), 1) for x_ in st["M"]["period"]])
            L_, W_ = st["M"]["pitch"]
            st["ball_t"] = np.column_stack([np.where(hd_ > 0, cb[:, 0], L_ - cb[:, 0]) / 10.0, cb[:, 1] / 10.0]).astype(np.float32)
        X, feat_names = build_features(st["M"], gap_feats=a.gap_feats, touch_feats=a.touch_feats, touch_set=a.touch_set, kin_feats=a.kin_feats, tta_feats=a.tta_feats, setp_feats=("clip" if a.setp_clip else a.setp_feats), shape_feats=a.shape_feats, press_feats=a.press_feats, roster_feats=a.roster_feats, dir_feats=a.dir_feats, comove_feats=("v2" if (a.comove_feats and a.comove_v2) else a.comove_feats), pset_feats=a.pset_feats)
        X, feat_names = drop_cols(X, feat_names, a.drop_feats)
        st["X"] = X.astype(np.float32)
        st["y"] = (np.asarray(st["gt"]) - 1).astype(np.int64)      # 0 H, 1 A, 2 D, -1 unlabelled
    if a.slim_m:                                               # --slim_m. Once the features are built, the large arrays in every M are discarded; only small fields such as blocks and graded are used afterwards
        for st in sets + to_sets:
            for k_ in ("player_xy", "ball_xy", "ball_speed", "ball_z", "t_ms", "tags", "slot_team"):
                st["M"].pop(k_, None)
    print("[progress] features built, %d sets, %d features, time %.0fs" % (len(sets) + len(to_sets), len(feat_names), time.time() - T0), flush=True)
    for st in to_sets:                                         # training-only sources must not have infinities in the features (missing values are NaN and become 0 at standardisation, as for the other sources)
        assert not np.isinf(st["X"]).any(), st["game"]
    extra = [(float(p.split(":")[0]), int(p.split(":")[1])) for p in a.extra_eval.split(",") if p]
    for xsw, xbs in extra:
        tag = "sw%g_bs%d" % (xsw, xbs)
        ev = build_sources(names_, 1, 0.01, a.alive_missing, a.idsse_min_spell, True, a.sc_matches, a.metrica_jitter,
                           a.impute_sigma, xsw, xbs)
        evm = {e["game"]: e for e in ev}
        for st in sets:
            if st["seed"] != 0:
                continue                                   # augmented copies are used for training only, not as benchmarks
            Xx, _ = build_features(evm[st["game"]]["M"], gap_feats=a.gap_feats, touch_feats=a.touch_feats, touch_set=a.touch_set, kin_feats=a.kin_feats, tta_feats=a.tta_feats, setp_feats=("clip" if a.setp_clip else a.setp_feats), shape_feats=a.shape_feats, press_feats=a.press_feats, roster_feats=a.roster_feats, dir_feats=a.dir_feats, comove_feats=("v2" if (a.comove_feats and a.comove_v2) else a.comove_feats), pset_feats=a.pset_feats)
            Xx, _ = drop_cols(Xx, _, a.drop_feats)
            st.setdefault("X_extra", {})[tag] = Xx.astype(np.float32)
        del ev, evm
        print("[progress] extra benchmark %s built, time %.0fs" % (tag, time.time() - T0), flush=True)
    for k_r in range(a.rand_bench):
        # randomised benchmarks, built the same way as mtah_v3 --rand_bench; with --rand_main, rnd0 is also the main benchmark
        tag = "rnd%d" % k_r
        ev = build_sources(names_, 1, 0.01, a.alive_missing, a.idsse_min_spell, True, a.sc_matches, a.metrica_jitter,
                           a.impute_sigma, 0.0, 0, rand_mode=tag)
        evm = {e["game"]: e for e in ev}
        for st in sets:
            if st["seed"] != 0:
                continue
            Xx, _ = build_features(evm[st["game"]]["M"], gap_feats=a.gap_feats, touch_feats=a.touch_feats, touch_set=a.touch_set, kin_feats=a.kin_feats, tta_feats=a.tta_feats, setp_feats=("clip" if a.setp_clip else a.setp_feats), shape_feats=a.shape_feats, press_feats=a.press_feats, roster_feats=a.roster_feats, dir_feats=a.dir_feats, comove_feats=("v2" if (a.comove_feats and a.comove_v2) else a.comove_feats), pset_feats=a.pset_feats)
            Xx, _ = drop_cols(Xx, _, a.drop_feats)
            st.setdefault("X_extra", {})[tag] = Xx.astype(np.float32)
            if k_r == 0 and a.rand_main:
                st["X_main"] = st["X_extra"][tag]
        del ev, evm
        print("[progress] randomised benchmark %s built%s, time %.0fs" % (tag, " (also the main benchmark)" if k_r == 0 and a.rand_main else "", time.time() - T0), flush=True)
    games = sorted(st["game"] for st in sets if st["seed"] == 0)
    by_game = {st["game"]: st for st in sets if st["seed"] == 0}
    aug_sets = [st for st in sets if st["seed"] != 0]           # copies made by --augment; they carry the name of their match and are excluded together with it when it is held out
    rep("n_aug_sets", len(aug_sets))
    rep("n_train_only_sets", len(to_sets))
    rep("n_games", len(games))
    rep("n_features", len(feat_names))
    build_s = time.time() - T0
    rep("build_s", build_s)
    costs = tuple(float(x) for x in a.costs.split(","))
    print("[progress] starting leave-one-match-out, %d folds" % len(games), flush=True)

    pseudo_oof = None
    if a.pseudo_run:
        pseudo_oof = dict(np.load(os.path.join(os.path.expanduser(os.environ.get("MTAH_RUNS", "~/Data/more_than_a_hack_2027/runs")), a.pseudo_run, "loso_oof_proba.npz")))
        assert all(pseudo_oof[g].shape[0] == by_game[g]["X"].shape[0] for g in games), "the out-of-fold probabilities of the pseudo label run do not match the lengths of the matches here"
    oof, oof_x, folds = {}, {}, {}
    oof_b, oof_bx = {}, {}                                   # --save_boundary
    drop_train = {p for p in a.drop_train.split(",") if p}   # --drop_train; with an empty set the filter below changes nothing
    assert drop_train <= set(games), "--drop_train contains unknown match names %s" % sorted(drop_train - set(games))
    if drop_train:
        rep("n_drop_train", len(drop_train))
    for n, g in enumerate(games):
        tr_games = [x for x in games if x != g and x not in drop_train]
        val_g = None if a.no_inner_val else tr_games[n % len(tr_games)]
        fit_games = [x for x in tr_games if x != val_g]
        Xtr = np.concatenate([by_game[x]["X"][by_game[x]["y"] >= 0] for x in fit_games])
        mu = np.nanmean(Xtr, axis=0)
        sd = np.nanstd(Xtr, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        del Xtr
        if a.pset_pool or a.pset_point:                          # per-player features are not standardised (they are already scaled to about 1 when built), and the team column must keep its value
            pc = [i for i, nm in enumerate(feat_names) if nm.startswith("pset")]
            assert pc == list(range(len(feat_names) - a.pset_feats * PSET_F, len(feat_names))), "the per-player features must be the last columns"
            mu[pc] = 0.0; sd[pc] = 1.0

        def Zs(X, M):                                            # --norm per_game. Every match uses the mean and standard deviation of its own graded frames; no labels are used
            if a.norm != "per_game":
                return standardize(X, mu, sd)
            gm = np.asarray(M["graded"], bool)
            Xg = X[gm] if gm.any() else X
            m_, s_ = np.nanmean(Xg, axis=0), np.nanstd(Xg, axis=0)
            return standardize(X, np.where(np.isfinite(m_), m_, 0.0), np.where(np.isfinite(s_) & (s_ > 1e-6), s_, 1.0))
        def norm_of(X, M):                                       # for --lazy_norm, the standardisation parameters of this set, identical to those used in Zs
            if a.norm != "per_game":
                return (mu, sd)
            gm = np.asarray(M["graded"], bool)
            Xg = X[gm] if gm.any() else X
            m_, s_ = np.nanmean(Xg, axis=0), np.nanstd(Xg, axis=0)
            return (np.where(np.isfinite(m_), m_, 0.0), np.where(np.isfinite(s_) & (s_ > 1e-6), s_, 1.0))
        if a.lazy_norm:
            train_sets = [{"Z": by_game[x]["X"], "norm": norm_of(by_game[x]["X"], by_game[x]["M"]), "y": by_game[x]["y"], "M": by_game[x]["M"], "w": 1.0, "ball": by_game[x].get("ball_t")} for x in fit_games]
            train_sets += [{"Z": st["X"], "norm": norm_of(st["X"], st["M"]), "y": st["y"], "M": st["M"], "w": 1.0, "ball": st.get("ball_t")} for st in aug_sets if st["game"] in fit_games]
            train_sets += [{"Z": st["X"], "norm": norm_of(st["X"], st["M"]), "y": st["y"], "M": st["M"], "w": 1.0, "ball": None} for st in to_sets]   # --train_only, used in every fold
        else:
            train_sets = [{"Z": Zs(by_game[x]["X"], by_game[x]["M"]), "y": by_game[x]["y"], "M": by_game[x]["M"], "w": 1.0, "ball": by_game[x].get("ball_t")} for x in fit_games]
            train_sets += [{"Z": Zs(st["X"], st["M"]), "y": st["y"], "M": st["M"], "w": 1.0, "ball": st.get("ball_t")} for st in aug_sets if st["game"] in fit_games]
            train_sets += [{"Z": Zs(st["X"], st["M"]), "y": st["y"], "M": st["M"], "w": 1.0, "ball": None} for st in to_sets]
        val_set = None if val_g is None else {"Z": Zs(by_game[val_g]["X"], by_game[val_g]["M"]), "y": by_game[val_g]["y"], "M": by_game[val_g]["M"]}
        if pseudo_oof is not None:                               # --pseudo_run. The held-out match enters training with pseudo labels taken from the out-of-fold probabilities of another run; its true labels are not touched
            Pp = pseudo_oof[g]
            yp = Pp.argmax(1).astype(np.int64)
            yp[(Pp.max(1) < a.pseudo_thr) | ~np.asarray(by_game[g]["M"]["graded"], bool)] = -1
            train_sets.append({"Z": by_game[g]["X"], "norm": norm_of(by_game[g]["X"], by_game[g]["M"]), "y": yp, "M": by_game[g]["M"], "w": a.pseudo_w} if a.lazy_norm else
                              {"Z": Zs(by_game[g]["X"], by_game[g]["M"]), "y": yp, "M": by_game[g]["M"], "w": a.pseudo_w})
            rec_pseudo = {"pseudo_frac": float((yp >= 0).mean()), "pseudo_acc": float((yp[yp >= 0] == by_game[g]["y"][yp >= 0]).mean()) if (yp >= 0).any() else None}
        else:
            rec_pseudo = {}
        model, rec = train_fold(train_sets, val_set, len(feat_names), a, device, a.seed * 1000 + n, "  held out %s" % g)
        held = by_game[g]
        Zm = held.get("X_main", held["X"])
        sb = np.zeros(len(Zm), np.float32) if a.save_boundary else None
        P = predict_game(model, Zs(Zm, held["M"]), held["M"], device, a.chunk, sb)   # X_main is the main benchmark of --rand_main
        oof[g] = P
        if a.save_boundary:
            oof_b[g] = sb
        for tag, Xx in held.get("X_extra", {}).items():
            sb = np.zeros(len(Xx), np.float32) if a.save_boundary else None
            oof_x.setdefault(tag, {})[g] = predict_game(model, Zs(Xx, held["M"]), held["M"], device, a.chunk, sb)
            if a.save_boundary:
                oof_bx.setdefault(tag, {})[g] = sb
        gr = held["M"]["graded"]
        pred = to_hadx(viterbi3_tv(np.log(np.clip(P, 1e-6, 1)), held["M"]["blocks"], *costs))
        pred[~gr] = 0
        r = evaluate(codes_from_hadx(held["gt"]), codes_from_hadx(pred))
        m = held["y"] >= 0
        acc = float((P[m].argmax(1) == held["y"][m]).mean())
        rec.update(rec_pseudo)
        rec.update({"inner_val_game": val_g, "spell_f1_train_labels": r["spell_f1"], "macro_f1": r["macro_f1"], "frame_acc": acc})
        folds[g] = rec
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "mu": mu, "sd": sd, "feature_names": feat_names,
                    "arch": {"n_in": len(feat_names), "channels": a.channels, "blocks": a.blocks, "kernel": a.kernel, "dropout": a.dropout, "arch": a.arch, "rnn_layers": a.rnn_layers, "n_out": 3 + int(a.aux_boundary > 0) + int(a.aux_next > 0) + 2 * int(a.aux_ball > 0), "xf_layers": a.xf_layers, "xf_heads": a.xf_heads, "xf_stride": a.xf_stride, **({"pset": list(a.pset_arch)} if a.pset_arch else {})}, "norm": a.norm},
                   os.path.join(out_dir, "seq_%s.pt" % g))
        el = time.time() - T0
        per = (el - build_s) / (n + 1)
        print("[progress] held out %d/%d done (%s)  spellF1 on training labels %.4f  per frame accuracy %.4f  internal validation %s best step %s  elapsed %.0fs  estimated remaining %.0fs" % (
            n + 1, len(games), g, r["spell_f1"], acc, val_g, rec["best_step"], el, per * (len(games) - n - 1)), flush=True)
        del model, train_sets, val_set
        if device.type == "mps":
            torch.mps.empty_cache()

    np.savez_compressed(os.path.join(out_dir, "loso_oof_proba.npz"), **{g: oof[g].astype(np.float32) for g in games})
    for tag, d in oof_x.items():
        np.savez_compressed(os.path.join(out_dir, "loso_oof_proba_eval_%s.npz" % tag), **{g: d[g].astype(np.float32) for g in games})
    if a.save_boundary:                                      # same file names as mtah_v3 --boundary
        np.savez_compressed(os.path.join(out_dir, "loso_oof_boundary.npz"), **{g: oof_b[g] for g in games})
        for tag, d in oof_bx.items():
            np.savez_compressed(os.path.join(out_dir, "loso_oof_boundary_eval_%s.npz" % tag), **{g: d[g] for g in games})
    rep("loso_mean_spell_f1_train_labels", float(np.mean([folds[g]["spell_f1_train_labels"] for g in games])))
    rep("loso_mean_frame_acc", float(np.mean([folds[g]["frame_acc"] for g in games])))
    for g in games:
        rep("loso_spell_f1_train_labels_%s" % g, folds[g]["spell_f1_train_labels"])
    OUT["folds"] = folds
    meta = {"feature_names": feat_names, "costs": costs, "sources": a.sources, "seeds": 1, "noise": a.noise, "alive_missing": a.alive_missing,
            "k": 0.0, "p0": 0.04, "gap_feats": a.gap_feats, "touch_feats": a.touch_feats, "kin_feats": a.kin_feats, "touch_set": a.touch_set, "spell_weight": 0.0,
            "min_spell_all": 0, "min_spell_src": "", "realistic": a.realistic, "train_only": a.train_only, "sc_matches": a.sc_matches, "train_stride": 1,
            "metrica_jitter": a.metrica_jitter, "eval_realistic": False, "idsse_min_spell": a.idsse_min_spell,
            "impute_sigma": a.impute_sigma, "hgb": None,
            "static_wander": a.static_wander, "ball_smooth": a.ball_smooth, "eval_static_wander": None,
            "eval_ball_smooth": None, "extra_eval": a.extra_eval, "model_kind": "seq_tcn",
            "augment": a.augment,
            "seq": {k: getattr(a, k) for k in ("channels", "blocks", "kernel", "dropout", "window", "batch", "steps", "eval_every", "lr", "wd", "seed", "no_inner_val", "label_smoothing", "select_by", "arch", "rnn_layers", "train_extra_degrade", "norm", "pseudo_run", "pseudo_thr", "pseudo_w", "trans_weight", "trans_win", "ema", "feat_drop", "time_mask", "aux_boundary", "aux_win")}}
    if a.drop_train:                                         # recorded only when used
        meta["drop_train"] = a.drop_train
    if a.rand_degrade or a.rand_bench:                       # recorded only when used
        meta.update({"rand_degrade": int(a.rand_degrade), "rand_bench": int(a.rand_bench), "rand_main": bool(a.rand_main)})
    joblib.dump({**meta, "model": None, "boundary_model": None}, os.path.join(out_dir, "model.joblib"))
    versions = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": str(device)}
    json.dump(versions, open(os.path.join(out_dir, "versions.json"), "w"), indent=1)
    rep("runtime_s", time.time() - T0)
    json.dump(OUT, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print("written to " + out_dir + " (leave-one-match-out results only; rescore_oof.py rescores these probabilities with the Viterbi decoder, spell_decode.py with the segment decoder)")


if __name__ == "__main__":
    main()
