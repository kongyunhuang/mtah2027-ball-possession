"""
spell_decode, decoding aimed directly at spellF1, candidate segment scoring plus semi-Markov selection
============================
The three-state Viterbi decoder only sees per frame probabilities and fixed transition costs. Here a segment is judged as a whole.
  1 Candidate boundaries. The union of the change frames of the six Viterbi cost settings of COST_GRID, the end points of ball gaps, the touch frames where the team changes (touch_events, thresholds chosen by the noise
    of the benchmark) and, with --switch_run, the local maxima of each change-point probability head that reach --switch_tau within plus or minus --switch_hw frames; boundaries within 2 frames of each other are merged into one.
  2 Candidate segments. Every stretch between candidate boundaries that spans 1 to kmax intervals, with one feature row for the home team label and one for the away team label (duration, mean probability of own team, opponent and dead ball,
    per frame argmax shares, share of the segment that each Viterbi setting labels as own team or as dead ball, share of frames with the ball visible, touch rate of each team inside the segment, probabilities in the neighbourhood before and after, origin of the two boundaries;
    with --switch_run 4 more features per change-point head, the highest change-point probability within plus or minus --switch_hw frames of each end, and the maximum and the mean inside the segment).
  3 Scorer. Leave-one-match-out (trained on the other matches, predicts the held-out match, never sees its ground truth). The target is 'labelled as this team, the segment has an IoU of at least τ with a ground-truth possession spell of the same team',
    τ defaults to 0.5, and with several values in --ious one scorer is trained per value and the probabilities are averaged; gradient boosted trees, negatives are subsampled and reweighted.
  4 Decoding. A semi-Markov dynamic programme over the candidate boundaries picks one labelling path that covers the whole block; two adjacent segments cannot both belong to the same team in possession.
    The reward of a possession segment is 'scorer probability minus θ', the reward of a dead-ball segment is γ times the duration in seconds times (mean dead-ball probability minus 0.5); θ and γ are picked on a grid using the matches other than the held-out one,
    and the result for fixed θ 0.3, γ 0 (the first θ and the first γ of the grid when these values are not in it) is reported as a control without selection.
The comparison is paired per match on the raw labels, against the Viterbi baseline (costs of the run) on the same out-of-fold probabilities. Only summary numbers are printed.
The degradation of each match is rebuilt as when the run built its benchmark (load_game_full). Randomised benchmarks rnd{k} use degrade_random; otherwise the fixed degradation of mtah_v3 is used
with the settings of the run (with eval_realistic, realistic ball gaps and noise 0.01); alive_missing, the ball position jitter of Metrica (metrica_jitter), the slow roll of a static ball,
the smoothing of the ball trajectory and the imputation error also follow the run (harness_extra). --label_min_spell makes the scorer targets and the evaluation use IDSSE labels with short spells merged
(0, the raw labels, for the submitted decoder).

Inputs  --run run name (reads loso_oof_proba.npz, model.joblib, metrics.json under $MTAH_RUNS/<run>/, default ~/Data/more_than_a_hack_2027/runs), public sources only
        --scorer hyperparameter overrides for the gradient boosted trees of the scorer, for example max_iter=500,max_leaf_nodes=63 (default SCORER_DEFAULT)
        --feat_set segment feature set. v1 is the base set; v2 appends 20 features to v1 (minimum, standard deviation and share of frames below 0.2 of the own team probability inside the segment, maximum of the 1 second moving average of the opponent probability,
          opponent probability in 3 second windows, touch counts of both teams inside the segment, number of team-switch touches inside the segment, whether the first and the last touch inside the segment belong to the own team, frame distance from both ends to the nearest own and opponent touch, whether the ball is visible at both ends,
          share of frames with the ball visible in the second before and after, position of the segment within the half); v3 is v2 without the 9 touch features (touch counts, number of team-switch touches, team of the first and last touch, distance to the nearest touch),
          which leaves the 11 probability shape and ball visibility features. The evaluation JSON and the bundle record which set was used
        --scorer_seeds number of scorers per IoU target (different random seeds for the negative sampling and the trees), probabilities averaged; default 1
        --src_weight source weights for the scorer training samples, for example metrica=3 (candidate segments of Metrica matches weigh 3 times, IDSSE unchanged); no weighting by default
        --cand_extra one more kind of candidate boundary. touch adds every owned touch frame (not only team-switch touches), argmax adds the change points of the per frame argmax after a 5 frame median filter, both adds both;
          empty by default
        --extra_benches out-of-fold probabilities of other benchmarks of the same run (--extra_eval of mtah_v3 and seq_model). Each match is degraded again with the slow roll and smoothing of that benchmark and candidate segments are built,
          which are only added to the training rows of the scorer (segments of the held-out match stay out); scoring is on the main benchmark only. The scorer then sees the probability shapes under several degradations, which makes it more robust when the training degradation differs from that of the tracking it is applied to
        --score_benches also scores the extra benchmarks, θγ are picked by the mean over the scored benchmarks; --report_benches benchmarks that are only reported, scored by the scorer of the same fold and decoded with the picked θγ
          (benchmark names rnd{k} are the randomised benchmarks of degrade_random; when the run has rand_main the main benchmark is rnd0, and extra or reported benchmarks may be rnd{k} as well)
Outputs  the JSON at --out (spellF1, macroF1, F1 per IoU and hit rate by spell length, per match and averaged), a table on screen;
         --save_pred (on by default) stores the new decoding of every held-out match as an npz, for inspection

Usage
  OMP_NUM_THREADS=1 python -u spell_decode.py --run ens --ious 0.3,0.5,0.7 --label_min_spell 0 --scorer max_iter=500 --thetas 0.4,0.45,0.5,0.55,0.6,0.7 \
      --gammas 0.1,0.15,0.2,0.3,0.4 --extra_benches rnd1,rnd2 --score_benches --report_benches sw3_bs15,sw0_bs0,sw4_bs21 --switch_run bnd_loso,seq_avg \
      --switch_main_tag rnd0 --save_pred ens_decode_pred.npz --out ens_decode.json
  add --games metrica1,idsse_J03WOH to try the pipeline on a few matches

Last modified 2026-11-15
"""
import argparse
import json
import os
import re
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from mtah_v2 import to_hadx, viterbi3  # noqa: E402
from possession.mtah_features import TOUCH_SETS  # noqa: E402
from possession.spell_metrics import IOUS, binned_hit_rates, codes_from_hadx, evaluate, spells, valid_runs  # noqa: E402
from possession.touch_events import detect_touches  # noqa: E402

RUNS = os.path.expanduser(os.environ.get("MTAH_RUNS", "~/Data/more_than_a_hack_2027/runs"))   # can be redirected with the MTAH_RUNS environment variable, as in mtah_v3 and seq_model
COST_GRID = [(4, 4, 3), (8, 8, 6), (8, 32, 24), (16, 16, 12), (32, 32, 24), (64, 64, 48)]
SCORER_DEFAULT = {"max_iter": 250, "learning_rate": 0.08, "max_leaf_nodes": 31, "l2_regularization": 1.0}


def src_weight_map(spec):
    """Turn a spec such as 'metrica=3,idsse=1' into a dict that maps source prefix to weight; an empty string gives {}"""
    out = {}
    for kv in (spec or "").split(","):
        if kv.strip():
            k, v = kv.split("="); out[k.strip()] = float(v)
    return out


def src_weight_of(game, src_w):
    for k, v in src_w.items():
        if game.startswith(k):
            return v
    return 1.0


def scorer_params(spec):
    """Overlay the comma separated k=v list of --scorer on SCORER_DEFAULT, integer parameters become int, the rest float; an empty string gives the defaults"""
    p = dict(SCORER_DEFAULT)
    for kv in (spec or "").split(","):
        if not kv.strip():
            continue
        k, v = kv.split("=")
        k = k.strip()
        p[k] = int(v) if k in ("max_iter", "max_leaf_nodes", "min_samples_leaf", "max_depth", "max_bins") else float(v)
    return p
FPS = 25.0
T0 = time.time()


def log(msg):
    print("[%5.0fs] %s" % (time.time() - T0, msg), flush=True)


# ---------- load one match (same degradation settings as rescore_oof.raw_game; also returns the degraded M, used to find touches) ----------
def eval_degrade(bundle):
    """Degradation settings of the held-out match (the benchmark), consistent with rescore_oof.raw_game and --eval_realistic of mtah_v3"""
    noise, realistic = bundle.get("noise", 0.3), bool(bundle.get("realistic", False))
    if bundle.get("eval_realistic"):
        noise, realistic = 0.01, True
    return noise, realistic


def default_touch_set(bundle):
    """Touch thresholds follow the noise of the benchmark, robust from 0.1 m upwards, clean otherwise"""
    return "robust" if eval_degrade(bundle)[0] >= 0.1 else "clean"


def harness_extra(bundle):
    """The degradation parameters of the fixed degradation beyond noise and ball gaps (slow roll of a static ball static_wander, smoothing of the IDSSE ball trajectory ball_smooth, imputation error impute_sigma).
    The held-out match must be degraded with the same settings as in training, otherwise ball visibility and touches in the segment features do not match the probabilities from training; runs without these keys use 0.
    When the run has eval_static_wander or eval_ball_smooth (the held-out match is degraded under another condition, the cross-condition leave-one-match-out of mtah_v3) that condition is used"""
    sw = float(bundle.get("static_wander") or 0.0)
    bs = int(bundle.get("ball_smooth") or 0)
    imp = float(bundle.get("impute_sigma") or 0.0)
    esw, ebs = bundle.get("eval_static_wander"), bundle.get("eval_ball_smooth")
    if esw is not None and esw >= 0:
        sw = float(esw)
    if ebs is not None and ebs >= 0:
        bs = int(ebs)
    return sw, bs, imp


LABEL_MIN_SPELL = 0      # when >0, the scorer targets and the evaluation of IDSSE use labels with possession spells shorter than this many frames merged (degradation and features still use the raw labels)


def bench_bundle(bundle, tag):
    """Degradation settings of the extra benchmark tag. sw{wander}_bs{smooth} is the fixed realistic degradation with that slow roll and smoothing (mtah_v3 --extra_eval), rnd{k} is a randomised benchmark
    (--rand_bench of mtah_v3 and seq_model, degrade_random.draw_bench). _bench_tag only lets load_game_full recognise a randomised benchmark"""
    if re.fullmatch(r"rnd\d+", tag):
        return dict(bundle, _bench_tag=tag)
    m_ = re.fullmatch(r"sw([0-9.]+)_bs(\d+)", tag)
    assert m_, "benchmark name should look like sw0_bs0 or rnd1, got %s" % tag
    return dict(bundle, eval_realistic=True, eval_static_wander=float(m_.group(1)), eval_ball_smooth=int(m_.group(2)), _bench_tag=tag)


def bundle_main_tag(bundle):
    """Name of the main benchmark, rnd0 for runs with a randomised main benchmark (--rand_main), main otherwise"""
    return "rnd0" if bundle.get("rand_main") else "main"


def avg_scores(lst):
    """Average the score_pred results of one match over several benchmarks (used by --score_benches)"""
    out = {"spell_f1": float(np.mean([r["spell_f1"] for r in lst])), "macro_f1": float(np.mean([r["macro_f1"] for r in lst])),
           "per_iou": {k: float(np.mean([r["per_iou"][k] for r in lst])) for k in lst[0]["per_iou"]},
           "hit_by_length": {k: float(np.nanmean([r["hit_by_length"][k] for r in lst])) for k in lst[0]["hit_by_length"]},
           "n_gt": int(round(np.mean([r["n_gt"] for r in lst]))), "n_pred": int(round(np.mean([r["n_pred"] for r in lst])))}
    return out


def load_game_full(g, bundle):
    tag = bundle.get("_bench_tag") or ("rnd0" if bundle.get("rand_main") else "")
    if tag.startswith("rnd"):
        # randomised benchmark, the same degradation as when mtah_v3 --rand_bench built the benchmark (parameters and seeds depend only on the match name and k)
        from degrade_random import load_degraded
        from mtah_v3 import IDSSE_ROOT
        M2, gt = load_degraded(g, tag, IDSSE_ROOT, idsse_min_spell=int(bundle.get("idsse_min_spell") or 0))
        if g.startswith("idsse_"):                       # the degradation only uses the dead-ball frames, and merging short spells leaves those untouched; scoring and scorer targets use the raw labels
            from possession.adapters.idsse import load_match
            gt = load_match(IDSSE_ROOT, g[len("idsse_"):], min_spell=LABEL_MIN_SPELL)[1]
        return M2, np.asarray(gt)
    noise, realistic = eval_degrade(bundle)
    sw, bs, imp = harness_extra(bundle)
    if g.startswith("metrica"):
        from mtah_v3 import augment_metrica
        from possession.adapters.metrica_sample import load_game
        M, gt = load_game("~/Data/alt_training/metrica_sample", int(g[-1]))
        M2 = augment_metrica(M, gt, 0, noise, bundle.get("alive_missing", 0.30), realistic, bundle.get("metrica_jitter", 0.0) or 0.0, imp, sw)
    elif g.startswith("idsse_"):
        from mtah_v3 import IDSSE_ROOT, degrade_like_wc
        from possession.adapters.idsse import load_match
        M, gt = load_match(IDSSE_ROOT, g[len("idsse_"):], min_spell=0)
        M2 = degrade_like_wc(M, gt, 0, noise, realistic=realistic, impute_sigma=imp, static_wander=sw, ball_smooth=bs)
        if LABEL_MIN_SPELL > 0:
            gt = load_match(IDSSE_ROOT, g[len("idsse_"):], min_spell=LABEL_MIN_SPELL)[1]
    elif g.startswith("skillcorner_"):
        from mtah_v3 import SC_ROOT, degrade_like_wc
        from possession.adapters.skillcorner import load_match as sc_load
        M, gt = sc_load(SC_ROOT, g[len("skillcorner_"):])
        M2 = degrade_like_wc(M, gt, 0, noise, realistic=realistic, keep_missing=True)
    else:
        raise ValueError("unsupported match %s" % g)
    return M2, np.asarray(gt)


def pick_switch(heads, g):
    """heads is a list with one dict per head that maps match name to probabilities; returns the change-point probabilities of the current match. With a single head a plain array is returned"""
    if heads is None:
        return None
    arrs = [h[g].astype(np.float64) for h in heads]
    return arrs[0] if len(arrs) == 1 else arrs


# ---------- candidate boundaries ----------
CAND_EXTRA = ""   # empty, touch, argmax or both
SEG_SETP = False  # --seg_setp
SEG_SHAPE = False  # --seg_shape


def switch_list():
    """The current change-point heads, always as a list; one head and several heads share the same code"""
    if SWITCH_CUR is None:
        return []
    return SWITCH_CUR if isinstance(SWITCH_CUR, list) else [SWITCH_CUR]


def extra_boundaries(P, tt, a, b):
    """Extra candidate boundary frames of block [a, b] according to CAND_EXTRA"""
    out = []
    if CAND_EXTRA in ("touch", "both"):
        out += [int(t["frame"]) for t in tt if a < t["frame"] <= b]
    if CAND_EXTRA in ("argmax", "both"):
        am = P.argmax(1).astype(np.int16)
        h = 2
        amp = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(am, (h, h), mode="edge"), 5), axis=1)
        out += [int(t) for t in (np.flatnonzero(amp[1:] != amp[:-1]) + 1).tolist() if a < t <= b]
    if SWITCH_CUR is not None and SWITCH_TAU > 0:
        # local maxima of the change-point probability (maximum within ±SWITCH_HW frames and not below SWITCH_TAU) become candidate boundaries
        hw = SWITCH_HW
        for sw in switch_list():
            pw = np.lib.stride_tricks.sliding_window_view(np.pad(sw, (hw, hw), mode="edge"), 2 * hw + 1).max(axis=1)
            pk = np.flatnonzero((sw >= SWITCH_TAU) & (sw >= pw))
            last = -10 ** 9
            for t in pk.tolist():
                if a < t <= b and t - last > hw:
                    out.append(int(t)); last = t
    return sorted(set(out))


def switch_touches(M, touch_set):
    """Touch frames where the team changes (owned touches whose team differs from that of the previous owned touch) and their teams"""
    tt = [d for d in detect_touches(M, **TOUCH_SETS[touch_set]) if d["kind"] != "reappear" and d["team"] > 0]
    fr, tm, last = [], [], 0
    for d in tt:
        if d["team"] != last:
            fr.append(d["frame"]); tm.append(d["team"])
        last = d["team"]
    return np.array(fr, int), np.array(tm, int), tt


def build_candidates(a, b, decs, vis, sw_fr, merge=2, extra=()):
    """Candidate boundaries inside block [a, b] (a and b+1 included); returns the frames and the origin of each boundary (number of Viterbi settings that change there, whether it is a ball gap end point, team of the touch); extra holds the frames added by CAND_EXTRA, with origin 0"""
    pts = {a: [0, 0, 0], b + 1: [0, 0, 0]}
    for t in extra:
        pts.setdefault(int(t), [0, 0, 0])
    for d in decs:
        w = d[a:b + 1]
        for t in (np.flatnonzero(w[1:] != w[:-1]) + 1 + a).tolist():
            pts.setdefault(t, [0, 0, 0])[0] += 1
    v = vis[a:b + 1]
    for t in (np.flatnonzero(v[1:] != v[:-1]) + 1 + a).tolist():
        pts.setdefault(t, [0, 0, 0])[1] = 1
    return pts


def merge_close(pts, sw_fr, sw_tm, a, b, merge=2):
    for t, tm in zip(sw_fr.tolist(), sw_tm.tolist()):
        if a < t <= b:
            pts.setdefault(t, [0, 0, 0])[2] = tm
    keys = sorted(pts)
    out, src = [], []
    for t in keys:
        if out and t - out[-1] <= merge and t != b + 1:
            s0 = src[-1]
            s0[0] = max(s0[0], pts[t][0]); s0[1] = max(s0[1], pts[t][1]); s0[2] = s0[2] or pts[t][2]
            continue
        out.append(t); src.append(list(pts[t]))
    if out[-1] != b + 1:
        out.append(b + 1); src.append([0, 0, 0])
    return np.array(out, int), np.array(src, int)


# ---------- segment features ----------
FEAT_NAMES = None
FEAT_SET = "v1"   # v1 is the base set; v2 appends to it
SHAPE_CUR = None   # --seg_shape, per frame team shape columns of the current match from shape_frame_arrays; None means off
SETP_CUR = None    # --seg_setp, per frame positioning columns of the current match from setp_frame_arrays; None means off
SEG_DEFL = False   # --seg_defl
DEFL_CUR = None    # --seg_defl, per frame ball deflection indicator of the current match; None means off


def defl_frame(B):
    """Per frame ball deflection indicator. 1 when the ball moves more than 0.2 m both in the 0.2 s before and in the 0.2 s after this frame and the angle between the two displacements exceeds 45 degrees, 0 when there is no deflection or it cannot be computed"""
    B = np.asarray(B, np.float64); T = len(B); out = np.zeros(T)
    if T <= 10:
        return out
    u = B[5:T - 5] - B[0:T - 10]; w = B[10:T] - B[5:T - 5]
    nu, nw = np.linalg.norm(u, axis=1), np.linalg.norm(w, axis=1)
    ok = np.isfinite(nu) & np.isfinite(nw) & (nu > 0.2) & (nw > 0.2)
    c = np.full(len(u), 1.0); c[ok] = (u[ok] * w[ok]).sum(1) / (nu[ok] * nw[ok])
    out[5:T - 5] = (ok & (c < np.cos(np.pi / 4))) * 1.0
    return out
SWITCH_CUR = None  # --switch_run, per frame change-point probabilities (T,) of the current match (loso_oof_boundary.npz of mtah_v3 --boundary), a list of such arrays with several heads; None means off
SWITCH_TAU = 0.5   # a local maximum of the change-point probability becomes a candidate boundary only when it is not below this value, 0 adds no candidates
SWITCH_HW = 6      # half window of the local maximum, also the half window for the maximum change-point probability at the two segment ends (frames)
SWITCH_LAMBDA = 0.0  # --switch_lambda, when >0 every boundary where the label changes in the segment decoding gets an extra λ·(S−0.5), where S is the maximum within ±SWITCH_HW frames of the boundary of the mean probability over the change-point heads; 0 is off
MEMBER_P = []     # --member_runs, the (T,3) probabilities of each ensemble member; every segment gets the mean own team, opponent and dead-ball probability of each member and the gap between members; empty means off


def _sparse(x, op):
    """Sparse table for range minimum or maximum queries, levels[j][i] is the extreme value of x[i:i+2^j]"""
    x = np.asarray(x, np.float32); T = len(x); levels = [x]; j = 1
    while (1 << j) <= T:
        prev = levels[-1]; half = 1 << (j - 1); n = T - (1 << j) + 1
        levels.append(op(prev[:n], prev[half:half + n])); j += 1
    return levels


def _rmq(levels, s, e, op):
    """Extreme value of every interval [s, e) (e > s)"""
    k = np.floor(np.log2(np.maximum(e - s, 1))).astype(int)
    out = np.empty(len(s), np.float32)
    for kk in np.unique(k):
        m = k == kk; lv = levels[kk]
        out[m] = op(lv[s[m]], lv[e[m] - (1 << kk)])
    return out


def _near(F, pos, clip=75):
    """Distance from pos to the nearest entry of the sorted frame array F, capped at clip; all clip when F is empty"""
    if len(F) == 0:
        return np.full(len(pos), float(clip), np.float32)
    j = np.searchsorted(F, pos)
    d1 = np.where(j < len(F), np.abs(F[np.minimum(j, len(F) - 1)] - pos), clip)
    d2 = np.where(j > 0, np.abs(pos - F[np.maximum(j - 1, 0)]), clip)
    return np.minimum(np.minimum(d1, d2), clip).astype(np.float32)


def _touch_in(F, TM, s, e, lab, first):
    """Whether the first (or last) touch in [s, e) belongs to team lab, 1 yes, 0 no, -1 no touch inside the segment"""
    out = np.full(len(s), -1.0, np.float32)
    if len(F) == 0:
        return out
    if first:
        j = np.searchsorted(F, s, "left"); jj = np.minimum(j, len(F) - 1); ok = (j < len(F)) & (F[jj] < e)
    else:
        j = np.searchsorted(F, e, "left") - 1; jj = np.maximum(j, 0); ok = (j >= 0) & (F[jj] >= s)
    out[ok] = (TM[jj[ok]] == lab) * 1.0
    return out


def segment_table(P, decs, vis, touches_all, cands_by_block, blocks, kmax):
    """Returns a dict keyed by block number with values (I, K, feats_H, feats_A, s, e), I is the index of the start candidate, K the number of intervals spanned"""
    global FEAT_NAMES
    T = len(P)
    pH, pA, pD = P[:, 0], P[:, 1], P[:, 2]
    am = P.argmax(1)
    th = np.zeros(T); ta = np.zeros(T); tu = np.zeros(T)
    for d in touches_all:
        f = d["frame"]
        if d["team"] == 1:
            th[f] += 1
        elif d["team"] == 2:
            ta[f] += 1
        else:
            tu[f] += 1
    arrs = {"pH": pH, "pA": pA, "pD": pD, "amH": (am == 0) * 1.0, "amA": (am == 1) * 1.0, "amD": (am == 2) * 1.0,
            "hiH": (pH > 0.8) * 1.0, "hiA": (pA > 0.8) * 1.0, "vis": vis * 1.0, "th": th, "ta": ta, "tu": tu}
    for k, d in enumerate(decs):
        arrs["d%dH" % k] = (d == 0) * 1.0; arrs["d%dA" % k] = (d == 1) * 1.0; arrs["d%dD" % k] = (d == 2) * 1.0
    for j, Pm in enumerate(MEMBER_P):
        arrs["mj%dH" % j] = Pm[:, 0]; arrs["mj%dA" % j] = Pm[:, 1]; arrs["mj%dD" % j] = Pm[:, 2]
    if FEAT_SET in ("v2", "v3"):
        sw = np.zeros(T); last = 0
        for d in touches_all:
            if d["team"] != last:
                sw[d["frame"]] = 1.0
            last = d["team"]
        arrs.update({"sqH": pH * pH, "sqA": pA * pA, "loH": (pH < 0.2) * 1.0, "loA": (pA < 0.2) * 1.0, "sw": sw})
        ker = np.ones(25) / 25.0
        st_min = {"H": _sparse(pH, np.minimum), "A": _sparse(pA, np.minimum)}
        st_max = {"H": _sparse(np.convolve(pH, ker, mode="same"), np.maximum), "A": _sparse(np.convolve(pA, ker, mode="same"), np.maximum)}
        tf_all = np.array([d["frame"] for d in touches_all], int); tm_all = np.array([d["team"] for d in touches_all], int)
        o = np.argsort(tf_all, kind="stable"); tf_all, tm_all = tf_all[o], tm_all[o]
        tf = {"H": tf_all[tm_all == 1], "A": tf_all[tm_all == 2]}
    sw_heads = switch_list(); st_swp = []
    for j, sw in enumerate(sw_heads):
        arrs["swp%d" % j] = sw.astype(np.float64); st_swp.append(_sparse(sw, np.maximum))
    if SETP_CUR is not None:
        for k_ in ("corner_d_h", "corner_d_a", "touch_out_h", "touch_out_a", "goal_out_h", "goal_out_a", "own_half_h", "own_half_a",
                   "box_att_h", "box_att_a", "box_def_h", "box_def_a", "edge_m", "edge_iso"):
            arrs["sp_" + k_] = SETP_CUR[k_].astype(np.float64)
        arrs["sp_still"] = (SETP_CUR["n_still_h"] + SETP_CUR["n_still_a"]).astype(np.float64)
        arrs["sp_maxspd"] = np.maximum(SETP_CUR["max_speed_h"], SETP_CUR["max_speed_a"]).astype(np.float64)
        arrs["sp_edgeH"] = (SETP_CUR["edge_home"] == 1) * 1.0; arrs["sp_edgeA"] = (SETP_CUR["edge_home"] == 0) * 1.0
    if SHAPE_CUR is not None:
        for k_, nm_ in (("v_fwd", "v"), ("f_fwd", "ff"), ("f_back", "fb")):
            arrs["sh_%sH" % nm_] = SHAPE_CUR[k_ + "_h"].astype(np.float64); arrs["sh_%sA" % nm_] = SHAPE_CUR[k_ + "_a"].astype(np.float64)
    if DEFL_CUR is not None:
        arrs["defl"] = DEFL_CUR
    cs = {k: np.r_[0.0, np.cumsum(v)] for k, v in arrs.items()}

    def mean(key, s, e):
        return (cs[key][e] - cs[key][s]) / np.maximum(e - s, 1)

    out = {}
    for h, (a, b) in blocks.items():
        c, src = cands_by_block[h]
        N = len(c) - 1
        I, K = [], []
        for k in range(1, kmax + 1):
            i = np.arange(0, N - k + 1)
            I.append(i); K.append(np.full(len(i), k))
        I, K = np.concatenate(I), np.concatenate(K)
        s, e = c[I], c[I + K]
        L = (e - s) / FPS
        W1, W3 = 25, 75
        sb1, ea1 = np.maximum(s - W1, a), np.minimum(e + W1, b + 1)
        sb3, ea3 = np.maximum(s - W3, a), np.minimum(e + W3, b + 1)
        common = {"len_s": L, "log_len": np.log(L + 0.04), "n_int": K * 1.0, "vis_frac": mean("vis", s, e), "m_D": mean("pD", s, e),
                  "am_D": mean("amD", s, e), "tu_rate": (cs["tu"][e] - cs["tu"][s]) / L,
                  "pre_D": mean("pD", sb1, s), "post_D": mean("pD", e, ea1),
                  "s_ndec": src[I, 0] * 1.0, "s_gap": src[I, 1] * 1.0, "e_ndec": src[I + K, 0] * 1.0, "e_gap": src[I + K, 1] * 1.0,
                  "is_bstart": (s == a) * 1.0, "is_bend": (e == b + 1) * 1.0}
        for k in range(len(decs)):
            common["dec%d_D" % k] = mean("d%dD" % k, s, e)
        if DEFL_CUR is not None:
            common["defl_frac"] = mean("defl", s, e)   # share of ball deflection frames inside the segment
        if sw_heads:
            # change-point probability features, the maximum change-point probability within ±SWITCH_HW frames of each segment end, and the maximum and mean inside the segment with both ends removed (the whole segment when it is too short);
            # with several heads the feature names of the second and later heads carry an index
            hw = SWITCH_HW
            s0, s1 = np.maximum(s - hw, a), np.minimum(s + hw + 1, b + 1)
            e0, e1 = np.maximum(e - hw, a), np.minimum(e + hw + 1, b + 1)
            i0, i1 = np.minimum(s + hw, e - 1), np.maximum(e - hw, s + 1)
            short = i1 <= i0
            i0 = np.where(short, s, i0); i1 = np.where(short, e, i1)
            for j in range(len(sw_heads)):
                tag = "" if j == 0 else str(j)
                common.update({"s_swp" + tag: _rmq(st_swp[j], s0, s1, np.maximum), "e_swp" + tag: _rmq(st_swp[j], e0, e1, np.maximum),
                               "in_swp_max" + tag: _rmq(st_swp[j], i0, i1, np.maximum), "in_swp_mean" + tag: mean("swp%d" % j, i0, i1)})
        if SETP_CUR is not None:
            # positioning features, the 3 s windows before the start and after the end of the segment, and the number of stationary players and the maximum speed inside the segment
            common.update({"sp_pre_still": mean("sp_still", sb3, s), "sp_post_still": mean("sp_still", e, ea3), "sp_in_still": mean("sp_still", s, e),
                           "sp_pre_maxspd": mean("sp_maxspd", sb3, s), "sp_post_maxspd": mean("sp_maxspd", e, ea3), "sp_in_maxspd": mean("sp_maxspd", s, e),
                           "sp_pre_edge_m": mean("sp_edge_m", sb3, s), "sp_post_edge_m": mean("sp_edge_m", e, ea3),
                           "sp_pre_edge_iso": mean("sp_edge_iso", sb3, s), "sp_post_edge_iso": mean("sp_edge_iso", e, ea3)})
        if FEAT_SET in ("v2", "v3"):
            common.update({"n_sw_in": cs["sw"][e] - cs["sw"][np.minimum(s + 1, e)], "vis_s": vis[s] * 1.0, "vis_e": vis[e - 1] * 1.0,
                           "pre_vis": mean("vis", sb1, s), "post_vis": mean("vis", e, ea1), "pos_frac": (s - a) / max(b + 1 - a, 1)})
        rows = {}
        for lab, own, oth in ((1, "H", "A"), (2, "A", "H")):
            f = dict(common)
            f.update({"lab_home": np.full(len(s), 1.0 if lab == 1 else 0.0), "m_own": mean("p" + own, s, e), "m_oth": mean("p" + oth, s, e),
                      "am_own": mean("am" + own, s, e), "am_oth": mean("am" + oth, s, e), "hi_own": mean("hi" + own, s, e),
                      "t_own_rate": (cs["t" + own.lower()][e] - cs["t" + own.lower()][s]) / L,
                      "t_oth_rate": (cs["t" + oth.lower()][e] - cs["t" + oth.lower()][s]) / L,
                      "pre_own": mean("p" + own, sb1, s), "post_own": mean("p" + own, e, ea1), "pre_oth": mean("p" + oth, sb1, s),
                      "post_oth": mean("p" + oth, e, ea1), "pre3_own": mean("p" + own, sb3, s), "post3_own": mean("p" + own, e, ea3),
                      "s_touch_own": (src[I, 2] == lab) * 1.0, "s_touch_oth": ((src[I, 2] > 0) & (src[I, 2] != lab)) * 1.0,
                      "e_touch_own": (src[I + K, 2] == lab) * 1.0, "e_touch_oth": ((src[I + K, 2] > 0) & (src[I + K, 2] != lab)) * 1.0})
            for k in range(len(decs)):
                f["dec%d_own" % k] = mean("d%d%s" % (k, own), s, e)
            if SETP_CUR is not None:
                o_, x_ = own.lower(), oth.lower()
                f.update({"sp_pre_edge_own": mean("sp_edge" + own, sb3, s), "sp_post_edge_own": mean("sp_edge" + own, e, ea3),
                          "sp_pre_tout_own": mean("sp_touch_out_" + o_, sb3, s), "sp_pre_tout_oth": mean("sp_touch_out_" + x_, sb3, s),
                          "sp_post_tout_oth": mean("sp_touch_out_" + x_, e, ea3),
                          "sp_pre_corner_own": mean("sp_corner_d_" + o_, sb3, s), "sp_pre_corner_oth": mean("sp_corner_d_" + x_, sb3, s),
                          "sp_post_corner_oth": mean("sp_corner_d_" + x_, e, ea3),
                          "sp_pre_gout_own": mean("sp_goal_out_" + o_, sb3, s), "sp_pre_half_own": mean("sp_own_half_" + o_, sb3, s),
                          "sp_pre_half_oth": mean("sp_own_half_" + x_, sb3, s), "sp_pre_boxatt_own": mean("sp_box_att_" + o_, sb3, s),
                          "sp_pre_boxdef_oth": mean("sp_box_def_" + x_, sb3, s), "sp_post_gout_oth": mean("sp_goal_out_" + x_, e, ea3)})
            if SHAPE_CUR is not None:
                # team shape direction features, mean speed of the own team and of the opponent along their attacking directions, inside the segment and in the 3 s before and after; shares of players running forwards and backwards inside the segment
                f.update({"sh_in_v_own": mean("sh_v" + own, s, e), "sh_in_v_oth": mean("sh_v" + oth, s, e), "sh_pre_v_own": mean("sh_v" + own, sb3, s), "sh_pre_v_oth": mean("sh_v" + oth, sb3, s),
                          "sh_post_v_own": mean("sh_v" + own, e, ea3), "sh_post_v_oth": mean("sh_v" + oth, e, ea3), "sh_in_ff_own": mean("sh_ff" + own, s, e), "sh_in_fb_own": mean("sh_fb" + own, s, e),
                          "sh_in_ff_oth": mean("sh_ff" + oth, s, e), "sh_in_fb_oth": mean("sh_fb" + oth, s, e)})
            for j in range(len(MEMBER_P)):
                f["mem%d_own" % j] = mean("mj%d%s" % (j, own), s, e); f["mem%d_oth" % j] = mean("mj%d%s" % (j, oth), s, e); f["mem%d_D" % j] = mean("mj%dD" % j, s, e)
            if len(MEMBER_P) >= 2:
                f["mem_gap_own"] = np.abs(f["mem0_own"] - f["mem1_own"]); f["mem_gap_D"] = np.abs(f["mem0_D"] - f["mem1_D"])
            if FEAT_SET in ("v2", "v3"):
                f.update({"min_own": _rmq(st_min[own], s, e, np.minimum),
                          "std_own": np.sqrt(np.maximum(mean("sq" + own, s, e) - mean("p" + own, s, e) ** 2, 0.0)),
                          "lo_own": mean("lo" + own, s, e), "oth25_max": _rmq(st_max[oth], s, e, np.maximum),
                          "pre3_oth": mean("p" + oth, sb3, s), "post3_oth": mean("p" + oth, e, ea3),
                          "t_own_n": cs["t" + own.lower()][e] - cs["t" + own.lower()][s], "t_oth_n": cs["t" + oth.lower()][e] - cs["t" + oth.lower()][s],
                          "s_near_own": _near(tf[own], s), "s_near_oth": _near(tf[oth], s), "e_near_own": _near(tf[own], e), "e_near_oth": _near(tf[oth], e),
                          "first_touch_own": _touch_in(tf_all, tm_all, s, e, lab, True), "last_touch_own": _touch_in(tf_all, tm_all, s, e, lab, False)})
                if FEAT_SET == "v3":
                    for kdrop in ("n_sw_in", "t_own_n", "t_oth_n", "s_near_own", "s_near_oth", "e_near_own", "e_near_oth", "first_touch_own", "last_touch_own"):
                        f.pop(kdrop)
            if FEAT_NAMES is None:
                FEAT_NAMES = list(f)
            rows[lab] = np.column_stack([f[n] for n in FEAT_NAMES]).astype(np.float32)
        out[h] = (I, K, rows[1], rows[2], s, e)
    return out


def spell_iou_targets(gt, s, e, lab):
    """Maximum IoU of each candidate segment, labelled as lab (1 home, 2 away), with the ground-truth possession spells of the same team"""
    G = codes_from_hadx(gt)
    gs = [(x, y) for x, y, cd in spells(G, valid_runs(G)) if cd == (0 if lab == 1 else 1)]
    if not gs:
        return np.zeros(len(s))
    gst = np.array([x for x, _ in gs]); gen = np.array([y for _, y in gs])
    best = np.zeros(len(s))
    lo = np.searchsorted(gen, s, side="right")
    hi = np.searchsorted(gst, e, side="left")
    for n in range(len(s)):
        for j in range(lo[n], hi[n]):
            inter = min(e[n], gen[j]) - max(s[n], gst[j])
            if inter > 0:
                v = inter / (max(e[n], gen[j]) - min(s[n], gst[j]))
                if v > best[n]:
                    best[n] = v
    return best


# ---------- semi-Markov dynamic programme ----------
def dp_block(c, I, K, rH, rA, rD, kmax, bonus=None):
    """c candidate boundaries (N+1 of them), I and K start index and span of the segments, rH, rA, rD rewards of the three labels for each segment; returns the label of every interval (0 H, 1 A, 2 D).
    bonus (N+1 values, may be None) is the reward added when the label changes at candidate boundary i, the start of the block does not count as a boundary; None means off"""
    N = len(c) - 1
    R = np.full((3, N, kmax + 1), -np.inf)
    R[0, I, K] = rH; R[1, I, K] = rA; R[2, I, K] = rD
    best = np.full((N + 1, 3), -np.inf)
    arg = np.zeros((N + 1, 3, 2), int)
    best[0, :] = 0.0
    for j in range(1, N + 1):
        ks = np.arange(1, min(kmax, j) + 1)
        i = j - ks
        for lab in range(3):
            if bonus is not None:
                bz = np.where(i > 0, bonus[i], 0.0)
                bb = best[i] + bz[:, None]
                if lab == 2:
                    bb[:, 2] = best[i][:, 2]          # dead ball followed by dead ball is not a label change, no bonus
                else:
                    bb[:, lab] = -np.inf
                bb[i == 0] = best[0]                  # at the start all three are 0, any label may open the path
                prev = bb.max(1); pl = bb.argmax(1)
            elif lab == 2:
                prev = best[i].max(1); pl = best[i].argmax(1)
            else:
                other = [x for x in range(3) if x != lab]
                bb = best[i][:, other]
                prev = bb.max(1); pl = np.array(other)[bb.argmax(1)]
            # there is no previous segment at the start, best[0] is 0 for all three and any label may open the path
            tot = prev + R[lab, i, ks]
            n = int(np.argmax(tot))
            best[j, lab] = tot[n]; arg[j, lab] = (i[n], pl[n])
    lab = int(np.argmax(best[N])); j = N
    out = np.zeros(N, np.int8)
    while j > 0:
        i, pl = arg[j, lab]
        out[i:j] = lab
        j, lab = i, pl
    return out


def decode_game(gd, pH, pA, theta, gamma, kmax):
    T = len(gd["P"])
    pred = np.zeros(T, np.int8)
    for h, (a, b) in gd["blocks"].items():
        c = gd["cands"][h][0]
        I, K, _, _, s, e = gd["seg"][h]
        L = (e - s) / FPS
        mD = gd["mD"][h]
        rD = gamma * L * (mD - 0.5)
        bonus = SWITCH_LAMBDA * (gd["bsw"][h] - 0.5) if SWITCH_LAMBDA > 0 and gd.get("bsw") else None
        lab_int = dp_block(c, I, K, pH[h] - theta, pA[h] - theta, rD, kmax, bonus)
        for n in range(len(c) - 1):
            pred[c[n]:c[n + 1]] = lab_int[n] + 1
    pred[~gd["graded"]] = 0
    if MERGE_SLIVER > 0:
        pred = merge_slivers(pred, MERGE_SLIVER)
    return pred


MERGE_SLIVER = 0     # --merge_dead_sliver, 0 is off


def merge_slivers(pred, n):
    """Merge a dead-ball segment shorter than n frames into the team in possession when both neighbours belong to that same team"""
    pred = pred.copy()
    b = np.flatnonzero(np.diff(np.r_[-99, pred, -99]))
    rr = [(int(i), int(j), int(pred[i])) for i, j in zip(b[:-1], b[1:])]
    for k in range(1, len(rr) - 1):
        a, m, c = rr[k - 1], rr[k], rr[k + 1]
        if m[2] == 3 and m[1] - m[0] < n and a[2] == c[2] and a[2] in (1, 2):
            pred[m[0]:m[1]] = a[2]
    return pred


def score_pred(gt, pred):
    G, Pc = codes_from_hadx(gt), codes_from_hadx(pred)
    r = evaluate(G, Pc)
    hb = {k: v[0] for k, v in binned_hit_rates(G, Pc).items()}
    return {"spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"], "per_iou": {str(k): v for k, v in r["spell_f1_per_iou"].items()},
            "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"], "hit_by_length": hb}


ALT_SPEC = {}       # --alt_labels, not used when empty


def alt_gt(g, gt_ref):
    """The alternative label set of --alt_labels. Metrica is reloaded with MTAH_METRICA_CONTROL set by the control switch, IDSSE is reloaded with min_spell;
    the number of frames equals that of the main labels, and frames that are not graded (0) in the main labels are 0 here too"""
    if g.startswith("metrica"):
        from possession.adapters.metrica_sample import load_game
        old = os.environ.get("MTAH_METRICA_CONTROL")
        if ALT_SPEC["control"]:
            os.environ["MTAH_METRICA_CONTROL"] = "1"
        else:
            os.environ.pop("MTAH_METRICA_CONTROL", None)
        try:
            gt = load_game("~/Data/alt_training/metrica_sample", int(g[-1]))[1]
        finally:
            if old is None:
                os.environ.pop("MTAH_METRICA_CONTROL", None)
            else:
                os.environ["MTAH_METRICA_CONTROL"] = old
    elif g.startswith("idsse_"):
        from mtah_v3 import IDSSE_ROOT
        from possession.adapters.idsse import load_match
        gt = load_match(IDSSE_ROOT, g[len("idsse_"):], min_spell=ALT_SPEC["min_spell"])[1]
    else:
        raise ValueError("--alt_labels only handles metrica and idsse matches, not %s" % g)
    gt = np.asarray(gt).copy()
    assert len(gt) == len(gt_ref), "%s alternative labels have %d frames, the main labels have %d" % (g, len(gt), len(gt_ref))
    gt[np.asarray(gt_ref) == 0] = 0
    return gt


def prep_game(M, gt, P, touch_set, kmax, base_costs, span_need):
    """Candidate boundaries, segment features, scorer targets and the Viterbi baseline of one match; span_need is a list to which the number of candidate intervals spanned by each ground-truth possession spell is appended"""
    global SETP_CUR, SHAPE_CUR, DEFL_CUR
    if SEG_DEFL:                                               # --seg_defl
        DEFL_CUR = defl_frame(M["ball_xy"])
    if SEG_SHAPE:                                             # --seg_shape
        from possession.mtah_features import shape_frame_arrays
        SHAPE_CUR = shape_frame_arrays(M)
    if SEG_SETP:                                               # --seg_setp
        from possession.mtah_features import setp_frame_arrays
        SETP_CUR = setp_frame_arrays(M)
    logp = np.log(np.clip(P, 1e-6, 1))
    blocks = {int(k): (int(v[0]), int(v[1])) for k, v in M["blocks"].items()}
    graded = np.asarray(M["graded"], bool)
    vis = np.asarray(M["ball_visible"], bool)
    decs = [viterbi3(logp, blocks, *c) for c in COST_GRID]
    base = to_hadx(viterbi3(logp, blocks, *base_costs)); base[~graded] = 0
    sw_fr, sw_tm, tt = switch_touches(M, touch_set)
    cands = {}
    for h, (bA, bB) in blocks.items():
        pts = build_candidates(bA, bB, decs, vis, sw_fr, extra=extra_boundaries(P, tt, bA, bB))
        cands[h] = merge_close(pts, sw_fr, sw_tm, bA, bB)
    seg = segment_table(P, decs, vis, tt, cands, blocks, kmax)
    tgt, mD = {}, {}
    for h in blocks:
        I, K, fH, fA, s, e = seg[h]
        tgt[h] = (spell_iou_targets(gt, s, e, 1), spell_iou_targets(gt, s, e, 2))
        mD[h] = fH[:, FEAT_NAMES.index("m_D")].astype(np.float64)
    bsw = {}
    if SWITCH_LAMBDA > 0 and switch_list():                    # --switch_lambda
        swm = np.mean([np.asarray(x, np.float64) for x in switch_list()], axis=0)
        for h in blocks:
            c = np.asarray(cands[h][0], int)
            bsw[h] = np.array([swm[max(0, x - SWITCH_HW):min(len(swm), x + SWITCH_HW + 1)].max() for x in c])
    # number of candidate intervals a ground-truth possession spell has to span (to check that kmax is large enough)
    G = codes_from_hadx(gt)
    for x, y, _ in spells(G, valid_runs(G)):
        for h, (bA, bB) in blocks.items():
            if bA <= x <= bB:
                c = cands[h][0]
                i0 = int(np.argmin(np.abs(c - x))); i1 = int(np.argmin(np.abs(c - y)))
                span_need.append(i1 - i0)
    return {"P": P, "gt": gt, "blocks": blocks, "graded": graded, "cands": cands, "seg": seg, "tgt": tgt, "mD": mD, "base": base, "n_sw": len(sw_fr), "bsw": bsw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, the directory under $MTAH_RUNS")
    ap.add_argument("--games", default=None)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--neg_frac", type=float, default=0.25)
    ap.add_argument("--touch_set", default=None, help="default chosen by the noise of the benchmark (0.01 when the run has eval_realistic), robust from 0.1 m upwards, clean otherwise")
    ap.add_argument("--thetas", default="0.15,0.2,0.25,0.3,0.35,0.4,0.5")
    ap.add_argument("--gammas", default="0,0.02,0.1")
    ap.add_argument("--ious", default="0.5", help="IoU targets of the scorer, with several values one scorer is trained per value and the probabilities are averaged (matching spellF1, which averages over 0.3 to 0.7)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_pred", default="auto", help="the new decoding of every held-out match (with the picked θ and γ) is stored as an npz, keys are match names, values the internal int8 codes (0 none, 1 H, 2 A, 3 D); "
                    "auto writes to runs/_decode/<run>_pred.npz, none stores nothing, a path may be given as well")
    ap.add_argument("--label_min_spell", type=int, default=0, help="the scorer targets and the evaluation of IDSSE use labels in which possession spells shorter than this many frames are merged, 0 for the raw labels")
    ap.add_argument("--merge_dead_sliver", type=int, default=0, help="after decoding, merge dead-ball segments shorter than this many frames into the team that has possession on both sides, 0 to disable")
    ap.add_argument("--alt_labels", default="", help="also score the same decoding (with the picked θγ) on an alternative label set, for example control=1,min_spell=50 or control=0,min_spell=0; "
                    "control is MTAH_METRICA_CONTROL of Metrica, min_spell the short spell merge length of IDSSE in frames; results go to alt in the JSON; empty by default")
    ap.add_argument("--scorer", default="", help="scorer hyperparameter overrides as a comma separated k=v list, default SCORER_DEFAULT")
    ap.add_argument("--feat_set", default="v1", choices=["v1", "v2", "v3"], help="segment feature set, v1 the base one, v2 appends 20 features, v3 appends only the 11 probability shape and ball visibility features (see the file header)")
    ap.add_argument("--src_weight", default="", help="source weights for the scorer training samples, for example metrica=3; no weighting by default")
    ap.add_argument("--cand_extra", default="", choices=["", "touch", "argmax", "both"], help="one more kind of candidate boundary (see the file header), empty by default")
    ap.add_argument("--member_runs", default="", help="run names of the ensemble members as a comma separated list; their loso_oof_proba.npz are read and every segment gets member features, empty by default")
    ap.add_argument("--scorer_seeds", type=int, default=1, help="number of scorers with different random seeds per IoU target, averaged; default 1")
    ap.add_argument("--extra_benches", default="", help="out-of-fold probabilities of other benchmarks of the same run also give candidate segments that only enter the scorer training, comma separated list such as sw0_bs0,sw4_bs21 or rnd1,rnd2 (reads loso_oof_proba_eval_<tag>.npz); empty by default")
    ap.add_argument("--score_benches", action="store_true", help="also score the benchmarks of --extra_benches, θγ are picked by the mean over the main benchmark and these, and base, new and grid_mean of the JSON become means over the benchmarks (per benchmark values in per_bench); off by default")
    ap.add_argument("--switch_run", default="", help="run names of the change-point probability heads, comma separated (loso_oof_boundary.npz written by mtah_v3 --boundary, matches and frame counts must agree with --run); when given, candidate boundaries are added and each head adds 4 segment features; empty by default")
    ap.add_argument("--switch_main_tag", default="", help="which benchmark of --switch_run provides the change-point probabilities of the main benchmark. Empty reads loso_oof_boundary.npz (the training degradation), a tag such as rnd0 reads loso_oof_boundary_eval_<tag>.npz (for runs whose main benchmark is rnd0)")
    ap.add_argument("--switch_tau", type=float, default=0.5, help="a local maximum of the change-point probability becomes a candidate boundary only when it is not below this value, 0 adds the features but no candidates")
    ap.add_argument("--switch_hw", type=int, default=6, help="half window of the local maximum of the change-point probability, also the half window for the maximum change-point probability at the two segment ends, in frames")
    ap.add_argument("--switch_lambda", type=float, default=0.0, help="every boundary where the label changes in the segment decoding gets an extra λ·(S−0.5), where S is the maximum within ±switch_hw frames of the boundary of the mean probability over the change-point heads; 0 adds nothing")
    ap.add_argument("--seg_shape", action="store_true", help="add the 10 team shape direction columns to the segment scorer, off by default")
    ap.add_argument("--seg_defl", action="store_true", help="add 1 column to the segment scorer, the share of ball deflection frames inside the segment (displacements in the 0.2 s before and after differ by more than 45 degrees), off by default")
    ap.add_argument("--seg_setp", action="store_true", help="add the 24 positioning columns to the segment scorer, off by default")
    ap.add_argument("--report_benches", default="", help="benchmarks that are only reported, comma separated list such as sw3_bs15,sw0_bs0,sw4_bs21. They do not enter the scorer training or the choice of θγ; the segments of the held-out match are scored by the scorer of the same fold and decoded with the picked θγ, results go to report in the JSON; empty by default")
    a = ap.parse_args()
    sp = scorer_params(a.scorer)
    src_w = src_weight_map(a.src_weight)
    global LABEL_MIN_SPELL, FEAT_SET, CAND_EXTRA, MEMBER_P, SWITCH_CUR, SWITCH_TAU, SWITCH_HW, SWITCH_LAMBDA
    global SEG_SETP, SEG_SHAPE, SEG_DEFL
    SEG_DEFL = bool(a.seg_defl)
    LABEL_MIN_SPELL = a.label_min_spell
    global MERGE_SLIVER
    MERGE_SLIVER = a.merge_dead_sliver
    if a.alt_labels:
        kv = dict(x.split("=") for x in a.alt_labels.split(","))
        ALT_SPEC.update({"control": int(kv["control"]), "min_spell": int(kv["min_spell"])})
    SWITCH_TAU, SWITCH_HW = a.switch_tau, a.switch_hw
    SWITCH_LAMBDA = a.switch_lambda
    SEG_SETP = bool(a.seg_setp)
    SEG_SHAPE = bool(a.seg_shape)
    FEAT_SET = a.feat_set
    CAND_EXTRA = a.cand_extra
    target_ious = [float(x) for x in a.ious.split(",")]
    d = os.path.join(RUNS, a.run)
    bundle = joblib.load(os.path.join(d, "model.joblib"))
    base_costs = tuple(float(x) for x in str(json.load(open(os.path.join(d, "metrics.json")))["costs_used"]).split(","))
    oof = dict(np.load(os.path.join(d, "loso_oof_proba.npz")))
    games = a.games.split(",") if a.games else list(oof)
    member_runs = [r for r in a.member_runs.split(",") if r]
    mem_oof = [dict(np.load(os.path.join(RUNS, r, "loso_oof_proba.npz"))) for r in member_runs]
    for r, mo in zip(member_runs, mem_oof):
        assert all(mo[g].shape == oof[g].shape for g in games), "the out-of-fold probabilities of member %s have a different shape from those of the ensemble" % r
    sw_oof = None
    if a.switch_run:
        sw_oof = [dict(np.load(os.path.join(RUNS, r, "loso_oof_boundary%s.npz" % ("_eval_" + a.switch_main_tag if a.switch_main_tag else "")))) for r in a.switch_run.split(",")]
        assert all(g in so and so[g].shape[0] == oof[g].shape[0] for so in sw_oof for g in games), "the change-point probabilities of --switch_run lack a match or have a different number of frames from the ensemble"
    def switch_bench(tag):
        """Change-point probabilities of --switch_run on another benchmark (loso_oof_boundary_eval_<tag>.npz, written by mtah_v3 --boundary together with --extra_eval)"""
        if sw_oof is None:
            return None
        fs = [os.path.join(RUNS, r, "loso_oof_boundary_eval_%s.npz" % tag) for r in a.switch_run.split(",")]
        for f in fs:
            assert os.path.exists(f), "--switch_run has no change-point probabilities for benchmark %s, %s" % (tag, f)
        return [dict(np.load(f)) for f in fs]
    touch_set = a.touch_set or default_touch_set(bundle)
    log("run %s, %d matches, baseline costs %s, benchmark degradation (noise, realistic ball gaps) %s, extra degradation (slow roll, smoothing, imputation) %s, touch thresholds %s, kmax %d, IDSSE short spells merged below %d frames, scorer %s, negative sampling %.2f, feature set %s, scorer seeds %d, source weights %s, extra candidate boundaries %s" % (
        a.run, len(games), base_costs, eval_degrade(bundle), harness_extra(bundle), touch_set, a.kmax, a.label_min_spell, sp, a.neg_frac, FEAT_SET, a.scorer_seeds, src_w, CAND_EXTRA or "none"))

    # candidates, segment features and targets of every match
    data = {}
    span_need = []
    for g in games:
        M, gt = load_game_full(g, bundle)
        P = oof[g].astype(np.float64)
        MEMBER_P = [mem_oof[j][g].astype(np.float64) for j in range(len(mem_oof))]
        SWITCH_CUR = pick_switch(sw_oof, g)
        data[g] = prep_game(M, gt, P, touch_set, a.kmax, base_costs, span_need)
        n_c = sum(len(data[g]["cands"][h][0]) for h in data[g]["blocks"]); n_s = sum(len(data[g]["seg"][h][0]) for h in data[g]["blocks"])
        log("%s loaded, %d candidate boundaries (%d team-switch touches), %d candidate segments" % (g, n_c, data[g]["n_sw"], n_s))
    sn = np.array(span_need)
    log("candidate intervals spanned by ground-truth possession spells, median %.0f, p90 %.0f, p99 %.0f, share above kmax %.3f" % (
        np.median(sn), np.percentile(sn, 90), np.percentile(sn, 99), (sn > a.kmax).mean()))
    # --extra_benches, the out-of-fold probabilities of other benchmarks of the same run (loso_oof_proba_eval_<tag>.npz, each match degraded again with the slow roll and smoothing of that benchmark)
    # also give candidate segments, which are only added to the training rows of the scorer (the held-out match stays out) and are not scored; empty by default
    data_x = []
    for tag in [t for t in a.extra_benches.split(",") if t]:
        assert not member_runs, "--extra_benches cannot be combined with --member_runs"
        bx = bench_bundle(bundle, tag)
        oof_x = dict(np.load(os.path.join(d, "loso_oof_proba_eval_%s.npz" % tag)))
        sw_x = switch_bench(tag)
        dx = {}
        for g in games:
            M, gt = load_game_full(g, bx)
            assert oof_x[g].shape[0] == len(gt), "%s %s, the number of frames of the out-of-fold probabilities does not match the rebuilt degradation" % (tag, g)
            MEMBER_P = []
            SWITCH_CUR = pick_switch(sw_x, g)
            dx[g] = prep_game(M, gt, oof_x[g].astype(np.float64), touch_set, a.kmax, base_costs, [])
        bs_x = float(np.mean([score_pred(dx[g]["gt"], dx[g]["base"])["spell_f1"] for g in games]))
        log("extra benchmark %s loaded (scorer training only), %d candidate segments, mean spellF1 of the Viterbi baseline %.4f" % (
            tag, sum(len(dx[g]["seg"][h][0]) for g in games for h in dx[g]["blocks"]), bs_x))
        data_x.append(dx)
    x_tags = [t for t in a.extra_benches.split(",") if t]
    assert not a.score_benches or x_tags, "--score_benches requires --extra_benches"
    # --report_benches, benchmarks that are only reported; their candidate segments are only scored for the held-out match and do not enter training
    r_tags = [t for t in a.report_benches.split(",") if t]
    data_r = []
    for tag in r_tags:
        assert not member_runs, "--report_benches cannot be combined with --member_runs"
        bx = bench_bundle(bundle, tag)
        oof_r = dict(np.load(os.path.join(d, "loso_oof_proba_eval_%s.npz" % tag)))
        sw_r = switch_bench(tag)
        dr = {}
        for g in games:
            M, gt = load_game_full(g, bx)
            assert oof_r[g].shape[0] == len(gt), "%s %s, the number of frames of the out-of-fold probabilities does not match the rebuilt degradation" % (tag, g)
            MEMBER_P = []
            SWITCH_CUR = pick_switch(sw_r, g)
            dr[g] = prep_game(M, gt, oof_r[g].astype(np.float64), touch_set, a.kmax, base_costs, [])
        log("reported benchmark %s loaded (not used in training), mean spellF1 of the Viterbi baseline %.4f" % (tag, float(np.mean([score_pred(dr[g]["gt"], dr[g]["base"])["spell_f1"] for g in games]))))
        data_r.append(dr)
    # the benchmarks to score, the main benchmark plus (with --score_benches) the extra benchmarks; the report-only ones are kept apart
    s_data = [data] + (data_x if a.score_benches else [])
    s_tags = [bundle_main_tag(bundle)] + (x_tags if a.score_benches else [])
    all_data = s_data + data_r

    # train the scorer leave-one-match-out to get the matching probability of every segment of every match; with several IoU targets one scorer per target, probabilities averaged
    rngs = [np.random.default_rng(0)] + [np.random.default_rng(100 + sd) for sd in range(1, a.scorer_seeds)]   # seed 0 for the first scorer
    probs = {}
    probs_all = [dict() for _ in all_data]       # matching probabilities of every segment of every match for benchmark j (in the order of s_data plus data_r); j 0 is probs
    for n, h_out in enumerate(games):
        prs = [{h: [np.zeros(len(dd[h_out]["seg"][h][0])), np.zeros(len(dd[h_out]["seg"][h][0]))] for h in dd[h_out]["blocks"]} for dd in all_data]
        pr = prs[0]
        n_rows = n_pos = 0
        for thr in target_ious:
          for sd in range(a.scorer_seeds):
            rng = rngs[sd]
            Xs, ys, ws = [], [], []
            for g in games:
                if g == h_out:
                    continue
                for dd in [data] + data_x:
                  for h in dd[g]["blocks"]:
                    _, _, fH, fA, _, _ = dd[g]["seg"][h]
                    for f, t in ((fH, dd[g]["tgt"][h][0]), (fA, dd[g]["tgt"][h][1])):
                        y = (t >= thr)
                        keep = y | (rng.random(len(y)) < a.neg_frac)
                        Xs.append(f[keep]); ys.append(y[keep]); ws.append(np.where(y[keep], 1.0, 1.0 / a.neg_frac) * src_weight_of(g, src_w))
            X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
            clf = HistGradientBoostingClassifier(random_state=sd, **sp)
            clf.fit(X, y, sample_weight=w)
            n_rows += len(y); n_pos += int(y.sum())
            for dd, pr_ in zip(all_data, prs):
              for h in dd[h_out]["blocks"]:
                _, _, fH, fA, _, _ = dd[h_out]["seg"][h]
                pr_[h][0] += clf.predict_proba(fH)[:, 1] / (len(target_ious) * a.scorer_seeds)
                pr_[h][1] += clf.predict_proba(fA)[:, 1] / (len(target_ious) * a.scorer_seeds)
        probs[h_out] = {h: tuple(v) for h, v in pr.items()}
        for j, pr_ in enumerate(prs):
            probs_all[j][h_out] = {h: tuple(v) for h, v in pr_.items()}
        log("scorer %d/%d, held out %s, IoU targets %s, %d training rows in total (%d positives)" % (n + 1, len(games), h_out, target_ious, n_rows, n_pos))

    thetas = [float(x) for x in a.thetas.split(",")]
    gammas = [float(x) for x in a.gammas.split(",")]
    # scores of every match under all (θ, γ)
    grid = {}
    for g in games:
        pH = {h: probs[g][h][0] for h in data[g]["blocks"]}; pA = {h: probs[g][h][1] for h in data[g]["blocks"]}
        for th in thetas:
            for ga in gammas:
                pred = decode_game(data[g], pH, pA, th, ga, a.kmax)
                grid[(g, th, ga)] = score_pred(data[g]["gt"], pred)
        log("%s decoding grid done" % g)
    base_s = {g: score_pred(data[g]["gt"], data[g]["base"]) for g in games}
    # the same grid for the other benchmarks (extra benchmarks of --score_benches, --report_benches)
    grids = [grid]
    for j in range(1, len(all_data)):
        dd, pj, gj = all_data[j], probs_all[j], {}
        for g in games:
            pH = {h: pj[g][h][0] for h in dd[g]["blocks"]}; pA = {h: pj[g][h][1] for h in dd[g]["blocks"]}
            for th in thetas:
                for ga in gammas:
                    gj[(g, th, ga)] = score_pred(dd[g]["gt"], decode_game(dd[g], pH, pA, th, ga, a.kmax))
        grids.append(gj)
        log("benchmark %s decoding grid done" % (s_tags + r_tags)[j])
    n_s = len(s_data)

    def sel_score(o, th, ga):                      # the score used to pick θγ; with only the main benchmark it is that benchmark's own score
        if n_s == 1:
            return grid[(o, th, ga)]["spell_f1"]
        return float(np.mean([grids[j][(o, th, ga)]["spell_f1"] for j in range(n_s)]))
    # θ and γ are picked on the matches other than the held-out one
    chosen, per_new = {}, {}
    for g in games:
        others = [x for x in games if x != g]
        best = max(((th, ga) for th in thetas for ga in gammas), key=lambda tg: np.mean([sel_score(o, tg[0], tg[1]) for o in others]))
        chosen[g] = best
        per_new[g] = grid[(g, best[0], best[1])]
    fixed = (0.3, 0.0) if 0.3 in thetas and 0.0 in gammas else (thetas[0], gammas[0])
    per_fixed = {g: grid[(g, fixed[0], fixed[1])] for g in games}
    per_bench = {}
    if len(all_data) > 1:
        for j, tag in enumerate(s_tags + r_tags):
            dd = all_data[j]
            bj = {g: score_pred(dd[g]["gt"], dd[g]["base"]) for g in games}
            nj = {g: grids[j][(g, chosen[g][0], chosen[g][1])] for g in games}
            per_bench[tag] = {"role": "score" if j < n_s else "report", "base_per_game": bj, "new_per_game": nj}
        if n_s > 1:                               # baseline, new decoding and fixed in the JSON and on screen become means over the scored benchmarks
            base_s = {g: avg_scores([per_bench[t]["base_per_game"][g] for t in s_tags]) for g in games}
            per_new = {g: avg_scores([per_bench[t]["new_per_game"][g] for t in s_tags]) for g in games}
            per_fixed = {g: avg_scores([grids[j][(g, fixed[0], fixed[1])] for j in range(n_s)]) for g in games}

    def summary(per):
        return {"mean_spell_f1": float(np.mean([per[g]["spell_f1"] for g in games])), "mean_macro_f1": float(np.mean([per[g]["macro_f1"] for g in games])),
                "per_iou": {k: float(np.mean([per[g]["per_iou"][k] for g in games])) for k in next(iter(per.values()))["per_iou"]},
                "hit_by_length": {k: float(np.nanmean([per[g]["hit_by_length"][k] for g in games])) for k in next(iter(per.values()))["hit_by_length"]},
                "n_pred": int(sum(per[g]["n_pred"] for g in games)), "n_gt": int(sum(per[g]["n_gt"] for g in games))}

    S_b, S_n, S_f = summary(base_s), summary(per_new), summary(per_fixed)
    for tag, pb in per_bench.items():
        pb["base"], pb["new"] = summary(pb["base_per_game"]), summary(pb["new_per_game"])
        dj = np.array([pb["new_per_game"][g]["spell_f1"] - pb["base_per_game"][g]["spell_f1"] for g in games])
        print("benchmark %-9s (%s) Viterbi %.4f  new decoding %.4f (diff %+.4f, positive in %d/%d matches)" % (
            tag, "scored" if pb["role"] == "score" else "report only", pb["base"]["mean_spell_f1"], pb["new"]["mean_spell_f1"], dj.mean(), int((dj > 0).sum()), len(games)))
    if n_s > 1:
        print("the baseline, new decoding and fixed values below are means over the scored benchmarks %s" % ",".join(s_tags))
    print()
    print("%-16s %8s %8s %8s   %8s %8s   %s" % ("match", "baseline", "new", "diff", "fixed θ", "diff", "picked θ,γ"))
    for g in games:
        print("%-16s %8.4f %8.4f %+8.4f   %8.4f %+8.4f   %s" % (g, base_s[g]["spell_f1"], per_new[g]["spell_f1"], per_new[g]["spell_f1"] - base_s[g]["spell_f1"],
                                                            per_fixed[g]["spell_f1"], per_fixed[g]["spell_f1"] - base_s[g]["spell_f1"], chosen[g]))
    dn = np.array([per_new[g]["spell_f1"] - base_s[g]["spell_f1"] for g in games])
    df = np.array([per_fixed[g]["spell_f1"] - base_s[g]["spell_f1"] for g in games])
    print("mean spellF1  baseline %.4f  new decoding %.4f (diff %+.4f, positive in %d/%d matches)  fixed θ%.2f γ%.2f %.4f (diff %+.4f, positive in %d/%d matches)" % (
        S_b["mean_spell_f1"], S_n["mean_spell_f1"], dn.mean(), int((dn > 0).sum()), len(games), fixed[0], fixed[1], S_f["mean_spell_f1"], df.mean(), int((df > 0).sum()), len(games)))
    print("mean macroF1  baseline %.4f  new decoding %.4f  fixed %.4f" % (S_b["mean_macro_f1"], S_n["mean_macro_f1"], S_f["mean_macro_f1"]))
    print("F1 per IoU       baseline     %s" % " ".join("%s %.3f" % (k, v) for k, v in S_b["per_iou"].items()))
    print("                 new decoding %s" % " ".join("%s %.3f" % (k, v) for k, v in S_n["per_iou"].items()))
    print("hit rate by len  baseline     %s" % " ".join("%s %.2f" % (k, v) for k, v in S_b["hit_by_length"].items()))
    print("                 new decoding %s" % " ".join("%s %.2f" % (k, v) for k, v in S_n["hit_by_length"].items()))
    print("predicted spells in total  baseline %d  new decoding %d  ground truth %d" % (S_b["n_pred"], S_n["n_pred"], S_b["n_gt"]))
    if a.save_pred and a.save_pred.lower() != "none":
        pred_path = os.path.join(RUNS, "_decode", "%s_pred.npz" % a.run) if a.save_pred == "auto" else os.path.expanduser(a.save_pred)
        os.makedirs(os.path.dirname(os.path.abspath(pred_path)), exist_ok=True)
        preds = {}
        for g in games:
            th, ga = chosen[g]
            pH = {h: probs[g][h][0] for h in data[g]["blocks"]}; pA = {h: probs[g][h][1] for h in data[g]["blocks"]}
            preds[g] = decode_game(data[g], pH, pA, th, ga, a.kmax).astype(np.int8)
        np.savez_compressed(pred_path, **preds)
        log("new decoding of every held-out match saved to %s" % pred_path)
    if a.out:
        out = {"run": a.run, "games": games, "label_min_spell_idsse": a.label_min_spell, "kmax": a.kmax, "touch_set": touch_set, "target_ious": target_ious, "scorer_params": sp, "neg_frac": a.neg_frac, "feat_set": FEAT_SET, "n_feat": len(FEAT_NAMES), "scorer_seeds": a.scorer_seeds, "src_weight": src_w, "cand_extra": CAND_EXTRA, "member_runs": member_runs, "extra_benches": a.extra_benches, "base_costs": base_costs, "cost_grid": COST_GRID,
               "base": {"summary": S_b, "per_game": base_s}, "new": {"summary": S_n, "per_game": per_new, "chosen": {g: list(chosen[g]) for g in games}},
               "fixed": {"theta_gamma": list(fixed), "summary": S_f, "per_game": per_fixed},
               "grid_mean": {"%g,%g" % (th, ga): float(np.mean([sel_score(g, th, ga) for g in games])) for th in thetas for ga in gammas},
               "span_need": {"median": float(np.median(sn)), "p90": float(np.percentile(sn, 90)), "p99": float(np.percentile(sn, 99)), "frac_over_kmax": float((sn > a.kmax).mean())}}
        if a.switch_run:                              # recorded only when used
            out.update({"switch_run": a.switch_run, "switch_main_tag": a.switch_main_tag, "switch_tau": a.switch_tau, "switch_hw": a.switch_hw})
        if a.score_benches or r_tags:                 # recorded only when used
            out.update({"score_benches": bool(a.score_benches), "score_bench_tags": s_tags, "report_benches": a.report_benches, "per_bench": per_bench})
        if ALT_SPEC:                                  # --alt_labels, per match scores of the same decoding on the alternative label set, for every benchmark
            alt = {}
            for j, tag in enumerate(s_tags + r_tags):
                dd, pj = all_data[j], probs_all[j]
                per = {}
                for g in games:
                    th, ga = chosen[g]
                    pH = {h: pj[g][h][0] for h in dd[g]["blocks"]}; pA = {h: pj[g][h][1] for h in dd[g]["blocks"]}
                    pred = decode_game(dd[g], pH, pA, th, ga, a.kmax)
                    per[g] = score_pred(alt_gt(g, dd[g]["gt"]), pred)
                    own = score_pred(dd[g]["gt"], pred)["spell_f1"]
                    assert abs(own - grids[j][(g, th, ga)]["spell_f1"]) < 1e-9, "%s %s, the re-decoded score does not match the one in the grid" % (tag, g)
                alt[tag] = {"new_per_game": per}
                log("alternative labels %s scored on %s, mean over matches %.4f" % (a.alt_labels, tag, float(np.mean([per[g]["spell_f1"] for g in games]))))
            out.update({"alt_labels": a.alt_labels, "alt": alt})
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False, default=str)
        log("written to %s" % a.out)


if __name__ == "__main__":
    main()
