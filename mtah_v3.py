"""
mtah_v3, training of the per frame gradient boosted tree classifier and of the tree change-point model on public matches, leave-one-match-out evaluation, then prediction of BAR-ATH
============================
The training matches are degraded copies of public tracking (build_sources), the features come from possession/mtah_features.py, the classifier and the
three-state Viterbi from mtah_v2. Every match is held out in turn; the held-out probabilities are saved for the ensembling and segment decoding scripts,
the held-out score of every match is reported, and the Viterbi costs are chosen by the mean held-out spellF1 over the matches. Sources
  metrica      matches 1, 2 and 3 of the Metrica public sample data (adapters/metrica_sample, ground truth from the event chain). In this dataset the ball is almost never shown during dead balls
               (visible on about 1 to 10 percent of the dead-ball frames), whereas automatic video tracking often keeps the ball during dead balls (in the SkillCorner open data the ball
               is present on about 44 percent of the dead-ball frames). The ball is therefore removed at random on additional live-ball frames, which weakens the shortcut "ball missing means dead ball".
  idsse        the seven IDSSE matches (adapters/idsse, official frame-level possession and ball status).
  skillcorner  optional, the SkillCorner open data (adapters/skillcorner), for --train_only.
With --rand_degrade every training copy draws its degradation from public ranges (degrade_random.py); --rand_bench and --extra_eval add held-out benchmarks.
Without --no_bar the model is trained on all matches and predicts BAR-ATH. Only summary numbers are printed.

Inputs  --name run name, --sources metrica,idsse, --seeds N, the degradation and feature switches described in the argument help
Outputs  metrics.json, model.joblib, versions.json and the held-out probabilities under $MTAH_RUNS/<name>/ (default ~/Data/more_than_a_hack_2027/runs), plus <name>.csv and proba_bar_ath.npy without --no_bar

Usage
  python -u mtah_v3.py --name tree_loso --sources metrica,idsse --seeds 1 --realistic --noise 0.01 --touch_set clean --touch_feats --gap_feats \
      --idsse_min_spell 25 --no_bar --augment swap --hgb max_iter=1500,learning_rate=0.03,max_leaf_nodes=63,min_samples_leaf=100 \
      --rand_bench 3 --rand_degrade --rand_main --extra_eval 3:15,0:0,4:21 --setp_feats --shape_feats
  (the other training commands are in the README)

Last modified 2026-11-15
"""
import argparse
import itertools
import json
import os
import platform
import sys
import time

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtah_v2 import boundary_labels, fit, fit_two_stage, spell_weights, to_hadx, viterbi3, viterbi3_tv  # noqa: E402
from possession.adapters.metrica_sample import load_game  # noqa: E402
from possession.adapters.mtah2027 import N_FRAMES, load  # noqa: E402
from possession.mtah_features import build_features  # noqa: E402
from possession.spell_metrics import binned_hit_rates, codes_from_hadx, evaluate, merge_short_spells, spells, valid_runs, validate  # noqa: E402

T0 = time.time()


def ar1_noise(rng, shape, sigma, phi=0.9):
    """AR(1) noise along axis 0 with stationary standard deviation sigma (same definition as possession.degrade.ar1_noise)"""
    e = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), shape)
    out = np.empty(shape)
    out[0] = rng.normal(0, sigma, shape[1:])
    for t in range(1, shape[0]):
        out[t] = phi * out[t - 1] + e[t]
    return out
OUT = {}
BUILD_T = 0.0


def rep(k, v):
    if hasattr(v, "item"):
        v = v.item()
    OUT[k] = v
    print("%s = %s" % (k, ("%.4f" % v) if isinstance(v, float) else v), flush=True)


def augment_metrica(M, gt, seed, noise_m=0.3, alive_missing=0.30, realistic=False, ball_jitter=0.0, impute_sigma=0.0, static_wander=0.0):
    """Fixed degradation of the Metrica sample data (the fixed benches), adds noise and removes the ball on additional live-ball frames up to the target share alive_missing.
    With realistic the gaps come from possession.degrade.realistic_missing; otherwise their lengths are log-normal with the median (268 frames) and p90 (845 frames) of the ball gaps
    of the competition tracking, a label-free statistic"""
    rng = np.random.default_rng(seed)
    M2 = dict(M)
    T = len(gt)
    P = M["player_xy"].astype(np.float32)
    b = M["ball_xy"].copy()
    if static_wander > 0:
        b = static_ball_wander(rng, b, M["fps"], static_wander, P=M["player_xy"])
    if ball_jitter > 0:
        from possession.degrade import ball_offset
        b = ball_offset(rng, b, P, M["fps"], ball_jitter)
    alive = (gt == 1) | (gt == 2)
    vis = ~np.isnan(b[:, 0])
    if realistic:
        from possession.degrade import player_dropout, realistic_missing
        # the ball is not shown during dead balls anyway (a convention of this dataset), so occlusion gaps are simulated on live-ball frames only and then merged with the frames that were already missing
        missing = realistic_missing(rng, b, P, M["fps"], alive & vis, alive, alive_missing) | ~vis
        P = player_dropout(rng, P, M["graded"])
    else:
        target = alive_missing * alive.sum()
        missing = ~vis
        sigma = np.log(845.0 / 268.0) / 1.2816
        cand = np.flatnonzero(alive & vis)
        guard = 0
        while (missing & alive).sum() < target and guard < 100000:
            guard += 1
            start = int(rng.choice(cand))
            length = int(np.exp(np.log(268.0) + sigma * rng.standard_normal()))
            missing[start:start + max(length, 3)] = True
    if noise_m > 0:
        P = P + ar1_noise(rng, P.shape, noise_m).astype(np.float32)
        b = b + ar1_noise(rng, b.shape, noise_m)
    b[missing] = np.nan
    if impute_sigma > 0:   # optional, removed ball positions are filled with the true position plus a slow drift error (same as degrade_like_wc)
        imp = M["ball_xy"] + ar1_noise(rng, b.shape, impute_sigma, phi=0.98)
        fill = missing & vis
        b[fill] = imp[fill]
    M2["player_xy"], M2["ball_xy"], M2["ball_visible"] = P, b, ~np.isnan(b[:, 0])
    return M2


def smooth_ball_track(b, w):
    """Moving average of window w frames over each run of consecutive visible ball frames. The raw IDSSE ball changes speed more sharply at touches than the ball of automatic tracking,
    and touch detection relies on speed changes, so smoothing brings the touch features of IDSSE closer to those of automatic tracking.
    The fixed benches use windows of 0, 15 and 21 frames; this smoothing level was set from a label-free comparison of the p90 of the second difference of the ball between raw IDSSE (about 26.5 m/s^2, the median over the seven matches, 25.8 to 53.8) and the competition tracking.
    degrade_random draws the window from a range set on public sources only (RANGES["smooth"] there)"""
    import pandas as pd
    b = np.asarray(b, np.float64).copy()
    vis = ~np.isnan(b[:, 0])
    e = np.diff(np.r_[0, vis.astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if z - a >= 3:
            b[a:z] = pd.DataFrame(b[a:z]).rolling(w, center=True, min_periods=1).mean().values
    return b


def static_ball_wander(rng, b, fps, sigma, P=None, disp_thresh=0.5, keep_frac=0.1, smooth_s=0.5, pause_p=0.3, reach_m=15.0):
    """Turns a still ball into one that rolls slowly between nearby players. In a label-free comparison of the tracking, on frames where the players are slow and the ball is visible, the ball of the competition tracking keeps moving, is rarely static,
    stays close to the nearest player and has a small second difference; on the dead-ball frames of the SkillCorner open data the median 1 s ball displacement is 1.6 m. On frames where the players are slow and the ball is visible IDSSE has a median 1 s displacement of 0.02 m,
    65 percent of static frames and 3.8 m to the nearest player, so a model trained on raw IDSSE can learn that a still ball means a dead ball, which does not hold for automatic tracking.
    Method, for every run of consecutive still frames (1 s displacement below disp_thresh and ball visible). A player within reach_m metres is picked repeatedly as the target and the ball rolls towards him at a log-normal speed (median sigma m/s);
    the target changes once the ball is within 0.8 m or after 4 s, and before each roll the ball pauses for 0.5 to 2 s with probability pause_p. The speed series is smoothed with a moving average of smooth_s seconds, so that turns do not create unbounded acceleration, and the position is
    the integral of the speed. A share keep_frac of the still segments is left as it is.
    The shape constants (disp_thresh, smooth_s, pause_p, reach_m, the 0.8 m target radius, the 4 s limit, the 0.5 to 2 s pauses and the log-normal spread 0.4 of the speed) were set in the label-free comparison
    with the competition tracking that the technical document mentions and are fixed in every training copy; the 5 m step in a random direction when no player is within reach_m is a fixed default as well.
    The fixed benches use sigma 3, 0 (no rolling) and 4 with keep_frac 0.1, also set from that comparison; degrade_random draws sigma and keep_frac for every training copy from ranges set on public sources only
    (RANGES["wander"] and RANGES["wander_keep"] there)"""
    import pandas as pd
    b = np.asarray(b, np.float64).copy()
    T = len(b)
    vis = ~np.isnan(b[:, 0])
    lag = int(round(fps))
    disp = np.full(T, np.nan)
    disp[lag:] = np.linalg.norm(b[lag:] - b[:-lag], axis=1)
    static = vis & (np.nan_to_num(disp, nan=np.inf) < disp_thresh)
    e = np.diff(np.r_[0, static.astype(np.int8), 0])
    Pf = None if P is None else np.asarray(P, np.float64)
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if rng.random() < keep_frac:
            continue
        n = z - a
        vel = np.zeros((n, 2))
        pos = b[a].copy()
        t = 0
        while t < n:
            if rng.random() < pause_p:
                t += int(rng.uniform(0.5, 2.0) * fps)
                continue
            spd = sigma * np.exp(rng.normal(0.0, 0.4))
            target = None
            if Pf is not None:
                cand = Pf[a + t]
                ok = ~np.isnan(cand[:, 0])
                if ok.any():
                    dd = np.linalg.norm(cand[ok] - pos, axis=1)
                    near = np.flatnonzero(dd < reach_m)
                    if len(near):
                        target = cand[ok][rng.choice(near)]
            if target is None:
                ang = rng.uniform(0, 2 * np.pi)
                target = pos + 5.0 * np.array([np.cos(ang), np.sin(ang)])
            dur = 0
            while t < n and dur < 4 * fps:
                vec = target - pos
                dist = np.linalg.norm(vec)
                if dist < 0.8:
                    break
                vel[t] = vec / dist * spd
                pos = pos + vel[t] / fps
                t += 1
                dur += 1
        vel = pd.DataFrame(vel).rolling(max(int(round(fps * smooth_s)), 1), center=True, min_periods=1).mean().values
        b[a:z] = b[a] + np.cumsum(vel, axis=0) / fps
    return b


def degrade_like_wc(M, gt, seed, noise_m=0.3, miss_frac=0.41, realistic=False, keep_missing=False, impute_sigma=0.0, static_wander=0.0, ball_smooth=0):
    """Applies the fixed degradation of the three fixed benches to IDSSE (and to the optional SkillCorner source); augment_metrica does the same for the Metrica sample data.
    The input is optical tracking without missing ball; the function removes the ball in segments and adds AR(1) noise.
    The default miss_frac 0.41 and the log-normal gap lengths of the non-realistic mode (median 268, p90 845 frames, fast balls lost more easily) are label-free statistics of the competition tracking;
    with realistic the gaps come from possession.degrade.realistic_missing. With impute_sigma above 0 the removed ball is not set to NaN but filled with the true ball position plus a slow drift error
    (AR(1), phi 0.98), as if produced by a ball imputer (public matches only, BAR-ATH has no true ball position)"""
    rng = np.random.default_rng(seed)
    M2 = dict(M)
    T = len(gt)
    graded = M["graded"]
    b = M["ball_xy"].copy()
    if ball_smooth > 1:
        b = smooth_ball_track(b, ball_smooth)
    if static_wander > 0:
        b = static_ball_wander(rng, b, M["fps"], static_wander, P=M["player_xy"])
    v = np.r_[0, np.linalg.norm(np.nan_to_num(b[1:]) - np.nan_to_num(b[:-1]), axis=1) * M["fps"]]
    weight = np.where(graded & ~np.isnan(b[:, 0]), 1.0 + 2.0 * (v > 8.0), 0.0)
    weight /= weight.sum()
    P = M["player_xy"].astype(np.float32)
    if realistic:
        from possession.degrade import player_dropout, realistic_missing
        # when keep_missing is true (sources such as SkillCorner that already miss the ball) the frames that are already missing count towards the share, which is topped up to miss_frac
        missing = realistic_missing(rng, b, P, M["fps"], graded & ~np.isnan(b[:, 0]), graded, miss_frac,
                                    existing=(np.isnan(b[:, 0]) & graded) if keep_missing else None)
        P = player_dropout(rng, P, graded)
    else:
        missing = np.zeros(T, bool)
        target = miss_frac * graded.sum()
        sigma = np.log(845.0 / 268.0) / 1.2816
        guard = 0
        while missing[graded].sum() < target and guard < 200000:
            guard += 1
            start = int(rng.choice(T, p=weight))
            length = int(np.exp(np.log(268.0) + sigma * rng.standard_normal()))
            missing[start:start + max(length, 3)] = True
    if noise_m > 0:
        P = P + ar1_noise(rng, P.shape, noise_m).astype(np.float32)
        b = b + ar1_noise(rng, b.shape, noise_m)
    b[missing] = np.nan
    if impute_sigma > 0:
        imp = M["ball_xy"] + ar1_noise(rng, b.shape, impute_sigma, phi=0.98)
        fill = missing & ~np.isnan(M["ball_xy"][:, 0])
        b[fill] = imp[fill]
    b[~graded] = np.nan
    M2["player_xy"], M2["ball_xy"], M2["ball_visible"] = P, b, ~np.isnan(b[:, 0])
    return M2


IDSSE_ROOT = "~/Data/alt_training/idsse"
SC_ROOT = os.environ.get("MTAH_SC_ROOT", "~/Data/alt_training/skillcorner_opendata")   # SkillCorner open data (the directory that contains matches/), can be redirected with the MTAH_SC_ROOT environment variable


def vis_speed_weights(M, target, edges=(3.0, 8.0)):
    """Sample weights of visible frames by ball-speed bin; the weight is the target share divided by the share in this source.
    The ball speed comes from the 3-frame displacement of the raw ball position; frames where it cannot be computed and frames with the ball missing get weight 1. Only input statistics are used, no ground truth"""
    b = np.asarray(M["ball_xy"], np.float64)
    T = len(b)
    vis = ~np.isnan(b[:, 0])
    v = np.full(T, np.nan)
    ok = vis.copy()
    ok[3:] &= vis[:-3]
    ok[:3] = False
    v[3:] = np.linalg.norm(b[3:] - b[:-3], axis=1) * M["fps"] / 3
    v[~ok] = np.nan
    m = M["graded"] & ok
    k = np.digitize(np.nan_to_num(v), edges)
    w = np.ones(T)
    for i, tgt in enumerate(target):
        share = (k[m] == i).mean() if m.any() else 0.0
        if share > 0:
            w[m & (k == i)] = tgt / share
    return w


def _swap_teams(M, gt):
    """Copy with home and away teams swapped (slot teams 1 and 2 exchanged, ground truth H and A exchanged); the other arrays are shared"""
    M2 = dict(M)
    st = np.asarray(M["slot_team"])
    M2["slot_team"] = np.where(st == 1, 2, np.where(st == 2, 1, st)).astype(st.dtype)
    M2["team_names"] = tuple(reversed(M.get("team_names", ("home", "away"))))
    gt2 = np.where(gt == 1, 2, np.where(gt == 2, 1, gt)).astype(gt.dtype)
    return M2, gt2


def _mirror_y(M, gt):
    """Copy mirrored along the long axis of the pitch (y becomes W minus y); teams and ground truth are unchanged"""
    M2 = dict(M)
    W = M["pitch"][1]
    P = np.array(M["player_xy"], copy=True); P[:, :, 1] = W - P[:, :, 1]
    B = np.array(M["ball_xy"], copy=True); B[:, 1] = W - B[:, 1]
    M2["player_xy"], M2["ball_xy"] = P, B
    return M2, gt.copy()


def augment_sets(out, augment):
    """Training-set augmentation. augment is a comma-separated list, swap exchanges home and away, mirror mirrors along the long axis, both does both;
    the copies keep the name of the original match and get seed plus 1000, 2000, 3000, so in leave-one-match-out they are held out together with the original match, and prediction uses only the original with seed 0"""
    kinds = [k for k in augment.split(",") if k]
    extra = []
    for st in out:
        if "swap" in kinds:
            M2, g2 = _swap_teams(st["M"], st["gt"]); extra.append({"game": st["game"], "seed": st["seed"] + 1000, "M": M2, "gt": g2, "aug": "swap"})
        if "mirror" in kinds:
            M2, g2 = _mirror_y(st["M"], st["gt"]); extra.append({"game": st["game"], "seed": st["seed"] + 2000, "M": M2, "gt": g2, "aug": "mirror"})
        if "both" in kinds:
            M2, g2 = _swap_teams(*_mirror_y(st["M"], st["gt"])); extra.append({"game": st["game"], "seed": st["seed"] + 3000, "M": M2, "gt": g2, "aug": "both"})
    return out + extra


def build_sources(names, seeds, noise, alive_missing, idsse_min_spell=50, realistic=False, sc_matches=20, metrica_jitter=0.0, impute_sigma=0.0, static_wander=0.0, ball_smooth=0, augment="", rand_mode=""):
    """Returns [{game, seed, M, gt}] for the sources metrica, idsse and skillcorner; with a non-empty augment the augmented copies are added (see augment_sets).
    With a non-empty rand_mode Metrica and IDSSE use the randomised degradation of degrade_random and the degradation parameters above are ignored.
    With train, the copy of seed s draws its parameters with draw_train(game, s); rnd0, rnd1 and rnd2 are the three randomised benchmarks (only with seeds 1). Empty rand_mode means the fixed degradation"""
    out = []
    if rand_mode:
        import degrade_random as DR
        assert set(names) <= {"metrica", "idsse"}, "randomised degradation only supports metrica and idsse, got %s" % names
        assert rand_mode == "train" or seeds == 1, "benchmark %s requires seeds 1" % rand_mode
    for name in names:
        if name == "metrica":
            for g in (1, 2, 3):
                M0, gt = load_game("~/Data/alt_training/metrica_sample", g)
                for s in range(seeds):
                    if rand_mode:
                        prm, dseed = DR.params_for("metrica%d" % g, "train:%d" % s if rand_mode == "train" else rand_mode)
                        out.append({"game": "metrica%d" % g, "seed": s, "M": DR.degrade(M0, gt, "metrica%d" % g, prm, dseed, "metrica"), "gt": gt})
                        continue
                    out.append({"game": "metrica%d" % g, "seed": s, "M": augment_metrica(M0, gt, s, noise, alive_missing, realistic, metrica_jitter, impute_sigma, static_wander), "gt": gt})
        elif name == "idsse":
            from possession.adapters.idsse import list_matches, load_match
            for mid in list_matches(IDSSE_ROOT):
                M0, gt = load_match(IDSSE_ROOT, mid, min_spell=idsse_min_spell)
                for s in range(seeds):
                    if rand_mode:
                        prm, dseed = DR.params_for("idsse_" + mid, "train:%d" % s if rand_mode == "train" else rand_mode)
                        out.append({"game": "idsse_" + mid, "seed": s, "M": DR.degrade(M0, gt, "idsse_" + mid, prm, dseed, "idsse"), "gt": gt})
                        continue
                    out.append({"game": "idsse_" + mid, "seed": s, "M": degrade_like_wc(M0, gt, s, noise, realistic=realistic, impute_sigma=impute_sigma, static_wander=static_wander, ball_smooth=ball_smooth), "gt": gt})
        elif name == "skillcorner":
            from possession.adapters.skillcorner import list_matches as sc_list, load_match as sc_load
            for mid in sc_list(SC_ROOT)[:sc_matches]:
                M0, gt = sc_load(SC_ROOT, mid)
                for s in range(seeds):
                    out.append({"game": "skillcorner_" + mid, "seed": s, "M": degrade_like_wc(M0, gt, s, noise, realistic=realistic, keep_missing=True), "gt": gt})
        else:
            raise ValueError(name)
    if augment:
        out = augment_sets(out, augment)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, the output directory under $MTAH_RUNS")
    ap.add_argument("--sources", default="metrica,idsse", help="comma-separated sources, metrica, idsse and (for --train_only) skillcorner")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--alive_missing", type=float, default=0.30)
    ap.add_argument("--grid", default="8,16,32;8,16,32;6,12,24", help="c_team;c_dead;c_alive values of the quick Viterbi cost search of this script (27 combinations); the cost search reported in the technical document is rescore_oof.py --grid_spec")
    ap.add_argument("--costs", default=None)
    ap.add_argument("--skip_loso", action="store_true")
    ap.add_argument("--loso_games", default="", help="run only the leave-one-match-out folds of these matches, in the given order, comma-separated; requires --no_bar; empty means all")
    ap.add_argument("--idsse_min_spell", type=int, default=50, help="possession spells shorter than this many frames in the IDSSE ground truth are merged into the previous spell (training labels and the held-out scores printed by this script; spell_decode.py and rescore_oof.py score on raw labels), 0 means no merging")
    ap.add_argument("--realistic", action="store_true", help="remove the ball the way automatic tracking loses it (possession.degrade.realistic_missing, gaps start when the ball is slow and close to a player, the ball reappears once it moves fast) and drop players at random")
    ap.add_argument("--boundary", action="store_true", help="also train a boundary detector (frames within bhalf frames of a ground-truth change of possession or of a switch between dead and live ball); the Viterbi transition cost then varies with it")
    ap.add_argument("--bhalf", type=int, default=6)
    ap.add_argument("--boundary_keep", action="store_true", help="train the boundary detector in the full training as well, even when k is 0; saves boundary_model.joblib and the BAR-ATH change-point probabilities proba_boundary_bar_ath.npy for the change-point head of the segment decoding")
    ap.add_argument("--boundary_kind", default="all", choices=("all", "team", "dead"), help="which kind of switch the boundary detector learns, all learns both kinds (default), team only direct changes of possession between home and away, dead only switches into and out of dead ball")
    ap.add_argument("--bhgb", default="", help="hyperparameters of the boundary detector, same syntax as --hgb, default mtah_v2.HGB_DEFAULT")
    ap.add_argument("--kgrid", default="0,2,4,8", help="values of the boundary detector strength k")
    ap.add_argument("--min_spell_all", type=int, default=0, help="possession spells shorter than this many frames in the ground truth of all sources are merged into the previous spell (one common definition of how short a spell does not count as a change of possession), 0 means no merging")
    ap.add_argument("--spell_weight", type=float, default=0.0, help="exponent alpha of the weighting by spell length, 0 means no weighting")
    ap.add_argument("--trans_weight", type=float, default=0.0, help="sample weights within --trans_win frames of a change point (two adjacent frames with different labels) are multiplied by 1 plus this value; 0 is off")
    ap.add_argument("--trans_win", type=int, default=12, help="half window of --trans_weight, in frames")
    ap.add_argument("--min_spell_src", default="", help="merge short spells per source, e.g. idsse=25,metrica=13 (game name prefix=frames), overrides --min_spell_all; the evaluation labels are merged too, so use rescore_oof.py to rescore against the original labels when comparing thresholds")
    ap.add_argument("--gap_feats", action="store_true", help="add features describing which player the ball is with inside ball gaps")
    ap.add_argument("--touch_feats", action="store_true", help="add touch sequence features (team and time of the previous and the next touch, and so on)")
    ap.add_argument("--kin_feats", action="store_true", help="add 11 player-ball kinematic features (which team the ball is heading to, how soon it arrives, near whom it lands 0.5 s later)")
    ap.add_argument("--dir_feats", action="store_true", help="add 32 movement direction features (possession/mtah_features.py dir_features; off by default)")
    ap.add_argument("--comove_feats", action="store_true", help="add 9 player-ball co-movement features (possession/mtah_features.py comove_features; off by default)")
    ap.add_argument("--shape_feats", action="store_true", help="add 53 team shape behaviour features (possession/mtah_features.py shape_features; off by default)")
    ap.add_argument("--setp_feats", action="store_true", help="add 49 set-piece positioning features (possession/mtah_features.py setp_features; off by default)")
    ap.add_argument("--tta_feats", action="store_true", help="add 4 time-to-arrival features (fastest time to the ball of each team, their difference, share of home players among the three fastest; computed from the most recent ball position when the ball is missing)")
    ap.add_argument("--train_extra_degrade", default="", help="also train on copies made with other fixed degradations, comma-separated list of wander:smooth, e.g. 0:0,4:21 (training only, held-out matches still use the main benchmark)")
    ap.add_argument("--touch_set", default="robust", help="touch detection threshold set, robust (for 0.3 m noise) or clean (for jitter of the order of 0.01 m)")
    ap.add_argument("--vis_speed_weight", default="", help="match the distribution of visible frames over ball-speed bins (<3, 3 to 8, >8 m/s) to target shares, three comma-separated values; empty means off")
    ap.add_argument("--rand_degrade", action="store_true", help="the training copies of Metrica and IDSSE use the randomised degradation of degrade_random, each of the --seeds copies draws its own parameter set from public ranges, and --realistic, --noise, --static_wander, --ball_smooth and --alive_missing do not affect training; off by default")
    ap.add_argument("--rand_bench", type=int, default=0, help="also build this many randomised benchmarks rnd0, rnd1 and so on (degrade_random.draw_bench, held-out probabilities saved to loso_oof_proba_eval_rnd{k}.npz); 0 means none")
    ap.add_argument("--rand_main", action="store_true", help="use the randomised benchmark rnd0 as the main benchmark (loso_oof_proba.npz and the cost search), requires --rand_bench")
    ap.add_argument("--check_args", action="store_true", help="only check that the arguments parse, print them and exit")
    ap.add_argument("--no_bar", action="store_true", help="stop after leave-one-match-out, without training the full model or reading the competition data (for runs on public data only)")
    ap.add_argument("--eval_realistic", action="store_true", help="train with the degradation given by the arguments and evaluate the held-out matches degraded with the fixed realistic degradation (realistic ball gaps, noise 0.01), so that different training degradations are compared on the same benchmark")
    ap.add_argument("--train_only", default="", help="sources used for training only, without a held-out fold (game name prefixes, comma-separated), e.g. skillcorner")
    ap.add_argument("--sc_matches", type=int, default=20, help="number of SkillCorner matches to use, the first ones sorted by match id")
    ap.add_argument("--two_stage", action="store_true", help="two-stage gradient boosted tree model, first dead ball against live ball, then home against away on live-ball frames, the probabilities multiplied into three columns (mtah_v2.TwoStageHGB; off by default)")
    ap.add_argument("--hgb", default="", help="hyperparameters of the per-frame model, e.g. max_iter=600,learning_rate=0.04,max_leaf_nodes=63,l2_regularization=1.0,min_samples_leaf=20 (default mtah_v2.HGB_DEFAULT)")
    ap.add_argument("--eval_static_wander", type=float, default=-1.0, help="degrade the held-out matches with this slow-rolling amplitude instead (-1 means same as training; with --eval_realistic it overrides its 0). Training and evaluation then differ, which measures how much it matters whether the training bench resembles the target")
    ap.add_argument("--eval_ball_smooth", type=int, default=-1, help="degrade the held-out matches with this ball-track smoothing window instead (-1 means same as training)")
    ap.add_argument("--extra_eval", default="", help="also degrade the held-out matches under each of these conditions and evaluate, e.g. 0:0,3:15 (wander:smooth, comma-separated); each fold trains one model and predicts several benchmarks; the held-out probabilities of each condition are saved to loso_oof_proba_eval_sw{sw}_bs{bs}.npz and the scores recorded as xeval_sw{sw}_bs{bs}_* (the whole cross-condition matrix in one run)")
    ap.add_argument("--ball_smooth", type=int, default=0, help="window (frames) of the moving average applied to the IDSSE ball track in the fixed degradation (smooth_ball_track), 0 means off")
    ap.add_argument("--static_wander", type=float, default=0.0, help="median speed (m/s) at which a still ball rolls slowly between nearby players in the fixed degradation (static_ball_wander), 0 means off")
    ap.add_argument("--impute_sigma", type=float, default=0.0, help="optional, removed ball positions are filled with the true position plus a slow drift error of this many metres (only for the fixed IDSSE and Metrica degradations), 0 means the ball is set missing as usual")
    ap.add_argument("--metrica_jitter", type=float, default=0.0, help="standard deviation (metres) of the AR(1) offset added to slow Metrica ball positions that stick to a player's feet, 0 means off")
    ap.add_argument("--train_stride", type=int, default=1, help="keep one frame in this many for training-only sources (SkillCorner is interpolated from 10 fps, so adjacent frames are highly redundant)")
    ap.add_argument("--augment", default="", help="training-set augmentation, comma-separated list, swap exchanges home and away, mirror mirrors along the long axis, both does both; the copies are used for training only, held-out folds still predict the original match")
    a = ap.parse_args()
    if a.rand_main and a.rand_bench < 1:
        ap.error("--rand_main needs --rand_bench")
    if a.rand_degrade and a.train_extra_degrade:
        ap.error("--rand_degrade cannot be combined with --train_extra_degrade")
    if a.check_args:
        print("argument check passed", a.name, {k: v for k, v in vars(a).items() if k not in ("name", "check_args")})
        return
    hgb = {k: (int(v) if v.lstrip("-").isdigit() else float(v)) for k, v in (kv.split("=") for kv in a.hgb.split(",") if kv)}
    bhgb = {k: (int(v) if v.lstrip("-").isdigit() else float(v)) for k, v in (kv.split("=") for kv in a.bhgb.split(",") if kv)}
    out_dir = os.path.join(os.path.expanduser(os.environ.get("MTAH_RUNS", "~/Data/more_than_a_hack_2027/runs")), a.name)   # the environment variable MTAH_RUNS can point to another output directory
    os.makedirs(out_dir, exist_ok=True)
    sets = build_sources(a.sources.split(","), a.seeds, a.noise, a.alive_missing, a.idsse_min_spell, a.realistic, a.sc_matches, a.metrica_jitter, a.impute_sigma, a.static_wander, a.ball_smooth, augment=a.augment, rand_mode="train" if a.rand_degrade else "")
    # --train_extra_degrade, copies made with other fixed degradations also go into training and, like the augmented copies, are held out together with the original match (excluded by game name)
    for k, p in enumerate([p for p in a.train_extra_degrade.split(",") if p]):
        xsw, xbs = float(p.split(":")[0]), int(p.split(":")[1])
        more = build_sources(a.sources.split(","), a.seeds, a.noise, a.alive_missing, a.idsse_min_spell, a.realistic, a.sc_matches, a.metrica_jitter, a.impute_sigma, xsw, xbs, augment=a.augment)
        for st in more:
            st["seed"] = st["seed"] + 10000 * (k + 1)
        sets += more
        print("[progress] training adds degraded copies sw%g bs%d, %d in total" % (xsw, xbs, len(more)), flush=True)
    train_only = [p for p in a.train_only.split(",") if p]
    names = None
    ms_src = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.min_spell_src.split(",") if kv}
    for st in sets:
        ms = a.min_spell_all
        for pre, v in ms_src.items():
            if st["game"].startswith(pre):
                ms = v
        if ms > 0:
            st["gt"] = merge_short_spells(st["gt"], st["M"]["blocks"], ms)
        st["w"] = spell_weights(st["gt"], st["M"]["blocks"], a.spell_weight) if a.spell_weight > 0 else np.ones(len(st["gt"]))
        if a.vis_speed_weight:
            st["w"] = st["w"] * vis_speed_weights(st["M"], [float(x) for x in a.vis_speed_weight.split(",")])
        if a.trans_weight > 0:                           # weights within --trans_win frames of a change point are multiplied by 1 plus --trans_weight (same as the argument of the same name in seq_model)
            g_ = st["gt"]; near = np.zeros(len(g_), bool)
            for b0, b1 in st["M"]["blocks"].values():
                seg = g_[b0:b1 + 1]
                for c in np.flatnonzero((seg[1:] != seg[:-1]) & (seg[1:] > 0) & (seg[:-1] > 0)) + 1:
                    near[b0 + max(0, c - a.trans_win):b0 + min(len(seg), c + a.trans_win + 1)] = True
            st["w"] = st["w"] * np.where(near, 1.0 + a.trans_weight, 1.0)
        X, names = build_features(st["M"], gap_feats=a.gap_feats, touch_feats=a.touch_feats, touch_set=a.touch_set, kin_feats=a.kin_feats, tta_feats=a.tta_feats, setp_feats=a.setp_feats, shape_feats=a.shape_feats, dir_feats=a.dir_feats, comove_feats=a.comove_feats)
        st["X"] = X
        st["y"] = (st["gt"] - 1).astype(int)        # 0 H, 1 A, 2 D, -1 none
        st["yb"] = boundary_labels(st["gt"], st["M"]["blocks"], a.bhalf, a.boundary_kind) if a.boundary else None
        st["train_only"] = any(st["game"].startswith(p) for p in train_only)
        if st["train_only"]:
            keep = np.flatnonzero(st["y"] >= 0)[::max(1, a.train_stride)]   # training-only sources need no prediction, so only the labelled frames are kept, which saves memory
            st["X"], st["y"], st["w"] = st["X"][keep], st["y"][keep], st["w"][keep]
            st["n_train_rows"] = int(len(keep))
    cross_eval = a.eval_static_wander >= 0 or a.eval_ball_smooth >= 0
    if a.eval_realistic or cross_eval:
        # the held-out matches are evaluated on the fixed realistic degradation; the seed 0 version of every match is loaded again, degraded that way, and its features are built with the same switches;
        # with cross_eval the slow rolling and the smoothing of the held-out matches may differ from training (values not given fall back to the training ones)
        ev_sw = a.eval_static_wander if a.eval_static_wander >= 0 else a.static_wander
        ev_bs = a.eval_ball_smooth if a.eval_ball_smooth >= 0 else a.ball_smooth
        ev = build_sources([n for n in a.sources.split(",")], 1, 0.01, a.alive_missing, a.idsse_min_spell, True, a.sc_matches, a.metrica_jitter, a.impute_sigma, ev_sw, ev_bs)
        evm = {e["game"]: e for e in ev}
        for st in sets:
            if st["seed"] == 0 and not st.get("train_only") and st["game"] in evm:
                st["X_eval"], _ = build_features(evm[st["game"]]["M"], gap_feats=a.gap_feats, touch_feats=a.touch_feats, touch_set=a.touch_set, kin_feats=a.kin_feats, tta_feats=a.tta_feats, setp_feats=a.setp_feats, shape_feats=a.shape_feats, dir_feats=a.dir_feats, comove_feats=a.comove_feats)
                st["M_eval"] = evm[st["game"]]["M"]
        del ev, evm
    extra_eval = [(float(p.split(":")[0]), int(p.split(":")[1])) for p in a.extra_eval.split(",") if p]
    for xsw, xbs in extra_eval:
        # extra benchmark conditions, the same matches, seed 0, fixed degradation (realistic ball gaps, noise 0.01) with slow rolling xsw and smoothing xbs, used only to predict the held-out folds
        tag = "sw%g_bs%d" % (xsw, xbs)
        ev = build_sources([n for n in a.sources.split(",")], 1, 0.01, a.alive_missing, a.idsse_min_spell, True, a.sc_matches, a.metrica_jitter, a.impute_sigma, xsw, xbs)
        evm = {e["game"]: e for e in ev}
        for st in sets:
            if st["seed"] == 0 and not st.get("train_only") and st["game"] in evm:
                Xx, _ = build_features(evm[st["game"]]["M"], gap_feats=a.gap_feats, touch_feats=a.touch_feats, touch_set=a.touch_set, kin_feats=a.kin_feats, tta_feats=a.tta_feats, setp_feats=a.setp_feats, shape_feats=a.shape_feats, dir_feats=a.dir_feats, comove_feats=a.comove_feats)
                st.setdefault("X_eval_extra", {})[tag] = Xx
        del ev, evm
        print("[progress] extra benchmark %s built" % tag, flush=True)
    for k_r in range(a.rand_bench):
        # randomised benchmarks, the same matches degraded with parameter set k of degrade_random.draw_bench, used only to predict the held-out folds; with --rand_main rnd0 is also the main benchmark
        tag = "rnd%d" % k_r
        ev = build_sources([n for n in a.sources.split(",")], 1, 0.01, a.alive_missing, a.idsse_min_spell, True, a.sc_matches, a.metrica_jitter, a.impute_sigma, 0.0, 0, rand_mode=tag)
        evm = {e["game"]: e for e in ev}
        for st in sets:
            if st["seed"] == 0 and not st.get("train_only") and st["game"] in evm:
                Xx, _ = build_features(evm[st["game"]]["M"], gap_feats=a.gap_feats, touch_feats=a.touch_feats, touch_set=a.touch_set, kin_feats=a.kin_feats, tta_feats=a.tta_feats, setp_feats=a.setp_feats, shape_feats=a.shape_feats, dir_feats=a.dir_feats, comove_feats=a.comove_feats)
                st.setdefault("X_eval_extra", {})[tag] = Xx
                if k_r == 0 and a.rand_main:
                    st["X_eval"] = Xx
        del ev, evm
        print("[progress] randomised benchmark %s built%s" % (tag, " (also the main benchmark)" if k_r == 0 and a.rand_main else ""), flush=True)
    kgrid = [float(x) for x in a.kgrid.split(",")] if a.boundary else [0.0]
    p0 = float(np.mean(np.concatenate([s["yb"][s["yb"] >= 0] for s in sets]))) if a.boundary else 0.04
    if a.boundary:
        rep("boundary_base_rate", p0)
    games = sorted({st["game"] for st in sets if not st.get("train_only")})
    rep("train_only_games", sorted({st["game"] for st in sets if st.get("train_only")}))
    rep("train_only_rows", int(sum(st.get("n_train_rows", 0) for st in sets)))
    rep("n_sets", len(sets))
    rep("n_features", len(names))
    rep("build_s", time.time() - T0)
    global BUILD_T
    BUILD_T = time.time() - T0
    print("[progress] features built, starting leave-one-match-out with %d folds; one line is printed after each fold, follow the log with tail -f" % len(games), flush=True)
    for g in games:
        st = [s for s in sets if s["game"] == g and s["seed"] == 0][0]
        gr = st["M"]["graded"]
        rep("src_%s_graded_frames" % g, int(gr.sum()))
        rep("src_%s_ball_visible_graded" % g, float(st["M"]["ball_visible"][gr].mean()))
        rep("src_%s_D_frac" % g, float((st["gt"][gr] == 3).mean()))
        rep("src_%s_D_given_missing" % g, float((st["gt"][gr & ~st["M"]["ball_visible"]] == 3).mean()))

    costs = None
    k_best = 0.0
    loso_sel = [x for x in a.loso_games.split(",") if x]
    if loso_sel:                                           # --loso_games, only these folds are run, and the cost search and the scoring below use only these matches
        assert a.no_bar and not a.skip_loso, "--loso_games requires --no_bar and cannot be combined with --skip_loso"
        assert set(loso_sel) <= set(games), "--loso_games contains matches that are not in the sources %s" % sorted(set(loso_sel) - set(games))
        games = loso_sel
    if not a.skip_loso:
        oof, oof_b, oof_x, oof_bx = {}, {}, {}, {}
        for n, g in enumerate(games):
            Xtr = np.concatenate([s["X"][s["y"] >= 0] for s in sets if s["game"] != g])
            ytr = np.concatenate([s["y"][s["y"] >= 0] for s in sets if s["game"] != g])
            wtr = np.concatenate([s["w"][s["y"] >= 0] for s in sets if s["game"] != g])
            clf = (fit_two_stage if a.two_stage else fit)(Xtr, ytr, weight=wtr, params=hgb)
            held = [s for s in sets if s["game"] == g and s["seed"] == 0][0]
            oof[g] = clf.predict_proba(held.get("X_eval", held["X"]))
            for tag, Xx in held.get("X_eval_extra", {}).items():
                oof_x.setdefault(tag, {})[g] = clf.predict_proba(Xx)
            if a.boundary:
                Xb = np.concatenate([s["X"][s["yb"] >= 0] for s in sets if s["game"] != g])
                yb = np.concatenate([s["yb"][s["yb"] >= 0] for s in sets if s["game"] != g])
                clf_bf = fit(Xb, yb, params=bhgb or None)
                oof_b[g] = clf_bf.predict_proba(held["X"])[:, 1]
                for tag, Xx in held.get("X_eval_extra", {}).items():      # change-point probabilities on the other benchmarks, for the reported benchmarks of the segment decoding
                    oof_bx.setdefault(tag, {})[g] = clf_bf.predict_proba(Xx)[:, 1]
            # the finished folds are written to disk after every fold, so that they are not lost if the process dies midway; the file name carries partial to keep it apart from the final complete file
            np.savez_compressed(os.path.join(out_dir, "loso_oof_partial.npz"), **{k_: v.astype(np.float32) for k_, v in oof.items()})
            for tag, od in oof_x.items():
                np.savez_compressed(os.path.join(out_dir, "loso_oof_partial_eval_%s.npz" % tag), **{k_: v.astype(np.float32) for k_, v in od.items()})
            el = time.time() - T0
            per = (el - BUILD_T) / (n + 1)
            print("[progress] held-out fold %d/%d done (%s)  elapsed %.0fs  estimated remaining %.0fs (%d held-out folds + full training about %.0fs)" % (
                n + 1, len(games), g, el, per * (len(games) - n - 1) + per, len(games) - n - 1, per), flush=True)
        # the held-out probabilities are saved, so that later ensemble and post-processing experiments need no retraining
        np.savez_compressed(os.path.join(out_dir, "loso_oof_proba.npz"), **{g: oof[g].astype(np.float32) for g in games})
        if a.boundary:
            np.savez_compressed(os.path.join(out_dir, "loso_oof_boundary.npz"), **{g: oof_b[g].astype(np.float32) for g in games})
            for tag, od in oof_bx.items():
                np.savez_compressed(os.path.join(out_dir, "loso_oof_boundary_eval_%s.npz" % tag), **{g: od[g].astype(np.float32) for g in games})
        grid = [tuple(float(x) for x in a.costs.split(","))] if a.costs else list(itertools.product(*[[float(x) for x in p.split(",")] for p in a.grid.split(";")]))

        def cost_rows(oofd, grid_):
            """For one set of held-out probabilities, scores the Viterbi decoding of every match on the cost grid and returns the rows sorted by decreasing mean spellF1"""
            rows_ = []
            for (ct, cd, ca), k in itertools.product(grid_, kgrid):
                per = {}
                for g in games:
                    held = [s for s in sets if s["game"] == g and s["seed"] == 0][0]
                    gr = held["M"]["graded"]
                    pred = to_hadx(viterbi3_tv(np.log(np.clip(oofd[g], 1e-6, 1)), held["M"]["blocks"], ct, cd, ca, oof_b.get(g), k, p0))
                    pred[~gr] = 0
                    r = evaluate(codes_from_hadx(held["gt"]), codes_from_hadx(pred))
                    per[g] = (r["spell_f1"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"])
                rows_.append({"costs": (ct, cd, ca), "k": k, "mean_spell_f1": float(np.mean([v[0] for v in per.values()])),
                              "mean_macro_f1": float(np.mean([v[1] for v in per.values()])), "per_game": per})
            rows_.sort(key=lambda r: -r["mean_spell_f1"])
            return rows_

        rows = cost_rows(oof, grid)
        # extra benchmarks, probabilities saved, costs searched for each, and the score at the best costs of the training benchmark recorded as well (only one set of costs exists at deployment)
        OUT["xeval"] = {}
        for tag, oofd in oof_x.items():
            np.savez_compressed(os.path.join(out_dir, "loso_oof_proba_eval_%s.npz" % tag), **{g: oofd[g].astype(np.float32) for g in games})
            rx = cost_rows(oofd, grid)
            at_primary = [r for r in rx if r["costs"] == rows[0]["costs"] and r["k"] == rows[0]["k"]][0]
            rep("xeval_%s_best_mean_spell_f1" % tag, rx[0]["mean_spell_f1"])
            rep("xeval_%s_best_costs" % tag, "%g,%g,%g" % rx[0]["costs"])
            rep("xeval_%s_at_primary_costs_mean_spell_f1" % tag, at_primary["mean_spell_f1"])
            for g in games:
                rep("xeval_%s_best_spell_f1_%s" % (tag, g), rx[0]["per_game"][g][0])
            OUT["xeval"][tag] = {"grid": [{"costs": r["costs"], "k": r["k"], "mean_spell_f1": r["mean_spell_f1"], "mean_macro_f1": r["mean_macro_f1"]} for r in rx]}
        print("%-14s %4s %9s %9s  " % ("costs", "k", "meanSpell", "meanMacro") + "  ".join("%12s" % g for g in games))
        for row in rows[:10]:
            print("%-14s %4g %9.4f %9.4f  " % ("%g,%g,%g" % row["costs"], row["k"], row["mean_spell_f1"], row["mean_macro_f1"]) + "  ".join("%6.3f/%3d" % (row["per_game"][g][0], row["per_game"][g][2]) for g in games))
        if a.boundary:
            for k in kgrid:
                bk = max((r for r in rows if r["k"] == k), key=lambda r: r["mean_spell_f1"])
                rep("loso_best_mean_spell_f1_k%g" % k, bk["mean_spell_f1"])
        best = rows[0]
        costs, k_best = best["costs"], best["k"]
        rep("loso_best_mean_spell_f1", best["mean_spell_f1"])
        rep("loso_best_mean_macro_f1", best["mean_macro_f1"])
        for g in games:
            rep("loso_best_spell_f1_%s" % g, best["per_game"][g][0])
            rep("loso_gt_spells_%s" % g, best["per_game"][g][3])
        # at the best setting, for every held-out match, the class F1, the spellF1 by IoU threshold and the hit rate by spell-length bin (for the report)
        detail = {}
        for g in games:
            held = [s for s in sets if s["game"] == g and s["seed"] == 0][0]
            gr = held["M"]["graded"]
            pred = to_hadx(viterbi3_tv(np.log(np.clip(oof[g], 1e-6, 1)), held["M"]["blocks"], *costs, pb=oof_b.get(g), k=k_best, p0=p0))
            pred[~gr] = 0
            G, Pc = codes_from_hadx(held["gt"]), codes_from_hadx(pred)
            r = evaluate(G, Pc)
            detail[g] = {"spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"], "f1_H": r["f1_H"], "f1_A": r["f1_A"], "f1_D": r["f1_D"],
                         "spell_f1_per_iou": {str(k_): v for k_, v in r["spell_f1_per_iou"].items()},
                         "n_gt_spells": r["n_gt_spells"], "n_pred_spells": r["n_pred_spells"],
                         "hit_by_length": {k_: v for k_, v in binned_hit_rates(G, Pc).items()}}
            rep("loso_best_f1_D_%s" % g, r["f1_D"])
        OUT["loso_detail"] = detail
        hb = {k_: float(np.nanmean([detail[g]["hit_by_length"][k_][0] for g in games])) for k_ in next(iter(detail.values()))["hit_by_length"]}
        for k_, v in hb.items():
            rep("loso_hit_iou05_len%s" % k_, v)
        OUT["loso_grid"] = [{"costs": r["costs"], "k": r["k"], "mean_spell_f1": r["mean_spell_f1"], "mean_macro_f1": r["mean_macro_f1"]} for r in rows]
    if a.costs:
        costs = tuple(float(x) for x in a.costs.split(","))
    rep("costs_used", "%g,%g,%g" % costs)
    rep("k_used", k_best)

    meta = {"feature_names": names, "costs": costs, "sources": a.sources, "seeds": a.seeds, "noise": a.noise,
            "alive_missing": a.alive_missing, "k": k_best, "p0": p0, "bhalf": a.bhalf,
            "gap_feats": a.gap_feats, "touch_feats": a.touch_feats, "kin_feats": a.kin_feats, "tta_feats": a.tta_feats, "setp_feats": a.setp_feats, "shape_feats": a.shape_feats, "dir_feats": a.dir_feats, "comove_feats": a.comove_feats, "train_extra_degrade": a.train_extra_degrade, "touch_set": a.touch_set, "spell_weight": a.spell_weight, "trans_weight": a.trans_weight, "trans_win": a.trans_win, "min_spell_all": a.min_spell_all, "min_spell_src": a.min_spell_src, "realistic": a.realistic, "train_only": a.train_only, "sc_matches": a.sc_matches, "train_stride": a.train_stride, "metrica_jitter": a.metrica_jitter, "eval_realistic": a.eval_realistic, "idsse_min_spell": a.idsse_min_spell, "impute_sigma": a.impute_sigma, "hgb": hgb, "static_wander": a.static_wander, "ball_smooth": a.ball_smooth, "eval_static_wander": a.eval_static_wander, "eval_ball_smooth": a.eval_ball_smooth, "extra_eval": a.extra_eval}
    if a.rand_degrade or a.rand_bench:                  # recorded only when used
        meta.update({"rand_degrade": bool(a.rand_degrade), "rand_bench": int(a.rand_bench), "rand_main": bool(a.rand_main)})
    import sklearn
    versions = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "pandas": __import__("pandas").__version__}
    if a.no_bar:
        joblib.dump({**meta, "model": None, "boundary_model": None}, os.path.join(out_dir, "model.joblib"))
        json.dump(versions, open(os.path.join(out_dir, "versions.json"), "w"), indent=1)
        rep("runtime_s", time.time() - T0)
        json.dump(OUT, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1, ensure_ascii=False, default=str)
        print("written to " + out_dir + " (--no_bar, leave-one-match-out results only, no full model and no BAR-ATH prediction)")
        return
    Xtr = np.concatenate([s["X"][s["y"] >= 0] for s in sets])
    ytr = np.concatenate([s["y"][s["y"] >= 0] for s in sets])
    wtr = np.concatenate([s["w"][s["y"] >= 0] for s in sets])
    clf = (fit_two_stage if a.two_stage else fit)(Xtr, ytr, weight=wtr, params=hgb)
    clf_b = None
    if a.boundary and (k_best > 0 or a.boundary_keep):
        Xb = np.concatenate([s["X"][s["yb"] >= 0] for s in sets])
        yb = np.concatenate([s["yb"][s["yb"] >= 0] for s in sets])
        clf_b = fit(Xb, yb, params=bhgb or None)
    joblib.dump({**meta, "model": clf, "boundary_model": clf_b}, os.path.join(out_dir, "model.joblib"))
    json.dump(versions, open(os.path.join(out_dir, "versions.json"), "w"), indent=1)
    Mc = load({"data_dir": "~/Data/more_than_a_hack_2027/data", "official_repo": "~/Data/more_than_a_hack_2027/official_repo"})
    Mc["blocks"] = {1: (11185, 90651), 2: (110808, 190197)}
    Xc, names_c = build_features(Mc, gap_feats=a.gap_feats, touch_feats=a.touch_feats, touch_set=a.touch_set, kin_feats=a.kin_feats, tta_feats=a.tta_feats, setp_feats=a.setp_feats, shape_feats=a.shape_feats, dir_feats=a.dir_feats, comove_feats=a.comove_feats)
    assert names_c == names
    p = clf.predict_proba(Xc)
    g = Mc["graded"]
    pb = clf_b.predict_proba(Xc)[:, 1] if clf_b is not None else None
    pred = to_hadx(viterbi3_tv(np.log(np.clip(p, 1e-6, 1)), Mc["blocks"], *costs, pb=pb, k=k_best, p0=p0))
    pred[~g] = 0
    for k, name in ((1, "H"), (2, "A"), (3, "D")):
        rep("pred_frac_%s_graded" % name, float((pred[g] == k).mean()))
    labels = np.array(["X", "H", "A", "D"])[pred]
    reason = validate(np.arange(N_FRAMES), labels, N_FRAMES)
    rep("format_ok", reason is None)
    path = os.path.join(out_dir, a.name + ".csv")
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, l) for i, l in enumerate(labels))
    pc = codes_from_hadx(pred)
    ps = spells(pc, valid_runs(codes_from_hadx(np.where(g, 1, 0))))
    rep("pred_possessions_full", len(ps))
    rep("pred_possession_mean_s_full", float(np.mean([b - a_ for a_, b, _ in ps]) / 25.0) if ps else 0.0)
    gt = Mc.get("public_gt")
    if gt is not None:
        r = evaluate(codes_from_hadx(gt), pc)
        print("--- public 5-minute window, pipeline check only ---")
        for k in ("spell_f1", "macro_f1", "f1_H", "f1_A", "f1_D", "n_gt_spells", "n_pred_spells", "gt_mean_spell_s", "pred_mean_spell_s"):
            rep("public_" + k, r[k])
    np.save(os.path.join(out_dir, "proba_bar_ath.npy"), p.astype(np.float32))
    if a.boundary_keep:                                    # full model of the change-point head and its BAR-ATH probabilities
        assert pb is not None
        np.save(os.path.join(out_dir, "proba_boundary_bar_ath.npy"), pb.astype(np.float32))
        joblib.dump({"boundary_model": clf_b, "feature_names": list(names), "bhalf": a.bhalf, "bhgb": bhgb}, os.path.join(out_dir, "boundary_model.joblib"))
    rep("runtime_s", time.time() - T0)
    json.dump(OUT, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print("written to " + path)


if __name__ == "__main__":
    main()
