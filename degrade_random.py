"""
degrade_random, randomised degradation of the public training matches; every training copy and every randomised benchmark draws its degradation parameters from ranges set on public data only
============================
So that no parameter of the training degradation comes from the competition data, every copy draws its parameters from ranges set on public data only. The three fixed benches of --extra_eval
(mtah_v3.degrade_like_wc, augment_metrica) use one fixed set of parameters chosen from label-free statistics of the competition tracking. Here the ranges rest on three kinds of evidence only,
the public data themselves (SkillCorner open data, IDSSE, Metrica sample data; summary statistics measured by external_stats.py), physical common sense, and the literature. No statistic of the competition data is used.
Where public sources measure a parameter, its range goes from the cleanest public source to twice the worst one, or spans the measured values where twice would be impossible, or is capped by a physical limit
(drop_frac); parameters that no public source can measure get a generic wide range from physical common sense. The basis of each parameter is written next to RANGES.
  Training copies  draw_train(game, copy), every parameter drawn independently and uniformly (log-uniformly for those marked log), seeded by game name and copy number, reproducible.
  Benchmarks       draw_bench(game, k), the three randomised benchmarks k = 0, 1, 2; each parameter takes the 1/6, 1/2 and 5/6 quantiles of its range across the three benchmarks,
                   in a random order fixed by the game name (Latin hypercube), so the three benchmarks together cover both ends and the middle of the range.
The steps of the degradation itself (degrade) are those of the fixed degradation of mtah_v3, with the drawn parameters.

Inputs  (M, gt) of a public source (the structures returned by adapters/idsse and adapters/metrica_sample)
Outputs  the degraded M (player_xy, ball_xy and ball_visible replaced, everything else shared)

Usage
  from degrade_random import draw_train, draw_bench, degrade
  M2 = degrade(M, gt, "idsse_J03WMX", draw_train("idsse_J03WMX", 0), seed=0, source="idsse")
  python -u degrade_random.py --check 24   # measures label-free statistics of the drawn degradations on public data, to compare with the external sources

Last modified 2026-11-15
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Each parameter is (distribution, lower bound, upper bound). uni is uniform, log is log-uniform, int is a uniform integer (both ends included).
# Numbers from external sources (measured by external_stats.py) are written as median (range across matches).
RANGES = {
    # Occlusion ball gaps, simulated with a hazard rate (possession.degrade.realistic_missing; the ball disappears more easily when it is slow and close to a player and reappears once it moves fast). This parameter is the target share of graded frames with the ball missing;
    # for IDSSE over all graded frames, for Metrica over live-ball frames (Metrica does not give the ball during dead balls anyway). Share of live-ball frames with the ball missing in the external sources, IDSSE 0, SkillCorner 0.053 (0.039 to 0.084),
    # Metrica sample data 0.007 (at most 0.012 in one match) after the dead-ball rules of adapters/metrica_sample, 0.25 in one match before them, and twice that, 0.5, sets the upper bound. Lower bound 0 (IDSSE)
    "occ_frac": ("uni", 0.0, 0.5),
    # Share of extra missing ball on dead-ball frames (IDSSE only, done by hiding one run of consecutive frames inside each dead-ball segment). Share of dead-ball frames with the ball missing in the external sources, IDSSE 0, SkillCorner 0.56 (0.44 to 0.60),
    # Metrica sample data 0.94 (0.90 to 0.99); the range spans the two extremes
    "dead_frac": ("uni", 0.0, 0.94),
    # Shape of the hazard rate. No external source measures it directly, so wide ranges are set from physical common sense, and --check verifies that the simulated gap statistics cover the external sources (the upper bounds of fast_w, p_fast and p_slow
    # and min_gap were set so that the simulated gap p90 and the share of fast balls at the first frame of a gap reach the SkillCorner values). SkillCorner has a median gap of 3.6 s (2.5 to 4.5),
    # p90 8.7 s (7.1 to 9.7), fast ball at the first frame of a gap 0.35, at the last frame 0.47; the Metrica sample data have a median gap of 8.7 s and p90 39 s.
    # d_scale is the scale at which the disappearance hazard decays with the distance to the nearest player; a human body is about half a metre wide, so 0.35 to 1.4 m
    "d_scale": ("log", 0.35, 1.4),
    # fast_w is the multiplier of the disappearance hazard for a fast ball (3-frame speed above 5 m/s, a short pass or faster). A fast ball is less often hidden by players but blurs more easily, so both directions are possible, 0.05 to 2
    "fast_w": ("log", 0.05, 2.0),
    # p_fast and p_slow are the per-frame probabilities of reappearing while missing (fast ball, slow ball), so a fast ball reappears after 0.2 to 4 s on average and a slow ball after 4 to 200 s (25 frames per second)
    "p_fast": ("log", 0.01, 0.2),
    "p_slow": ("log", 2e-4, 1e-2),
    # Minimum gap length in frames, 0 to 2 s
    "min_gap": ("int", 0, 50),
    # Player dropout, 1 to 3 slots dropped per segment. This parameter is the share of graded frames with fewer than 20 players on the pitch. IDSSE and Metrica 0; SkillCorner detects a median of 14 players per frame and
    # 0.96 of its frames have fewer than 20 players (the broadcast image covers only about 28% of the pitch). Twice that is impossible and the range does not reach it; the upper bound 0.6 is a physical cap that leaves at least four frames in ten complete
    "drop_frac": ("uni", 0.0, 0.6),
    # Median length of a dropout segment (frames). The median undetected-player segment in SkillCorner lasts 3.6 s (3.3 to 4.1), that is 90 frames; the range goes from half to twice that. The sigma of the log-normal segment length comes from
    # the ratio of the SkillCorner p90 to its median (14.8 / 3.6, ln 4.1 / 1.2816 is about 1.1), see DROP_SIGMA
    "drop_med": ("log", 45.0, 180.0),
    # Stationary standard deviation (metres) of the position noise (AR(1), coefficient 0.9, added to ball and players). Linke, Link and Lames 2020 (PLOS ONE 15(3) e0230179) measured a position RMSE of 0.08 to 0.09 m
    # for TRACAB optical tracking; the public data already carry an error of that size, so the added noise goes from a negligible 0.01 m to about twice that size, 0.2 m
    "noise": ("log", 0.01, 0.2),
    # Slow rolling of a still ball (mtah_v3.static_ball_wander turns a stationary ball into one that rolls slowly between nearby players), median rolling speed (m/s). Median 1 s displacement of the ball on dead-ball frames,
    # IDSSE 0 m (97% static), SkillCorner (automatic video tracking) 1.6 m (0.9 to 2.1, 28% static). The rolling has pauses, so the displacement is smaller than the speed; the speed ranges from 1 to 6,
    # and --check verifies that the simulated 1 s displacement on dead-ball frames goes from the IDSSE value of 0 to about twice the upper end of SkillCorner
    "wander": ("uni", 1.0, 6.0),
    # Share of still segments that stay still; 1 is fully static as in IDSSE, 0 means all of them roll slowly
    "wander_keep": ("uni", 0.0, 1.0),
    # Window (frames) of the moving average applied to the IDSSE ball track over each run of consecutive visible frames, 1 means no smoothing. The p90 of the second difference of the raw IDSSE ball is 26.5 m/s^2 (25.8 to 53.8),
    # and 13.4 after a window of 25, still above the 6.5 of the Metrica sample data; the range lies between the two public sources, 1 to 25 frames (1 s)
    "smooth": ("int", 1, 25),
}
PARAM_ORDER = tuple(RANGES)          # the sampling order is fixed; new parameters may only be appended
DROP_SIGMA = 1.1
BENCH_Q = (1 / 6, 1 / 2, 5 / 6)
BENCH_SEED0 = 500                    # the degradation seed of benchmark k is 500 + k, kept apart from the training copies (0, 1, 2 and so on)


def _inv(name, u):
    kind, lo, hi = RANGES[name]
    if kind == "uni":
        return float(lo + u * (hi - lo))
    if kind == "log":
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    return int(min(hi, lo + int(np.floor(u * (hi - lo + 1)))))


def _key(game):
    return zlib.crc32(game.encode("utf-8"))


def draw_train(game, copy):
    """Parameters of a training copy, seeded by game name and copy number, each parameter drawn independently"""
    rng = np.random.default_rng([_key(game), int(copy), 125])
    return {k: _inv(k, rng.random()) for k in PARAM_ORDER}


def draw_bench(game, k):
    """Parameters of randomised benchmark k (0, 1, 2). Each parameter takes the 1/6, 1/2 and 5/6 quantiles across the three benchmarks, in an order fixed by the game name"""
    rng = np.random.default_rng([_key(game), 9125])
    out = {}
    for name in PARAM_ORDER:
        perm = rng.permutation(len(BENCH_Q))
        out[name] = _inv(name, BENCH_Q[perm[k]])
    return out


def params_for(game, mode):
    """mode looks like train:3 (training copy 3) or rnd1 (benchmark 1). Returns (parameters, degradation seed)"""
    if mode.startswith("train:"):
        c = int(mode.split(":")[1])
        return draw_train(game, c), c
    assert mode.startswith("rnd"), mode
    k = int(mode[3:])
    return draw_bench(game, k), BENCH_SEED0 + k


def player_dropout_r(rng, P, graded, frac, med, sigma=DROP_SIGMA):
    """Player dropout in segments, same approach as possession.degrade.player_dropout, with adjustable share, median segment length and segment-length sigma"""
    P = np.asarray(P, np.float32).copy()
    if frac <= 0:
        return P
    T, S, _ = P.shape
    n = (~np.isnan(P[:, :, 0])).sum(1)
    guard = 0
    while ((n < 20) & graded).sum() < frac * graded.sum() and guard < 100000:
        guard += 1
        st = int(rng.integers(0, T))
        if not graded[st]:
            continue
        length = max(5, int(np.exp(np.log(med) + sigma * rng.standard_normal())))
        for slot in rng.choice(S, size=int(rng.integers(1, 4)), replace=False):
            P[st:st + length, slot] = np.nan
        n[st:st + length] = (~np.isnan(P[st:st + length, :, 0])).sum(1)
    return P


def dead_missing(rng, gt, graded, frac):
    """Hides one run of consecutive frames inside each dead-ball segment (gt 3). Its length is a share U(0, 2·frac) of the segment, capped at 1, and its position is uniform; the expected share is about frac (lower when frac is above 0.5)"""
    T = len(gt)
    miss = np.zeros(T, bool)
    if frac <= 0:
        return miss
    dead = (np.asarray(gt) == 3) & graded
    e = np.diff(np.r_[0, dead.astype(np.int8), 0])
    for a, z in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        n = z - a
        m = int(round(min(1.0, rng.uniform(0, 2 * frac)) * n))
        if m <= 0:
            continue
        s = a + int(rng.integers(0, n - m + 1))
        miss[s:s + m] = True
    return miss


def degrade(M, gt, game, prm, seed, source):
    """Degrades one public match with the parameters prm. source is idsse or metrica, and the steps are those of the fixed degradation of mtah_v3,
    smoothing (IDSSE only), slow rolling of a still ball, occlusion ball gaps (hazard rate), extra missing ball during dead balls (IDSSE only), player dropout, position noise"""
    from mtah_v3 import ar1_noise, smooth_ball_track, static_ball_wander
    from possession.degrade import realistic_missing
    rng = np.random.default_rng(seed)
    M2 = dict(M)
    graded = np.asarray(M["graded"], bool)
    fps = M["fps"]
    b = np.asarray(M["ball_xy"], np.float64).copy()
    if source == "idsse" and prm["smooth"] > 1:
        b = smooth_ball_track(b, prm["smooth"])
    if prm["wander_keep"] < 1.0:
        b = static_ball_wander(rng, b, fps, prm["wander"], P=M["player_xy"], keep_frac=prm["wander_keep"])
    P = np.asarray(M["player_xy"], np.float32)
    vis = ~np.isnan(b[:, 0])
    hz = dict(d_scale=prm["d_scale"], fast_w=prm["fast_w"], p_fast=prm["p_fast"], p_slow=prm["p_slow"], min_gap=prm["min_gap"])
    if source == "idsse":
        missing = realistic_missing(rng, b, P, fps, graded & vis, graded, prm["occ_frac"], **hz) if prm["occ_frac"] >= 0.01 else np.zeros(len(b), bool)
        missing = missing | dead_missing(rng, gt, graded, prm["dead_frac"])
    elif source == "metrica":
        alive = (np.asarray(gt) == 1) | (np.asarray(gt) == 2)
        missing = (realistic_missing(rng, b, P, fps, alive & vis, alive, prm["occ_frac"], **hz) if prm["occ_frac"] >= 0.01 else np.zeros(len(b), bool)) | ~vis
    else:
        raise ValueError("randomised degradation only supports idsse and metrica, got %s" % source)
    P = player_dropout_r(rng, P, graded, prm["drop_frac"], prm["drop_med"])
    P = P + ar1_noise(rng, P.shape, prm["noise"]).astype(np.float32)
    b = b + ar1_noise(rng, b.shape, prm["noise"])
    b[missing] = np.nan
    b[~graded] = np.nan
    M2["player_xy"], M2["ball_xy"], M2["ball_visible"] = P, b, ~np.isnan(b[:, 0])
    return M2


def load_degraded(game, mode, idsse_root, idsse_min_spell=0, metrica_root="~/Data/alt_training/metrica_sample"):
    """Loads a public match by game name and degrades it according to mode (train:c or rnd k). Returns (M2, gt). Used by spell_decode to rebuild the benchmark degradation"""
    prm, seed = params_for(game, mode)
    if game.startswith("idsse_"):
        from possession.adapters.idsse import load_match
        M, gt = load_match(idsse_root, game[len("idsse_"):], min_spell=idsse_min_spell)
        return degrade(M, gt, game, prm, seed, "idsse"), gt
    if game.startswith("metrica"):
        from possession.adapters.metrica_sample import load_game
        M, gt = load_game(metrica_root, int(game[len("metrica"):]))
        return degrade(M, gt, game, prm, seed, "metrica"), gt
    raise ValueError(game)


def check(n_draws, games):
    """Draws n_draws sets of training parameters and the three benchmarks, degrades a few public matches and measures label-free statistics (same definitions as external_stats) to see whether they cover the external sources"""
    from external_stats import dead_ball_motion, lt20
    from mtah_v3 import IDSSE_ROOT
    from possession.degrade import visibility_stats
    rows = []
    t0 = time.time()
    for g in games:
        for mode in ["rnd0", "rnd1", "rnd2"] + ["train:%d" % c for c in range(n_draws)]:
            M2, gt = load_degraded(g, mode, IDSSE_ROOT)
            r = visibility_stats(M2)
            r.update(dead_ball_motion(M2, np.asarray(gt), M2["fps"]))
            r["lt20_players"] = lt20(M2)
            g_ = M2["graded"]
            alive = g_ & ((gt == 1) | (gt == 2))
            r["vis_alive"] = float(M2["ball_visible"][alive].mean())
            r["vis_dead"] = float(M2["ball_visible"][g_ & (gt == 3)].mean())
            r.update({"game": g, "mode": mode})
            rows.append(r)
            print("[progress] %s %s visible %.3f live-ball visible %.3f median gap %.1f s p90 %.1f s  elapsed %.0fs" % (
                g, mode, r["vis"], r["vis_alive"], r.get("gap_med_s", np.nan), r.get("gap_p90_s", np.nan), time.time() - t0), flush=True)
    keys = ["vis", "vis_alive", "vis_dead", "dmin_med", "lt15", "gap_start_fast", "gap_end_fast", "gap_med_s", "gap_p90_s", "dead_disp1s_med", "dead_static_frac", "lt20_players"]
    summ = {}
    for src in ("idsse", "metrica"):
        rr = [r for r in rows if r["game"].startswith(src) and r["mode"].startswith("train")]
        if not rr:
            continue
        summ[src] = {}
        print("== %s, %d training copies" % (src, len(rr)))
        for k in keys:
            v = np.array([r.get(k) for r in rr if r.get(k) is not None], float)
            if len(v):
                q = np.percentile(v, [5, 25, 50, 75, 95])
                summ[src][k] = dict(zip(("p5", "p25", "p50", "p75", "p95"), map(float, q)))
                print("  %-16s p5 %7.3f  p25 %7.3f  p50 %7.3f  p75 %7.3f  p95 %7.3f" % ((k,) + tuple(q)))
    return rows, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0, help="draw this many sets of training parameters for the check")
    ap.add_argument("--games", default="idsse_J03WMX,idsse_J03WN1,metrica1")
    ap.add_argument("--out", default="degrade_check.json")
    ap.add_argument("--show", action="store_true", help="only print the parameters of the three benchmarks and of the first three training copies")
    a = ap.parse_args()
    if a.show:
        for g in a.games.split(","):
            for mode in ("rnd0", "rnd1", "rnd2", "train:0", "train:1", "train:2"):
                print(g, mode, json.dumps({k: round(v, 4) for k, v in params_for(g, mode)[0].items()}))
        return
    if a.check:
        rows, summ = check(a.check, a.games.split(","))
        out = os.path.expanduser(a.out)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump({"rows": rows, "summary": summ, "ranges": RANGES}, open(out, "w"), indent=1, default=str)
        print("written to " + out)


if __name__ == "__main__":
    main()
