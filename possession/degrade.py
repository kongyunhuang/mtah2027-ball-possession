"""
degrade, degradations of public training matches (ball removal, player dropout, position noise, ball offset) and their label-free summary statistics, numpy only
============================
Shared by the training scripts (mtah_v3.py, seq_model.py, degrade_random.py), so that they do not depend on any dataset-specific script.
  realistic_missing  removes the ball the way automatic tracking data (ATD) loses it, with a per-frame hazard rate; the ball disappears when it is slow and close to a player and reappears when it moves fast; the scale is bisected to reach the total missing share
  player_dropout     drops players in segments until the share of graded frames with fewer than 20 players reaches the target
  ball_offset        pushes away a ball position that sticks to a player's feet (used for the Metrica sample data)
  visibility_stats   summary statistics telling whether the ball removal looks like ATD (share visible, share within 1.5 m, share of fast balls at the first and last frame of ball gaps, gap duration, share of frames with fewer than 20 players)
  ar1_noise          AR(1) noise along time

Inputs  see the docstring of each function
Outputs  degraded arrays or a dictionary of statistics

Usage
  from possession.degrade import realistic_missing, player_dropout, visibility_stats, ar1_noise

Last modified 2026-11-15
"""
import numpy as np


def realistic_missing(rng, bxy, P, fps, cand, denom, miss_frac, d_scale=0.7, fast_w=0.05, p_fast=0.02, p_slow=0.0002, v_fast=5.0,
                      existing=None, tol=0.01, min_gap=75, near_boost=1.0, near_d=0.5):
    """Removes the ball the way automatic tracking loses it (per-frame hazard rate version). Such tracking loses the ball mostly when it is slow and close to a player (occlusion),
    and the ball reappears once it moves fast. Two hazard rates are simulated frame by frame. While visible, the ball disappears with probability a·exp(-nearest distance/d_scale)·(times fast_w for a fast ball);
    while missing, it reappears with probability p_fast (fast ball) or p_slow (slow ball). The scale a is found by bisection so that the total missing share equals miss_frac.
    cand marks the frames where a gap may start, denom the frames over which the missing share is counted, existing the frames that are already missing. Returns the boolean array missing.
    The default values of d_scale, fast_w, p_fast, p_slow and min_gap were set with visibility_stats from label-free summary statistics of the competition tracking (no labels).
    They are used only by the fixed degradation of mtah_v3 (degrade_like_wc, augment_metrica), which in the submitted pipeline builds only the three fixed benches of --extra_eval.
    The randomised degradation of degrade_random draws d_scale, fast_w, p_fast, p_slow, min_gap and the missing share from public ranges. v_fast (5 m/s, a short pass or faster), tol (tolerance of the
    bisection), near_boost (1, no effect) and near_d (without effect while near_boost is 1) are fixed defaults of the ball-loss model and are the same in every training copy"""
    T = len(bxy)
    b = np.asarray(bxy, np.float64)
    v = np.zeros(T)
    v[3:] = np.nan_to_num(np.linalg.norm(b[3:] - b[:-3], axis=1) * fps / 3, nan=0.0)
    d = np.linalg.norm(np.asarray(P, np.float32) - b[:, None, :].astype(np.float32), axis=2)
    d = np.where(np.isnan(d), 99.0, d).min(1)
    fast = v > v_fast
    base = np.where(cand, np.exp(-d / d_scale) * np.where(fast, fast_w, 1.0) * np.where(d < near_d, near_boost, 1.0), 0.0)   # a ball at a player's feet is further multiplied by near_boost
    u = rng.random(T)                      # one random number per frame, reused across the bisection steps so that the result is monotonic in a
    def simulate(a):
        off = np.minimum(1.0, a * base)
        on = np.where(fast, p_fast, p_slow)
        miss = np.zeros(T, bool) if existing is None else existing.copy()
        m = False
        since = 0
        for t in range(T):
            if existing is not None and existing[t]:
                m = True
                since = 0
                continue
            if m:
                since += 1
                if since >= min_gap and u[t] < on[t]:
                    m = False
            else:
                if u[t] < off[t]:
                    m = True
                    since = 0
            miss[t] = m
        return miss
    lo, hi = np.log(1e-4), np.log(1.0)
    best = None
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        miss = simulate(np.exp(mid))
        frac = (miss & denom).sum() / max(1, denom.sum())
        best = miss
        if abs(frac - miss_frac) < tol:
            break
        if frac < miss_frac:
            lo = mid
        else:
            hi = mid
    return best


def ball_offset(rng, bxy, P, fps, sigma, phi=0.98, near=0.3, v_slow=3.0):
    """Pushes away a ball position that sticks to a player's feet (the ball of the Metrica public sample data is rebuilt from manually tagged events, and on slow-ball frames its median distance
    to the nearest player is only 0.06 to 0.11 m, much closer than in automatic tracking). Only frames where the ball is visible, slower than v_slow and less than near metres from the nearest player are changed. A two-dimensional AR(1) offset is added (coefficient phi,
    stationary standard deviation sigma metres); the offset drifts slowly over time and is not per-frame white noise. Returns the new ball positions"""
    b = np.asarray(bxy, np.float64).copy()
    T = len(b)
    vis = ~np.isnan(b[:, 0])
    v = np.full(T, np.inf)
    v[3:] = np.linalg.norm(b[3:] - b[:-3], axis=1) * fps / 3
    d = np.linalg.norm(np.asarray(P, np.float32) - b[:, None, :].astype(np.float32), axis=2)
    d = np.where(np.isnan(d), 99.0, d).min(1)
    m = vis & (np.nan_to_num(v, nan=np.inf) < v_slow) & (d < near)
    off = ar1_noise(rng, (T, 2), sigma, phi)
    b[m] += off[m]
    return b


def player_dropout(rng, P, graded, frac_lt20=0.15, med=100.0):
    """Random player dropout in segments (automatic tracking has graded frames with fewer than 20 players on the pitch, the other sources have none). Each event drops 1 to 3 slots, with a log-normal length (median med frames).
    The defaults frac_lt20 0.15 and med 100 were set from label-free statistics of the competition tracking and are used only by the fixed degradation of mtah_v3 (the three fixed benches);
    degrade_random uses its own player_dropout_r with the share and the segment length drawn from public ranges"""
    P = np.asarray(P, np.float32).copy()
    T, S, _ = P.shape
    n = (~np.isnan(P[:, :, 0])).sum(1)
    guard = 0
    while ((n < 20) & graded).sum() < frac_lt20 * graded.sum() and guard < 100000:
        guard += 1
        st = int(rng.integers(0, T))
        if not graded[st]:
            continue
        length = max(5, int(np.exp(np.log(med) + 0.8 * rng.standard_normal())))
        for slot in rng.choice(S, size=int(rng.integers(1, 4)), replace=False):
            P[st:st + length, slot] = np.nan
        n[st:st + length] = (~np.isnan(P[st:st + length, :, 0])).sum(1)
    return P


def visibility_stats(M, mask=None):
    """Summary statistics telling whether the ball removal looks like automatic tracking (shares and medians only). Share of frames with the ball visible, median distance from the visible ball to the nearest player,
    share within 1.5 m, share of ball gaps whose first frame has a ball speed above 5 m/s, the same for the last frame, median and 90th percentile of the gap duration, share of frames with fewer than 20 players"""
    g = M["graded"] if mask is None else mask
    b = np.asarray(M["ball_xy"], np.float64)
    P = np.asarray(M["player_xy"], np.float32)
    fps = float(M["fps"])
    vis = ~np.isnan(b[:, 0])
    d = np.linalg.norm(P - b[:, None, :].astype(np.float32), axis=2)
    dmin = np.where(np.isnan(d), 99.0, d).min(1)
    out = {"vis": float(vis[g].mean()), "dmin_med": float(np.median(dmin[g & vis])), "lt15": float((dmin[g & vis] < 1.5).mean()),
           "lt20_players": float(((~np.isnan(P[:, :, 0])).sum(1)[g] < 20).mean())}
    starts, ends = [], []
    for a, bb in M["blocks"].values():
        vv = vis[a:bb + 1]
        e = np.diff(np.r_[1, vv.astype(np.int8), 1])
        s_ = np.flatnonzero(e == -1) + a
        e_ = np.flatnonzero(e == 1) + a
        for s0, e0 in zip(s_, e_):
            if s0 <= a or e0 > bb or e0 - s0 < 2:
                continue
            starts.append(s0 - 1)
            ends.append(e0)
    starts, ends = np.array(starts, int), np.array(ends, int)
    def spd(t, fwd):
        t0, t1 = (t - 3, t) if fwd else (t, t + 3)
        ok = (t0 >= 0) & (t1 < len(b)) & vis[np.clip(t0, 0, len(b) - 1)] & vis[np.clip(t1, 0, len(b) - 1)]
        sp = np.zeros(len(t))
        sp[ok] = np.linalg.norm(b[t1[ok]] - b[t0[ok]], axis=1) * fps / 3
        return sp
    if len(starts):
        out["gap_start_fast"] = float((spd(starts, True) > 5).mean())
        out["gap_end_fast"] = float((spd(ends, False) > 5).mean())
        out["gap_med_s"] = float(np.median(ends - starts - 1) / fps)
        out["gap_p90_s"] = float(np.percentile(ends - starts - 1, 90) / fps)
        out["n_gaps"] = int(len(starts))
    return out


def ar1_noise(rng, shape, sigma, phi=0.9):
    """AR(1) noise along axis 0 with stationary standard deviation sigma"""
    e = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), shape)
    out = np.empty(shape)
    out[0] = rng.normal(0, sigma, shape[1:])
    for t in range(1, shape[0]):
        out[t] = phi * out[t - 1] + e[t]
    return out
