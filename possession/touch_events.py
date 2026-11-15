"""
touch_events, find touch events from abrupt velocity changes of the raw ball position and assign each touch to the nearest player within r metres (2 m in both threshold sets of mtah_features.TOUCH_SETS)
============================
Possession is defined by the sequence of touches, not by the nearest player per frame. This module only answers "in which frame did someone touch the ball, and who was it". The method is as follows.
  1 Work within a block and only on frames where the ball is visible. With frame t as the boundary, the mean velocity vector vb over the k frames before and the mean velocity vector va over the k frames after both require the ball to be continuously visible over those k frames;
  2 when neither vector is slower than min_speed the angle between them is computed, and an angle above angle_deg counts as a "turn"; a speed difference above dv counts as a "speed change"; the score is the larger of the two;
  3 frames whose score reaches 1 are candidates, and within +/- nms frames only the one with the highest score is kept (non-maximum suppression);
  4 the toucher is the player slot nearest to the ball in that frame and within r metres; if there is none, extrapolate backwards along the velocity after the touch for at most back_max frames to find a player (the ball is often occluded while at a player's feet
    and has already travelled for a few frames when it becomes visible), and record the number of frames stepped back in back_frames; if still nobody is found the touch is unowned (team 0);
  5 the first frame in which the ball reappears after a ball gap is recorded as one more candidate touch, with kind reappear and no v_before, and the caller decides whether to use it.
No events and no ground truth are read; training and prediction share this code. The threshold sets are in mtah_features.TOUCH_SETS. The robust set was checked against the event table of the Metrica sample data;
the clean set is meant for low-jitter tracking such as the competition tracking (ball jitter about 0.01 m), and its assignment radius of 2 m and step back of 25 frames were set while checking that
precision and recall against the event table of the Metrica sample data stay unchanged and that the share of touches assigned to a player on the competition tracking rises (label-free). No ground truth of the competition match was used.

Inputs  M (the unified structure from the adapters; uses ball_xy in metres, player_xy in metres, slot_team, fps, blocks)
Outputs  a list sorted by frame number, each item a dict(frame, slot, team, dist, v_before, v_after, angle_deg, kind, score, back_frames)

Usage
  from possession.touch_events import detect_touches, summarize
  touches = detect_touches(M, angle_deg=30.0, dv=3.0, r=1.5)

Last modified 2026-11-15
"""
import numpy as np


def _run_mean_vectors(bxy, vis, k, fps):
    """Return vb (T,2) and va (T,2), the mean velocity vectors over the k frames before t and the k frames after t; the k frames must be continuously visible, otherwise NaN"""
    T = len(bxy)
    vb = np.full((T, 2), np.nan); va = np.full((T, 2), np.nan)
    # running count of consecutive visible frames, cnt[t] is the number of consecutive visible frames ending at t
    cnt = np.zeros(T, int)
    c = 0
    for t in range(T):
        c = c + 1 if vis[t] else 0
        cnt[t] = c
    ok_b = cnt >= k + 1                       # t-k to t all visible
    idx = np.flatnonzero(ok_b)
    idx = idx[idx >= k]
    vb[idx] = (bxy[idx] - bxy[idx - k]) * (fps / k)
    # the k frames after, t to t+k all visible, equivalent to cnt[t+k] >= k+1
    idx = np.flatnonzero(cnt >= k + 1)
    idx = idx[idx >= k]
    t0 = idx - k
    va[t0] = (bxy[idx] - bxy[t0]) * (fps / k)
    return vb, va


def detect_touches(M, angle_deg=30.0, dv=3.0, r=1.5, k=4, nms=6, min_speed=1.5, v_fast=0.0, include_reappear=True, back_max=10):
    """When v_fast is above 0, the speed change test also requires the ball speed on at least one side of the touch to be no lower than v_fast (noise while the ball rolls slowly at a player's feet does not count as a speed change).
    back_max is the maximum number of frames to extrapolate backwards along the velocity when the assignment finds no player (see _assign); 0 disables it."""
    fps = float(M["fps"])
    bxy = np.asarray(M["ball_xy"], np.float64)
    vis = ~np.isnan(bxy[:, 0])
    P = np.asarray(M["player_xy"], np.float32)
    st = np.asarray(M["slot_team"])
    out = []
    for h, (a, b) in M["blocks"].items():
        a, b = int(a), int(b)
        bb, vv = bxy[a:b + 1], vis[a:b + 1]
        vb, va = _run_mean_vectors(bb, vv, k, fps)
        sb, sa = np.linalg.norm(vb, axis=1), np.linalg.norm(va, axis=1)
        both = ~np.isnan(sb) & ~np.isnan(sa)
        # angle, computed only when neither side is slow
        cosang = np.full(len(bb), np.nan)
        m = both & (sb >= min_speed) & (sa >= min_speed)
        dot = (vb[m] * va[m]).sum(1)
        cosang[m] = np.clip(dot / (sb[m] * sa[m]), -1, 1)
        ang = np.degrees(np.arccos(cosang))
        s_turn = np.where(np.isnan(ang), 0.0, ang / angle_deg)
        s_speed = np.where(both & (np.maximum(np.nan_to_num(sa), np.nan_to_num(sb)) >= v_fast), np.abs(sa - sb) / dv, 0.0)
        score = np.maximum(s_turn, s_speed)
        cand = np.flatnonzero(score >= 1.0)
        # non-maximum suppression
        keep = []
        taken = np.zeros(len(bb), bool)
        for t in cand[np.argsort(-score[cand], kind="stable")]:
            if taken[t]:
                continue
            keep.append(t)
            taken[max(0, t - nms):t + nms + 1] = True
        for t in keep:
            kind = "both" if (s_turn[t] >= 1 and s_speed[t] >= 1) else ("turn" if s_turn[t] >= 1 else "speed")
            out.append(_assign(P, st, bxy, a + t, r, kind, float(sb[t]), float(sa[t]), float(ang[t]) if not np.isnan(ang[t]) else None, float(score[t]),
                               va[t], fps, back_max, a))
        if include_reappear:
            # first visible frame after a gap, with the previous frame not visible (the start of a block does not count)
            re = np.flatnonzero(vv[1:] & ~vv[:-1]) + 1
            for t in re:
                if taken[t]:
                    continue
                out.append(_assign(P, st, bxy, a + t, r, "reappear", None, float(sa[t]) if not np.isnan(sa[t]) else None, None, 0.0,
                                   va[t], fps, back_max, a))
    out.sort(key=lambda d: d["frame"])
    return out


def _assign(P, st, bxy, t, r, kind, v_before, v_after, angle, score, va_vec=None, fps=25.0, back_max=10, block_start=0):
    """Touch assignment. First look for the nearest player within r metres in frame t; if there is none, extrapolate backwards along the velocity vector after the touch for at most back_max frames (the ball is often occluded while at a player's feet
    and has already travelled for a few frames when it becomes visible), look in frame t-b for the nearest player within r metres of the extrapolated position, and take the smallest b. back_frames records the number of frames stepped back, 0 if none."""
    d = np.linalg.norm(P[t] - bxy[t][None, :].astype(np.float32), axis=1)
    d = np.where(np.isnan(d), np.inf, d)
    s = int(np.argmin(d)) if np.isfinite(d).any() else -1
    dist = float(d[s]) if s >= 0 else None
    back = 0
    if s >= 0 and dist <= r:
        slot, team = s, int(st[s])
    else:
        slot, team = -1, 0
        if va_vec is not None and not np.isnan(va_vec[0]) and np.linalg.norm(va_vec) > 1e-6:
            for b in range(1, back_max + 1):
                tb = t - b
                if tb < block_start:
                    break
                pos = (bxy[t] - va_vec * (b / fps)).astype(np.float32)
                db = np.linalg.norm(P[tb] - pos[None, :], axis=1)
                db = np.where(np.isnan(db), np.inf, db)
                sb_ = int(np.argmin(db)) if np.isfinite(db).any() else -1
                if sb_ >= 0 and db[sb_] <= r:
                    slot, team, dist, back = sb_, int(st[sb_]), float(db[sb_]), b
                    break
    return {"frame": int(t), "slot": slot, "team": team, "dist": dist, "v_before": v_before, "v_after": v_after,
            "angle_deg": angle, "kind": kind, "score": score, "back_frames": back}


def summarize(touches, fps=25.0):
    """Summary with counts only, for logging"""
    n = len(touches)
    kinds = {}
    for d in touches:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    owned = sum(1 for d in touches if d["team"] > 0)
    gaps = np.diff([d["frame"] for d in touches]) / fps if n > 1 else np.array([])
    return {"n_touches": n, "by_kind": kinds, "frac_owned": owned / n if n else None,
            "gap_s_median": float(np.median(gaps)) if len(gaps) else None}
