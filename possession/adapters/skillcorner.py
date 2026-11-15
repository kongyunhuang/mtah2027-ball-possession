"""
adapters/skillcorner, reads the SkillCorner open data (A-League 2024-25, broadcast tracking, MIT licence) and builds frame-level possession and live or dead ball ground truth labels
============================
All data conventions come from the README of the SkillCorner open data repository.
  Tracking file {id}_tracking_extrapolated.jsonl has one frame per line at 10 fps, with keys frame, timestamp, period, ball_data (x, y, z, is_detected),
  possession (player_id, group) and player_data (x, y, player_id, is_detected). Units are metres, the origin is the pitch centre, x runs along the long side, y along the short side.
  {id}_match.json gives home_team.id, players (id, team_id), pitch_length and pitch_width.
  {id}_phases_of_play.csv has one phase per row with frame_start and frame_end; the README states that phases are only defined while the ball is live.
  {id}_dynamic_events.csv rows whose event_type is player_possession carry frame_start, frame_end and team_id.
Method
  1 Coordinates are shifted by half a pitch so that the origin is at a corner (same as adapters/idsse.py). The ball is kept only on frames where is_detected is true;
    an extrapolated ball is treated as missing (close to the tactical camera habit that an unseen ball does not exist). Player positions include extrapolated ones
    (players outside the picture), which is close to tracking data where nearly all players on the pitch are present.
  2 When the broadcast cuts away (replays, crowd shots) a whole frame has neither players nor ball; this is about 27% of the frames, almost all during dead ball time.
    Each half is split into contiguous blocks of frames that have player data; blocks shorter than min_block_s seconds are not graded.
    The block number is written into period, so feature time windows never cross a block boundary.
  3 Ground truth labels. The player_possession intervals give the team in possession. Live ball frames (inside a phase) are forward filled from these intervals
    (the same convention as the event chain of the Metrica sample data); frames with data but outside any phase are labelled dead ball D.
    A check on 3 matches showed that on frames with a slow, detected ball the team of the nearest player equals the ground truth labels in 0.90 to 0.95 of cases,
    while the frame-level possession.group reaches only 0.80 to 0.88, so group is not used.
  4 Linear interpolation from 10 fps to 25 fps. Ball and players are interpolated only where both ends have values; labels take the nearest original frame.
    The parsed match is cached as npz in cache/ under root.

Inputs  root (the opendata directory, containing matches/), match_id such as "1874553"
Outputs  (M, gt), same structure as adapters/idsse.py; gt is 0 not graded, 1 home team, 2 away team, 3 dead ball

Usage
  from possession.adapters.skillcorner import list_matches, load_match
  M, gt = load_match("~/Data/alt_training/skillcorner_opendata", "1874553")

Last modified 2026-11-15
"""
import json
import os

import numpy as np
import pandas as pd

SRC_FPS = 10.0
FPS = 25.0


def list_matches(root):
    d = os.path.join(os.path.expanduser(root), "matches")
    return sorted(m for m in os.listdir(d) if m.isdigit())


def _parse(root, mid):
    """Read the raw files and return arrays at 10 fps."""
    d = os.path.join(os.path.expanduser(root), "matches", mid)
    mj = json.load(open(os.path.join(d, "%s_match.json" % mid)))
    home = mj["home_team"]["id"]
    pids = sorted({p["id"] for p in mj["players"]})
    slot = {p: i for i, p in enumerate(pids)}
    team_of = {p["id"]: (1 if p["team_id"] == home else 2) for p in mj["players"]}
    slot_team = np.array([team_of[p] for p in pids], np.int8)
    rows = [json.loads(line) for line in open(os.path.join(d, "%s_tracking_extrapolated.jsonl" % mid))]
    T = len(rows)
    L, W = float(mj["pitch_length"]), float(mj["pitch_width"])
    fr = np.array([r["frame"] for r in rows], int)
    period = np.array([r["period"] or 0 for r in rows], np.int16)
    B = np.full((T, 2), np.nan, np.float32)
    P = np.full((T, len(pids), 2), np.nan, np.float32)
    has = np.zeros(T, bool)
    for i, r in enumerate(rows):
        b = r["ball_data"] or {}
        if b.get("x") is not None and b.get("is_detected"):
            B[i] = (b["x"] + L / 2, b["y"] + W / 2)
        pl = r["player_data"] or []
        if pl:
            has[i] = True
            for q in pl:
                j = slot.get(q["player_id"])
                if j is not None:
                    P[i, j] = (q["x"] + L / 2, q["y"] + W / 2)
    idx = {f: i for i, f in enumerate(fr)}
    alive = np.zeros(T, bool)
    ph = pd.read_csv(os.path.join(d, "%s_phases_of_play.csv" % mid))
    for a, b in zip(ph.frame_start, ph.frame_end):
        if a in idx and b in idx:
            alive[idx[a]:idx[b] + 1] = True
    ev = pd.read_csv(os.path.join(d, "%s_dynamic_events.csv" % mid), low_memory=False)
    pp = ev[ev.event_type == "player_possession"]
    raw = np.zeros(T, np.int8)
    for a, b, t in zip(pp.frame_start, pp.frame_end, pp.team_id):
        a, b = int(a), int(b)
        if a in idx and b in idx:
            raw[idx[a]:idx[b] + 1] = 1 if t == home else 2
    return {"B": B, "P": P, "has": has, "alive": alive, "raw": raw, "period": period, "slot_team": slot_team, "pitch": np.array([L, W])}


def _upsample(x, n_out, ratio):
    """Linear interpolation from 10 fps to 25 fps, only where both ends have values. x has shape (T, ...)."""
    s = np.arange(n_out) / ratio
    i0 = np.floor(s).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (s - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return (x[i0] * (1 - w) + x[i1] * w).astype(np.float32)   # NaN at either end gives NaN


def load_match(root, mid, min_block_s=30.0, cache=True):
    cache_dir = os.path.join(os.path.expanduser(root), "cache")
    cpath = os.path.join(cache_dir, "%s.npz" % mid)
    if cache and os.path.exists(cpath):
        z = np.load(cpath)
        A = {k: z[k] for k in z.files}
    else:
        A = _parse(root, mid)
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cpath, **A)
    T10 = len(A["B"])
    ratio = FPS / SRC_FPS
    T = int(np.floor((T10 - 1) * ratio)) + 1
    near = np.minimum(np.round(np.arange(T) / ratio).astype(int), T10 - 1)
    B = _upsample(A["B"], T, ratio)
    P = _upsample(A["P"], T, ratio)
    per10 = A["period"].astype(int)
    has10, alive10, raw10 = A["has"].astype(bool), A["alive"].astype(bool), A["raw"].astype(np.int8)
    # Ground truth labels at 10 fps. The team in possession is forward filled within each half; live ball frames take it, frames with data but no live ball are dead ball
    gt10 = np.zeros(T10, np.int8)
    last, prev_p = 0, 0
    for i in range(T10):
        p = per10[i]
        if p != prev_p:
            last, prev_p = 0, p
        if p == 0:
            continue
        if raw10[i] > 0:
            last = raw10[i]
        if not has10[i]:
            continue
        gt10[i] = last if alive10[i] else 3
    gt = gt10[near]
    has = has10[near]
    per = per10[near]
    # Split into blocks of frames that have player data; block number = half * 100 + index
    seg = np.zeros(T, np.int16)
    blocks = {}
    for p in (1, 2):
        m = (per == p) & has
        e = np.diff(np.r_[0, m.astype(np.int8), 0])
        for k, (a, b) in enumerate(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))):
            if (b - a) / FPS < min_block_s:
                continue
            sid = p * 100 + k
            seg[a:b] = sid
            blocks[int(sid)] = (int(a), int(b - 1))
    graded = seg > 0
    gt[~graded] = 0
    B[~graded] = np.nan
    P[~graded] = np.nan
    M = {"fps": FPS, "pitch": tuple(float(v) for v in A["pitch"]), "t_ms": (np.arange(T) * (1000.0 / FPS)).astype(np.int64),
         "period": seg, "graded": graded, "alive": graded.copy(), "ball_xy": B.astype(np.float64), "ball_visible": ~np.isnan(B[:, 0]),
         "ball_speed": np.full(T, np.nan), "ball_z": np.full(T, np.nan), "has_ball_z": False,
         "player_xy": P, "slot_team": A["slot_team"].astype(np.int8), "tags": [], "blocks": blocks, "team_names": ("home", "away"),
         "game": "skillcorner_%s" % mid}
    return M, gt
