"""
adapters/mtah2027, reads the More Than A Hack 2027 competition data (BAR-ATH, Metrica automatic tracking)
============================
All formats follow DATA.md and starter/ of the official repository github.com/metrica-sports/more-than-a-hack-2027; nothing is inferred from the data content.
  Tracking (ATD), FIFA/EPTS export. The xml lists the two teams (the first one is the home team) and the player slots (teamId of each Player); every line of the txt is
  "<frame_id>:<slot 1>;<slot 2>;...:<ball>", each group is "x,y,speed", missing values are written NaN, coordinates are normalised to 0 to 1,
  frame_id starts at 1, video_frame = frame_id - 1, 199,319 lines in total. The pitch is 105.3 × 68.0 m, taken from games.py of the official repository.
  Only these two ATD files are read; the homography, the Smart Tagging and the video of the bundle are not used.
  Graded frames, first half 11185 to 90651, second half 110808 to 190197 (published in DATA.md).
  The second half is graded from frame 110808 but only kicks off at frame 113742; the 2934 frames in between are graded and are all D (stated in the section "Which frames are actually graded?" of the official FAQ.md), see SECOND_HALF_KICKOFF.
The tracking has no player identities, no goalkeeper flag and no live or dead ball flag; the field alive starts out equal to "inside the graded frames" and dead ball is decided by the method itself.
The tracking is parsed once and stored as an npz cache in cache/ under the data directory (the bundle directory must be writable).

Inputs  paths dictionary with the keys data_dir (root directory that contains tactical/) and official_repo (copy of the official repository that contains evaluation/gt, optional)
Outputs  a dictionary with the common match fields (fps, pitch, t_ms, period, blocks, ball_xy and so on), plus ball_visible, ball_speed, player_xy (all slots), slot_team, team_names, graded, public_gt (only with official_repo)

Usage
  from possession.adapters.mtah2027 import load
  M = load({"data_dir": "~/Data/more_than_a_hack_2027/data", "official_repo": "~/Data/more_than_a_hack_2027/official_repo"})

Last modified 2026-11-15
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

GAME = "BAR-ATH"
N_FRAMES = 199319
FPS = 25.0
PITCH = (105.3, 68.0)
GRADED_BLOCKS = {1: (11185, 90651), 2: (110808, 190197)}
SECOND_HALF_KICKOFF = 113742   # official FAQ.md, the second half is graded from frame 110,808 but does not kick off until 113,742, those 2,934 frames are graded and all of them are D


def _paths(data_dir, source="tactical"):
    stem = os.path.join(data_dir, source, "%s_%s" % (GAME, source))
    return {"atd_meta": stem + "_atd.xml", "atd_raw": stem + "_atd.txt"}


def read_meta(meta_path):
    """(home team name, away team name, array with the team of every slot, 1 home, 2 away). The first team in the xml is the home team (FIFA/EPTS convention, the official baseline does the same)"""
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("expected 2 teams in the ATD xml, found %d" % len(teams))
    home_id = teams[0][0]
    slot_team = np.array([1 if p.get("teamId") == home_id else 2 for p in root.iter("Player")], np.int8)
    if len(slot_team) == 0:
        raise ValueError("no Player in the ATD xml")
    return teams[0][1], teams[1][1], slot_team


def parse_atd_raw(raw_path, n_slots, n_frames=N_FRAMES):
    """Parses line by line and returns players (T, n_slots, 3) and ball (T, 3); the columns are x, y (0 to 1) and speed, missing values are NaN"""
    P = np.full((n_frames, n_slots, 3), np.nan, np.float32)
    B = np.full((n_frames, 3), np.nan, np.float32)
    n_lines = 0
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n_frames:
                continue
            n_lines += 1
            for s, g in enumerate(parts[1].split(";")[:n_slots]):
                if not g or g[0] == "N":
                    continue
                f = g.split(",")
                P[i, s, 0], P[i, s, 1] = float(f[0]), float(f[1])
                if len(f) > 2 and f[2][:1] != "N":
                    P[i, s, 2] = float(f[2])
            b = parts[2]
            if b and b[0] != "N":
                f = b.split(",")
                B[i, 0], B[i, 1] = float(f[0]), float(f[1])
                if len(f) > 2 and f[2][:1] != "N":
                    B[i, 2] = float(f[2])
    return P, B, n_lines


def load_atd(data_dir, source="tactical", cache_dir=None):
    """Reads the tracking, with an npz cache. Returns dict(players, ball, slot_team, home_name, away_name, n_lines)"""
    p = _paths(data_dir, source)
    home_name, away_name, slot_team = read_meta(p["atd_meta"])
    cache_dir = cache_dir or os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "%s_%s_atd.npz" % (GAME, source))
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(p["atd_raw"]):
        d = np.load(cache)
        players, ball, n_lines = d["players"], d["ball"], int(d["n_lines"])
    else:
        players, ball, n_lines = parse_atd_raw(p["atd_raw"], len(slot_team))
        np.savez_compressed(cache, players=players, ball=ball, n_lines=n_lines)
    return {"players": players, "ball": ball, "slot_team": slot_team, "home_name": home_name,
            "away_name": away_name, "n_lines": n_lines}


def load_public_gt(official_repo):
    """Public 5-minute ground truth labels, returns (T,) integers 0 none (X), 1 H, 2 A, 3 D. Used only to check the pipeline, never for training or tuning"""
    import pandas as pd
    path = os.path.join(official_repo, "evaluation", "gt", "%s_possession_gt_public.csv" % GAME)
    if not os.path.isfile(path):
        return None
    t = pd.read_csv(path, dtype={"label": "object"})
    lab = t["label"].astype(str).str.strip().str.upper().values
    return np.select([lab == "H", lab == "A", lab == "D"], [1, 2, 3], 0).astype(np.int8)


def load(paths, source="tactical", **kw):
    data_dir = os.path.expanduser(paths["data_dir"])
    atd = load_atd(data_dir, source)
    P, B, st = atd["players"], atd["ball"], atd["slot_team"]
    T = len(B)
    L, W = PITCH
    period = np.zeros(T, int)
    for h, (a, b) in GRADED_BLOCKS.items():
        period[a:b + 1] = h
    graded = period > 0
    hc, ac = np.where(st == 1)[0], np.where(st == 2)[0]
    M = {
        "fps": FPS, "pitch": PITCH, "t_ms": (np.arange(T) * (1000.0 / FPS)).astype(np.int64),
        "period": period, "alive": graded.copy(), "graded": graded,
        "ball_xy": np.column_stack([B[:, 0] * L, B[:, 1] * W]).astype(np.float64),
        "ball_z": np.full(T, np.nan), "has_ball_z": False,
        "ball_speed": B[:, 2].astype(np.float64), "ball_visible": ~np.isnan(B[:, 0]),
        "home_xy": np.stack([P[:, hc, 0] * L, P[:, hc, 1] * W], axis=2).astype(np.float64),
        "away_xy": np.stack([P[:, ac, 0] * L, P[:, ac, 1] * W], axis=2).astype(np.float64),
        "home_ids": [int(c) for c in hc], "away_ids": [int(c) for c in ac], "gk_home": [], "gk_away": [],
        "player_xy": np.stack([P[:, :, 0] * L, P[:, :, 1] * W], axis=2).astype(np.float32),
        "player_speed": P[:, :, 2], "slot_team": st, "team_names": (atd["home_name"], atd["away_name"]),
        "events": None, "labels": None, "n_lines": atd["n_lines"], "data_dir": data_dir,
    }
    if paths.get("official_repo"):
        M["public_gt"] = load_public_gt(os.path.expanduser(paths["official_repo"]))
    return M
