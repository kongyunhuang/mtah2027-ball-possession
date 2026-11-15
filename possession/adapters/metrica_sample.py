"""
adapters/metrica_sample, reads the public Metrica Sports sample data (github.com/metrica-sports/sample-data, 3 games) and derives frame-level ground truth labels from the events
============================
Games 1 and 2 are in the Metrica CSV format (one tracking file per team; the first two rows hold the team name and the shirt numbers, the third row the column names Period, Frame, Time [s],
x,y of each player and x,y of the ball; coordinates run from 0 to 1 with (0,0) at the top left), and the events are a CSV (Team, Type, Subtype, Period, Start Frame, End Frame and so on).
Game 3 is in the EPTS FIFA format (metadata xml plus tracking txt, the same format as the competition ATD files, 22 slots), and the events are JSON.
The pitch is 105 × 68 m at 25 fps; the README says so for all three games. The ground truth labels are derived from the event chain.
  SET PIECE starts live ball with possession for that team; RECOVERY also starts live ball when the ball is dead; PASS, CARRY and SHOT give possession to that team; BALL LOST hands possession
  to the opponent from its end frame; the team that wins a CHALLENGE takes the ball; BALL OUT and FAULT RECEIVED start dead ball from their end frame; a SHOT with subtype GOAL is followed by dead ball;
  the ball is dead before the kick-off of each half; between events the previous state is kept. Labels are 0 none, 1 home, 2 away, 3 dead ball.
With the environment variable MTAH_METRICA_FIX_DEAD=1 (set for all submitted models, off by default) four more dead-ball rules apply, so that goals, offsides, handballs, fouls, shots that go out and the end of a half
start a dead ball; the base chain above recognises only BALL OUT, FAULT RECEIVED and one spelling of a goal as the start of a dead ball. The four rules.
  When a set piece arrives while the state is still live ball, the frames from after the end frame of the previous on-ball event (CARD is skipped) up to the start of the set piece become dead ball.
  A SHOT whose subtype contains OUT starts dead ball from its end frame, until the next RECOVERY or SET PIECE (goal kicks are recorded as RECOVERY in this dataset).
  A BALL LOST whose subtype contains END HALF starts dead ball from its end frame.
  Within live-ball labels, a segment where the ball is missing for more than 8 s in a row becomes dead ball (in this dataset the ball is shown during live play and hardly ever during dead balls; with all rules
  the ball is visible in about 99 percent of the live-ball frames and in at most 7 percent of the dead-ball frames, as the technical document reports).
With the environment variable MTAH_METRICA_FIX_OVERLAP=1 (set for all submitted models, off by default) dead ball after the ball goes out of play starts after the end frame of the out-of-play event,
where the ball leaves the tracking. When the state is already dead ball, an event whose start falls inside the time span of the previous event writes only the frames after that span
(without the switch, dead ball starts at the start of the out-of-play event in 53 of 64 cases of game 3).
With the environment variable MTAH_METRICA_CONTROL=1 (off by default, not used by the submitted models) two rules follow the possession definition in CHALLENGE.md of the official repository, which says that
while the ball is contested or under nobody's control possession stays with the team that last had it, and that winning an aerial duel without the team then keeping the ball is not a change of possession.
BALL LOST then hands possession to the opponent at the opponent's next possession event, and winning a CHALLENGE keeps the team.
The output has the same structure as adapters/mtah2027.py (ball_xy in metres, player_xy for every slot, slot_team, period, blocks).

Inputs  game_dir (root directory that holds the three games), game number 1, 2 or 3
Outputs  (M, gt)

Usage
  from possession.adapters.metrica_sample import load_game
  M, gt = load_game("~/Data/alt_training/metrica_sample", 3)

Last modified 2026-11-15
"""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .mtah2027 import parse_atd_raw, read_meta

FPS = 25.0
PITCH = (105.0, 68.0)
POSSESSING = ("PASS", "CARRY", "SHOT", "RECOVERY")


def _read_csv_tracking(path):
    t = pd.read_csv(path, skiprows=2)
    cols = list(t.columns)
    # player columns come in pairs PlayerNN, Unnamed; the ball is the last two columns
    xy_cols = [(cols[i], cols[i + 1]) for i in range(3, len(cols) - 2, 2)]
    P = np.stack([t[[a, b]].values.astype(np.float32) for a, b in xy_cols], axis=1)
    ball = t[cols[-2:]].values.astype(np.float32)
    return t["Period"].values.astype(int), t["Frame"].values.astype(int), P, ball


def _events_csv(path):
    e = pd.read_csv(path)
    out = []
    for r in e.itertuples(index=False):
        out.append({"team": 1 if str(r[0]).strip().lower() == "home" else 2, "type": str(r[1]).strip().upper(),
                    "subtype": str(r[2]).strip().upper() if isinstance(r[2], str) else "",
                    "period": int(r[3]), "start": int(r[4]), "end": int(r[6]) if r[6] == r[6] else int(r[4])})
    return out


def _events_json(path, home_id):
    d = json.load(open(path))["data"]
    out = []
    for e in d:
        subs = e.get("subtypes")
        if isinstance(subs, dict):
            subs = [subs]
        names = " ".join((s.get("name") or "").upper() for s in (subs or []) if isinstance(s, dict))
        end = e.get("end") or e["start"]
        out.append({"team": 1 if (e.get("team") or {}).get("id") == home_id else 2, "type": (e["type"]["name"] or "").upper(),
                    "subtype": names, "period": int(e["period"]), "start": int(e["start"]["frame"]), "end": int(end.get("frame") or e["start"]["frame"])})
    return out


def chain_labels(events, period, T):
    """Derives the per-frame labels from the event chain, 0 none, 1 home, 2 away, 3 dead ball"""
    fix_dead = os.environ.get("MTAH_METRICA_FIX_DEAD") == "1"
    # MTAH_METRICA_CONTROL=1 (off by default) follows the possession definition of CHALLENGE.md. After BALL LOST, while nobody controls the ball, possession still counts for the previous team
    # and only changes at the opponent's next PASS, CARRY, SHOT or RECOVERY; winning a CHALLENGE without then keeping the ball does not change the team
    control = os.environ.get("MTAH_METRICA_CONTROL") == "1"
    # MTAH_METRICA_FIX_OVERLAP=1 (off by default). In the JSON events of game 3 an out-of-play event (BALL OUT, spanning from the touch to the ball leaving the pitch) is often followed directly by
    # a one-frame CARRY with the same start, and in the tracking the ball disappears about 1 frame after the end of the out-of-play event (game 1 is similar). With the switch, when the state is already dead ball,
    # an event whose start falls inside the span already processed (except set pieces and RECOVERY, which restart live ball) only writes the frames after cur, and cur is not moved back, so dead ball starts
    # after the out-of-play event; without it the CARRY moves cur back and dead ball starts at the start of the out-of-play event (53 of 64 cases, 32 frames early in the median)
    fix_ov = os.environ.get("MTAH_METRICA_FIX_OVERLAP") == "1"
    lab = np.zeros(T, np.int8)
    ev = sorted(events, key=lambda e: (e["period"], e["start"]))
    # first and last frame of each half (in the tracking)
    bounds = {}
    for p in np.unique(period):
        if p <= 0:
            continue
        idx = np.flatnonzero(period == p)
        bounds[int(p)] = (int(idx[0]), int(idx[-1]))
    for p, (a, b) in bounds.items():
        state, team = 3, 0
        cur = a
        last_ball_end = None                      # end frame of the previous on-ball event (CARD excluded), used by MTAH_METRICA_FIX_DEAD
        for e in [x for x in ev if x["period"] == p]:
            s0 = min(max(e["start"] - 1, a), b)       # frame numbers start at 1, array indices at 0
            s1 = min(max(e["end"] - 1, a), b)
            lab[cur:s0] = team if (state == 1 and team) else 3
            ty, tm, sub = e["type"], e["team"], e["subtype"]
            if fix_ov and state == 3 and s0 < cur and ty not in ("SET PIECE", "RECOVERY"):
                if s1 >= cur:
                    lab[cur:s1 + 1] = 3
                    cur = s1 + 1
                if ty != "CARD":
                    last_ball_end = s1 if last_ball_end is None else max(last_ball_end, s1)
                continue
            if fix_dead and ty == "SET PIECE" and state == 1 and last_ball_end is not None:
                lab[last_ball_end + 1:s0] = 3
            if ty != "CARD":
                last_ball_end = s1
            if ty == "SET PIECE":
                state, team = 1, tm
                lab[s0:s1 + 1] = tm
            elif ty == "RECOVERY":
                state, team = 1, tm
                lab[s0:s1 + 1] = tm
            elif ty in POSSESSING:
                if state == 1:
                    team = tm
                    lab[s0:s1 + 1] = tm
                else:
                    lab[s0:s1 + 1] = 3
                if ty == "SHOT" and "GOAL" in sub and "OWN" not in sub and "ON TARGET" not in sub.replace("GOAL", "") or (ty == "SHOT" and sub.split() and sub.split()[-1] == "GOAL"):
                    state = 3
                if fix_dead and ty == "SHOT" and "OUT" in sub:
                    state = 3
            elif ty == "BALL LOST":
                lab[s0:s1 + 1] = team if (state == 1 and team) else 3
                if state == 1 and not control:
                    team = 3 - tm
                if fix_dead and "END HALF" in sub:
                    state = 3
            elif ty == "CHALLENGE":
                lab[s0:s1 + 1] = team if (state == 1 and team) else 3
                if state == 1 and "WON" in sub and not control:
                    team = tm
            elif ty in ("BALL OUT", "FAULT RECEIVED"):
                lab[s0:s1 + 1] = team if (state == 1 and team) else 3
                state = 3
            else:
                lab[s0:s1 + 1] = team if (state == 1 and team) else 3
            cur = s1 + 1
        lab[cur:b + 1] = team if (state == 1 and team) else 3
    return lab


def load_game(root, game):
    root = os.path.expanduser(root)
    g = os.path.join(root, "Sample_Game_%d" % game)
    L, W = PITCH
    if game in (1, 2):
        ph, fh, Ph, bh = _read_csv_tracking(os.path.join(g, "Sample_Game_%d_RawTrackingData_Home_Team.csv" % game))
        pa, fa, Pa, ba = _read_csv_tracking(os.path.join(g, "Sample_Game_%d_RawTrackingData_Away_Team.csv" % game))
        assert np.array_equal(fh, fa), "home and away tracking frames differ"
        period, frames = ph, fh
        P = np.concatenate([Ph, Pa], axis=1)
        slot_team = np.array([1] * Ph.shape[1] + [2] * Pa.shape[1], np.int8)
        ball = bh
        events = _events_csv(os.path.join(g, "Sample_Game_%d_RawEventsData.csv" % game))
    else:
        # The EPTS column order is set by the DataFormatSpecifications (it changes after substitutions), not by the order of the Player elements.
        # Work out who is in each column from the channel order of every specification, then rearrange into fixed slots, the first n columns home and the last n columns away.
        root_xml = ET.parse(os.path.join(g, "Sample_Game_3_metadata.xml")).getroot()
        teams = [t.get("id") for t in root_xml.iter("Team")]
        home_id = teams[0]
        team_of_player = {p.get("id"): (1 if p.get("teamId") == home_id else 2) for p in root_xml.iter("Player")}
        chan_player = {c.get("id"): c.get("playerId") for c in root_xml.iter("PlayerChannel")}
        specs = []
        for dfs in root_xml.iter("DataFormatSpecification"):
            pids = []
            for ref in dfs.iter("PlayerChannelRef"):
                pid = chan_player[ref.get("playerChannelId")]
                if not pids or pids[-1] != pid:
                    pids.append(pid)
            specs.append((int(dfs.get("startFrame")), int(dfs.get("endFrame")), pids))
        params = {p.findtext("Name"): p.findtext("Value") for p in root_xml.iter("ProviderParameter") if p.findtext("Name")}
        n = int(params["second_half_end"])
        n_cols = max(len(s[2]) for s in specs)
        P3, B3, _ = parse_atd_raw(os.path.join(g, "Sample_Game_3_tracking.txt"), n_cols, n)
        n_per = max(max(sum(team_of_player[p] == t for p in s[2]) for s in specs) for t in (1, 2))
        P = np.full((n, 2 * n_per, 2), np.nan, np.float32)
        for a, b, pids in specs:
            a0, b0 = max(a - 1, 0), min(b, n)      # frame numbers start at 1
            col = {1: 0, 2: n_per}
            for s, pid in enumerate(pids):
                t = team_of_player[pid]
                if col[t] < (n_per if t == 1 else 2 * n_per):
                    P[a0:b0, col[t]] = P3[a0:b0, s, :2]
                    col[t] += 1
        slot_team = np.array([1] * n_per + [2] * n_per, np.int8)
        ball = B3[:, :2]
        frames = np.arange(1, n + 1)
        period = np.zeros(n, int)
        period[int(params["first_half_start"]) - 1:int(params["first_half_end"])] = 1
        period[int(params["second_half_start"]) - 1:int(params["second_half_end"])] = 2
        events = _events_json(os.path.join(g, "Sample_Game_3_events.json"), home_id)
    T = len(frames)
    bxy = np.column_stack([ball[:, 0] * L, ball[:, 1] * W]).astype(np.float64)
    graded = period > 0
    blocks = {}
    for p in np.unique(period):
        if p > 0:
            idx = np.flatnonzero(period == p)
            blocks[int(p)] = (int(idx[0]), int(idx[-1]))
    M = {"fps": FPS, "pitch": PITCH, "t_ms": (frames * (1000.0 / FPS)).astype(np.int64), "period": period, "graded": graded,
         "alive": graded.copy(), "ball_xy": bxy, "ball_visible": ~np.isnan(bxy[:, 0]), "ball_speed": np.full(T, np.nan),
         "ball_z": np.full(T, np.nan), "has_ball_z": False,
         "player_xy": np.stack([P[:, :, 0] * L, P[:, :, 1] * W], axis=2).astype(np.float32), "slot_team": slot_team,
         "tags": [], "blocks": blocks, "team_names": ("home", "away"), "game": "metrica_sample_%d" % game}
    gt = chain_labels(events, period, T)
    gt[~graded] = 0
    if os.environ.get("MTAH_METRICA_FIX_DEAD") == "1":     # within live-ball labels, a segment where the ball is missing for more than 8 s in a row becomes dead ball
        miss = np.isin(gt, (1, 2)) & np.isnan(bxy[:, 0])
        b = np.flatnonzero(np.diff(np.r_[0, miss.astype(np.int8), 0]))
        for i, j in zip(b[::2], b[1::2]):
            if j - i > 8 * FPS:
                gt[i:j] = 3
    return M, gt
