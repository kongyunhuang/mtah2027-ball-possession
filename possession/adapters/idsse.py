"""
adapters/idsse, reads the public IDSSE dataset (seven matches of German professional football, Bassek et al. 2025, CC BY 4.0) and takes the official frame-level possession and ball status as ground truth labels
============================
The files are in the DFL format. The matchinformation xml gives the home and away team ids (the Role of a Team is home or guest) and the pitch size (PitchX and PitchY of Environment, in metres);
the positions xml has one FrameSet for each player and for the ball (one per GameSection), and a Frame carries N (frame number), X and Y (metres, origin at the pitch centre);
a Frame of the ball also carries BallPossession (1 or 2) and BallStatus (1 is live ball). Coordinates are converted here to the convention with the origin at a pitch corner, x plus PitchX/2 and y plus PitchY/2.
Ground truth labels. BallStatus other than 1 is dead ball; in live-ball frames the team comes from BallPossession, and which of 1 and 2 is the home team is settled by one check against the team of the nearest player
(the agreement rate should be clearly above 0.5). The DFL possession field can change hands at a single touch, while CHALLENGE.md of the official repository lists sustained control
over a few seconds as one possible proxy of possession, so possession spells shorter than min_spell frames can optionally be merged into the previous spell. The files are parsed once and cached as npz.
Optional environment variables, all off by default. MTAH_IDSSE_DROP_OFFICIALS=1 drops the match officials, who are otherwise counted in the away team. MTAH_IDSSE_LABEL_SHIFT=k moves the changes of possession
in the training labels k frames earlier (the scoring labels, with min_spell 0, are not affected). MTAH_IDSSE_CHAIN=early or isolated treats a run of short alternating spells as a whole.

Inputs  root (directory that holds the files of the 7 matches), match_id such as J03WOH
Outputs  (M, gt), same structure as adapters/mtah2027.py

Usage
  from possession.adapters.idsse import load_match, list_matches
  M, gt = load_match("~/Data/alt_training/idsse", "J03WOH", min_spell=50)

Last modified 2026-11-15
"""
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25.0


def list_matches(root):
    root = os.path.expanduser(root)
    return sorted(os.path.basename(p).split("DFL-MAT-")[1].split(".")[0] for p in glob.glob(os.path.join(root, "DFL_04_03_positions_raw_observed_*.xml")))


def _match_info(root, mid):
    path = glob.glob(os.path.join(root, "DFL_02_01_matchinformation_*%s.xml" % mid))[0]
    r = ET.parse(path).getroot()
    gen = r.find(".//General")
    env = r.find(".//Environment")
    home, guest = gen.get("HomeTeamId"), gen.get("GuestTeamId")
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    return home, guest, pitch


def _parse_positions(root, mid):
    path = glob.glob(os.path.join(root, "DFL_04_03_positions_raw_observed_*%s.xml" % mid))[0]
    sections = {}       # section -> {"ball": [(N, X, Y, poss, status)], "players": {(team, person): [(N, X, Y)]}}
    cur = None
    for ev, el in ET.iterparse(path, events=("start", "end")):
        if ev == "start" and el.tag == "FrameSet":
            cur = (el.get("GameSection"), el.get("TeamId"), el.get("PersonId"))
            sec = sections.setdefault(cur[0], {"ball": [], "players": {}})
            if cur[1] != "BALL":
                sec["players"].setdefault((cur[1], cur[2]), [])
        elif ev == "end" and el.tag == "Frame" and cur is not None:
            sec = sections[cur[0]]
            if cur[1] == "BALL":
                sec["ball"].append((int(el.get("N")), float(el.get("X")), float(el.get("Y")),
                                    int(el.get("BallPossession", 0)), int(el.get("BallStatus", 0))))
            else:
                sec["players"][(cur[1], cur[2])].append((int(el.get("N")), float(el.get("X")), float(el.get("Y"))))
            el.clear()
        elif ev == "end" and el.tag == "FrameSet":
            cur = None
            el.clear()
    return sections


REF_GAMES = ("J03WMX", "J03WN1")      # the two matches whose raw positions file has TeamId="referee" (the referee and two assistant referees)


def load_match(root, mid, min_spell=0, cache=True):
    root = os.path.expanduser(root)
    home, guest, pitch = _match_info(root, mid)
    L, W = pitch
    # With the environment variable MTAH_IDSSE_DROP_OFFICIALS=1 the slots of the match officials are dropped. Off by default, and then the match officials are counted in the away team
    drop_off = os.environ.get("MTAH_IDSSE_DROP_OFFICIALS") == "1"
    cache_path = os.path.join(root, "cache", "%s%s.npz" % (mid, "_noref" if drop_off and mid in REF_GAMES else ""))
    if cache and os.path.isfile(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        P, B, poss, status, period = d["P"], d["B"], d["poss"], d["status"], d["period"]
        slot_team = d["slot_team"]
    else:
        sections = _parse_positions(root, mid)
        order = [s for s in ("firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra") if s in sections] + \
                [s for s in sections if s not in ("firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra")]
        # player slots, home team first
        persons = sorted({k for s in sections.values() for k in s["players"] if not drop_off or k[0] in (home, guest)}, key=lambda k: (0 if k[0] == home else 1, k[1]))
        slot = {k: i for i, k in enumerate(persons)}
        slot_team = np.array([1 if k[0] == home else 2 for k in persons], np.int8)
        rows = []
        for si, s in enumerate(order):
            ball = sorted(sections[s]["ball"])
            n0, n1 = ball[0][0], ball[-1][0]
            rows.append((s, si + 1, n0, n1))
        T = sum(n1 - n0 + 1 for _, _, n0, n1 in rows)
        P = np.full((T, len(persons), 2), np.nan, np.float32)
        B = np.full((T, 2), np.nan, np.float32)
        poss = np.zeros(T, np.int8)
        status = np.zeros(T, np.int8)
        period = np.zeros(T, int)
        off = 0
        for s, pi, n0, n1 in rows:
            n = n1 - n0 + 1
            period[off:off + n] = pi
            for N, X, Y, pz, st in sections[s]["ball"]:
                i = off + N - n0
                B[i] = (X + L / 2, Y + W / 2)
                poss[i], status[i] = pz, st
            for k, fr in sections[s]["players"].items():
                if k not in slot:
                    continue
                j = slot[k]
                for N, X, Y in fr:
                    if n0 <= N <= n1:
                        P[off + N - n0, j] = (X + L / 2, Y + W / 2)
            off += n
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            np.savez_compressed(cache_path, P=P, B=B, poss=poss, status=status, period=period, slot_team=slot_team)
    T = len(B)
    graded = period > 0
    # which of BallPossession 1 and 2 is the home team, checked against the nearest player
    d = np.linalg.norm(P - B[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    nt = np.where(np.isinf(d.min(1)), 0, slot_team[d.argmin(1)])
    ok = (status == 1) & (poss > 0) & (nt > 0)
    agree = float((nt[ok] == poss[ok]).mean()) if ok.any() else 0.5
    flip = agree < 0.5
    gt = np.where(status == 1, np.where(flip, 3 - poss, poss), 3).astype(np.int8)
    gt[~graded] = 0
    gt[(status == 1) & (poss == 0)] = 0
    # The environment variable MTAH_IDSSE_LABEL_SHIFT=k (frames) changes the training labels only (min_spell>0); changes of possession within live ball move k frames earlier, and the boundaries between dead and live ball stay put.
    # The reason is that, while the ball is close to a player, the labels agree best with the team of the nearest player when shifted by 10 to 30 frames depending on the match, so the IDSSE possession field lags the tracking, typically by 0.4 to 0.6 s; the Metrica sample data show no such lag.
    lab_shift = int(os.environ.get("MTAH_IDSSE_LABEL_SHIFT", "0"))
    if min_spell > 0 and lab_shift > 0:
        live = np.isin(gt, (1, 2))
        edges = np.flatnonzero(np.diff(np.r_[0, live.astype(np.int8), 0]))
        for a, b in zip(edges[::2], edges[1::2]):   # shift inside each continuous live-ball segment, clipped at the segment end; it does not cross dead ball and creates no fragments
            j = np.minimum(np.arange(a, b) + lab_shift, b - 1)
            gt[a:b] = gt[j]
    # The environment variable MTAH_IDSSE_CHAIN=early or isolated (off by default) changes the training labels only (min_spell>0) and runs before the merging of short spells.
    # A run in which the two teams alternate within live ball is treated as a whole; a short spell is one shorter than MTAH_IDSSE_CHAIN_FR frames (default 50, 2 s) whose neighbours on both sides are spells of the opponent.
    #   isolated  a single short spell between two long spells (A long, B short, A long) is merged back into A; a run with several short spells is left alone
    #   early     in a run (A long, then several alternating short spells, then a long spell), if the first and last team are the same the whole run is merged back into A; otherwise possession changes only once, and the opponent has it from its first short spell
    chain = os.environ.get("MTAH_IDSSE_CHAIN", "")
    if min_spell > 0 and chain in ("early", "isolated"):
        cfr = int(os.environ.get("MTAH_IDSSE_CHAIN_FR", "50"))
        b_ = np.flatnonzero(np.diff(np.r_[-99, gt, -99]))
        rr = [[int(i), int(j), int(gt[i])] for i, j in zip(b_[:-1], b_[1:])]
        k = 0
        while k < len(rr):
            if rr[k][2] not in (1, 2) or (rr[k][1] - rr[k][0] < cfr and k > 0):
                k += 1
                continue
            m = k + 1                                   # from the long spell (or the start) rr[k], count the adjacent alternating short spells that follow
            while m < len(rr) and rr[m][2] in (1, 2) and rr[m][0] == rr[m - 1][1] and rr[m][2] != rr[m - 1][2] and rr[m][1] - rr[m][0] < cfr:
                m += 1
            shorts = list(range(k + 1, m))
            nxt = m if (m < len(rr) and rr[m][2] in (1, 2) and rr[m][0] == rr[m - 1][1] and rr[m][2] != rr[m - 1][2]) else None
            if shorts and nxt is not None:
                a_team, end_team = rr[k][2], rr[nxt][2]
                if chain == "isolated":
                    if len(shorts) == 1 and a_team == end_team:
                        gt[rr[shorts[0]][0]:rr[shorts[0]][1]] = a_team
                elif a_team == end_team:
                    for q in shorts:
                        gt[rr[q][0]:rr[q][1]] = a_team
                else:
                    first_b = shorts[0] if rr[shorts[0]][2] == end_team else (shorts[1] if len(shorts) > 1 else nxt)
                    for q in shorts:
                        gt[rr[q][0]:rr[q][1]] = a_team if q < first_b else end_team
            k = m if m > k + 1 else k + 1
    if min_spell > 0:
        for p in np.unique(period):
            if p == 0:
                continue
            idx = np.flatnonzero(period == p)
            w = gt[idx]
            br = np.flatnonzero(w[1:] != w[:-1]) + 1
            bounds = np.r_[0, br, len(w)]
            for i in range(len(bounds) - 1):
                a, b = bounds[i], bounds[i + 1]
                if w[a] in (1, 2) and b - a < min_spell and i > 0 and w[bounds[i - 1]] in (1, 2):
                    w[a:b] = w[bounds[i - 1]]
            gt[idx] = w
    blocks = {int(p): (int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1])) for p in np.unique(period) if p > 0}
    M = {"fps": FPS, "pitch": pitch, "t_ms": (np.arange(T) * (1000.0 / FPS)).astype(np.int64), "period": period,
         "graded": graded, "alive": graded.copy(), "ball_xy": B.astype(np.float64), "ball_visible": ~np.isnan(B[:, 0]),
         "ball_speed": np.full(T, np.nan), "ball_z": np.full(T, np.nan), "has_ball_z": False,
         "player_xy": P, "slot_team": slot_team, "tags": [], "blocks": blocks, "team_names": ("home", "away"),
         "game": "idsse_%s" % mid, "possession_agree": agree, "possession_flipped": bool(flip)}
    return M, gt
