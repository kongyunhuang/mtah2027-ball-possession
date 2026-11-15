"""
external_stats, summary statistics of public data that back the parameter ranges of the randomised degradation (reads public data only, never any competition data)
============================
The degradation parameters of the training data are set from external data, physical common sense or the literature only. This script measures a few kinds of summary numbers on public sources,
so that every parameter range in degrade_random.py has an external justification.
  SkillCorner open data (Australian A-League 2024-25, automatic tracking from broadcast video, MIT licence, 20 matches). To our knowledge it is the only public dataset tracked automatically from a single video feed,
    and an undetected ball is simply missing (adapters/skillcorner keeps the ball only where is_detected is true). Over the frames that have player data (the camera view is on the pitch) the script measures
    the share of frames with the ball visible (all, live ball, dead ball), the duration of ball gaps, the ball speed at the first and last frame of a gap, the distance from the ball to the nearest player in visible frames and the 1 s ball displacement in dead-ball frames;
    from the raw jsonl it measures the number of detected (is_detected) players per frame, the duration of the runs in which a player is not detected and the share of the pitch area covered by the camera view.
  IDSSE (DFL optical tracking, 7 matches) and the Metrica sample data (3 matches). The script measures the share of frames with the ball visible, the 1 s ball displacement in dead-ball frames,
    the p90 of the second difference of the ball position (raw and after moving averages of several widths) and the share of frames with fewer than 20 players on the pitch.
Only summary numbers (shares, quantiles) are printed and saved. No frame-level data are printed.

Inputs  public data in ~/Data/alt_training/{idsse,metrica_sample} and the SkillCorner open data in $MTAH_SC_ROOT (default ~/Data/alt_training/skillcorner_opendata)
Outputs  summary on screen and the JSON file given by --out (default external_stats.json); needs matplotlib for the camera view polygon

Usage
  python -u external_stats.py --procs 8

Last modified 2026-11-15
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtah_v3 import IDSSE_ROOT, SC_ROOT, smooth_ball_track  # noqa: E402
from possession.degrade import visibility_stats  # noqa: E402

T0 = time.time()
CORNERS = ("top_left", "bottom_left", "bottom_right", "top_right")
SC_CACHE = os.environ.get("MTAH_SC_NOCACHE") != "1"   # with --sc_no_cache the raw files are read directly and no parse cache is used (set through an environment variable so that the worker processes see it too)


def dead_ball_motion(M, gt, fps):
    """Over dead-ball frames (gt 3) where the ball is visible both now and 1 s earlier, the median 1 s ball displacement and the share of 'static' frames (under 0.5 m)"""
    b = np.asarray(M["ball_xy"], np.float64)
    lag = int(round(fps))
    vis = ~np.isnan(b[:, 0])
    m = np.zeros(len(b), bool)
    m[lag:] = vis[lag:] & vis[:-lag] & (gt[lag:] == 3) & M["graded"][lag:]
    if m.sum() < 100:
        return {"dead_disp1s_med": None, "dead_static_frac": None, "n_dead_vis": int(m.sum())}
    disp = np.zeros(len(b))
    disp[lag:] = np.linalg.norm(b[lag:] - b[:-lag], axis=1)
    return {"dead_disp1s_med": float(np.median(disp[m])), "dead_static_frac": float((disp[m] < 0.5).mean()), "n_dead_vis": int(m.sum())}


def accel_p90(b, fps):
    """p90 of the second difference of the ball position (metres per second squared) where three consecutive frames are all visible"""
    b = np.asarray(b, np.float64)
    a = np.linalg.norm(b[2:] - 2 * b[1:-1] + b[:-2], axis=1) * fps * fps
    a = a[~np.isnan(a)]
    return float(np.percentile(a, 90)) if len(a) else None


def lt20(M):
    P = np.asarray(M["player_xy"], np.float32)
    n = (~np.isnan(P[:, :, 0])).sum(1)
    return float((n[M["graded"]] < 20).mean())


def basic(M, gt):
    g = M["graded"]
    vis = ~np.isnan(np.asarray(M["ball_xy"])[:, 0])
    alive, dead = g & ((gt == 1) | (gt == 2)), g & (gt == 3)
    out = {"vis_all": float(vis[g].mean()), "vis_alive": float(vis[alive].mean()) if alive.any() else None,
           "vis_dead": float(vis[dead].mean()) if dead.any() else None, "dead_frac": float(dead[g].mean())}
    out.update(visibility_stats(M))
    out.update(dead_ball_motion(M, gt, M["fps"]))
    return out


def sc_raw_detection(mid):
    """SkillCorner raw 10 fps jsonl. Number of detected players per frame, duration of the runs in which a player is not detected, share of the pitch area covered by the camera view"""
    d = os.path.join(os.path.expanduser(SC_ROOT), "matches", mid)
    from matplotlib.path import Path
    mj = json.load(open(os.path.join(d, "%s_match.json" % mid)))
    L, W = float(mj["pitch_length"]), float(mj["pitch_width"])
    gx, gy = np.meshgrid(np.arange(-L / 2 + 1, L / 2, 2.0), np.arange(-W / 2 + 1, W / 2, 2.0))
    grid = np.c_[gx.ravel(), gy.ravel()]        # origin at the pitch centre, as in the coordinate convention of the SkillCorner open data README
    n_det, cover = [], []
    runs = []
    cur = {}
    for line in open(os.path.join(d, "%s_tracking_extrapolated.jsonl" % mid)):
        r = json.loads(line)
        pl = r.get("player_data") or []
        if not pl:
            for k in list(cur):
                runs.append(cur.pop(k))
            continue
        det = 0
        seen = set()
        for q in pl:
            pid = q.get("player_id")
            seen.add(pid)
            if q.get("is_detected"):
                det += 1
                if pid in cur:
                    runs.append(cur.pop(pid))
            else:
                cur[pid] = cur.get(pid, 0) + 1
        for k in [k for k in cur if k not in seen]:
            runs.append(cur.pop(k))
        n_det.append(det)
        poly = r.get("image_corners_projection") or {}
        if len(n_det) % 10 == 0 and all(poly.get("%s_%s" % (c, k)) is not None for c in ("x", "y") for k in CORNERS):
            # projection of the four image corners onto the pitch plane (image_corners_projection in the SkillCorner open data README); one frame in 10 is used, and the share of the points of a 2 m pitch grid that fall inside the view is counted
            path = Path([(poly["x_" + k], poly["y_" + k]) for k in CORNERS])
            cover.append(float(path.contains_points(grid).mean()))
    n_det = np.array(n_det)
    runs = np.array(runs, float) / 10.0
    return {"det_players_med": float(np.median(n_det)), "det_lt20_frac": float((n_det < 20).mean()),
            "det_lt11_frac": float((n_det < 11).mean()),
            "undet_run_med_s": float(np.median(runs)) if len(runs) else None, "undet_run_p90_s": float(np.percentile(runs, 90)) if len(runs) else None,
            "view_cover_med": float(np.median(cover)) if cover else None}


def job(spec):
    kind, key = spec
    t = time.time()
    if kind == "sc":
        from possession.adapters.skillcorner import load_match
        M, gt = load_match(SC_ROOT, key, cache=SC_CACHE)
        out = basic(M, gt)
        try:
            out.update(sc_raw_detection(key))
        except Exception as e:                     # if the raw file format differs from what is expected, only these items are missing
            out["raw_err"] = repr(e)[:200]
    elif kind == "idsse":
        from possession.adapters.idsse import load_match
        M, gt = load_match(IDSSE_ROOT, key, min_spell=0)
        out = basic(M, gt)
        out["lt20_players"] = lt20(M)
        out["accel_p90_raw"] = accel_p90(M["ball_xy"], M["fps"])
        for w in (5, 9, 15, 21, 25):
            out["accel_p90_ma%d" % w] = accel_p90(smooth_ball_track(M["ball_xy"], w), M["fps"])
    else:
        from possession.adapters.metrica_sample import load_game
        M, gt = load_game("~/Data/alt_training/metrica_sample", int(key))
        out = basic(M, gt)
        out["lt20_players"] = lt20(M)
        out["accel_p90_raw"] = accel_p90(M["ball_xy"], M["fps"])
    out["load_s"] = time.time() - t
    return kind, key, out


def summarize(rows, keys):
    s = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        if len(v):
            s[k] = {"min": float(v.min()), "p25": float(np.percentile(v, 25)), "med": float(np.median(v)), "p75": float(np.percentile(v, 75)), "max": float(v.max()), "n": int(len(v))}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default="external_stats.json")
    ap.add_argument("--sc_only", action="store_true", help="measure SkillCorner only (needs the raw jsonl files, not only the parse cache)")
    ap.add_argument("--skip_sc", action="store_true", help="skip SkillCorner")
    ap.add_argument("--sc_no_cache", action="store_true", help="read the raw SkillCorner files and ignore the parse cache")
    a = ap.parse_args()
    if a.sc_no_cache:
        os.environ["MTAH_SC_NOCACHE"] = "1"
    from possession.adapters.idsse import list_matches
    from possession.adapters.skillcorner import list_matches as sc_list
    specs = [] if a.skip_sc else [("sc", m) for m in sc_list(SC_ROOT)]
    if not a.sc_only:
        specs += [("idsse", m) for m in list_matches(IDSSE_ROOT)] + [("metrica", str(g)) for g in (1, 2, 3)]
    res = {"sc": {}, "idsse": {}, "metrica": {}}
    with Pool(a.procs) as pool:
        for n, (kind, key, out) in enumerate(pool.imap_unordered(job, specs)):
            res[kind][key] = out
            print("[progress] %d/%d  %s %s  elapsed %.0fs" % (n + 1, len(specs), kind, key, time.time() - T0), flush=True)
    keys = ["vis_all", "vis_alive", "vis_dead", "dead_frac", "dmin_med", "lt15", "gap_start_fast", "gap_end_fast", "gap_med_s", "gap_p90_s", "n_gaps",
            "dead_disp1s_med", "dead_static_frac", "det_players_med", "det_lt20_frac", "det_lt11_frac", "undet_run_med_s", "undet_run_p90_s", "view_cover_med",
            "lt20_players", "accel_p90_raw"] + ["accel_p90_ma%d" % w for w in (5, 9, 15, 21, 25)]
    summ = {k: summarize(list(v.values()), keys) for k, v in res.items() if v}
    for kind, s in summ.items():
        print("== %s (%d matches)" % (kind, len(res[kind])))
        for k, v in s.items():
            print("  %-18s median %8.3f  range %8.3f to %8.3f  quartiles %8.3f to %8.3f" % (k, v["med"], v["min"], v["max"], v["p25"], v["p75"]))
    errs = {k: r["raw_err"] for k, r in res["sc"].items() if "raw_err" in r}
    if errs:
        print("SkillCorner raw files, reading the detection flags failed for", list(errs.items())[:2])
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump({"per_match": res, "summary": summ}, open(out, "w"), indent=1)
    print("Written to %s, total time %.0fs" % (out, time.time() - T0))


if __name__ == "__main__":
    main()
