"""
ensemble_oof, average the out-of-fold probabilities of several mtah_v3 runs, decode again, and see how much the ensemble gains over the single models
============================
The runs must share the same held-out matches (same keys in loso_oof_proba.npz). For every match the per frame three-class
probabilities of the runs are averaged with weights, decoded with the three-state Viterbi, and scored against the raw labels
(optionally also against labels with spells shorter than N frames merged). The costs are searched again on a small grid.
Every single run is searched on the same grid, so the numbers are comparable. Only summary numbers are printed.

Inputs  --runs A,B,C (run names), --weights optional, --merged N
Outputs  mean and per match scores on screen, written to runs/_ensembles/<name>.json

Usage
  python -u ensemble_oof.py --runs tree_loso,seq_s0 --weights 0.2,0.8 --merged 25

Last modified 2026-11-15
"""
import argparse
import itertools
import json
import os
import sys
import warnings

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from rescore_oof import RUNS, raw_game  # noqa: E402
from mtah_v2 import to_hadx, viterbi3_tv  # noqa: E402
from possession.spell_metrics import codes_from_hadx, evaluate, merge_short_spells  # noqa: E402


def best_over_grid(P, games, labels, blocks, graded, grid):
    best = None
    for c in grid:
        per = {}
        for g in games:
            pred = to_hadx(viterbi3_tv(np.log(np.clip(P[g], 1e-6, 1)), blocks[g], *c))
            pred[~graded[g]] = 0
            per[g] = {lab: evaluate(codes_from_hadx(labels[lab][g]), codes_from_hadx(pred))["spell_f1"] for lab in labels}
        m = float(np.mean([per[g]["raw"] for g in games]))
        if best is None or m > best[0]:
            best = (m, c, per)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--merged", type=int, default=0)
    ap.add_argument("--grid_spec", default="4,8,16,32;16,32;12,24")
    a = ap.parse_args()
    runs = a.runs.split(",")
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(runs)
    oofs = [dict(np.load(os.path.join(RUNS, r, "loso_oof_proba.npz"))) for r in runs]
    games = sorted(oofs[0])
    for r, o in zip(runs, oofs):
        if sorted(o) != games:
            sys.exit("held-out matches differ, %s" % r)
    bundle = joblib.load(os.path.join(RUNS, runs[0], "model.joblib"))
    raw, blocks, graded = {}, {}, {}
    for g in games:
        raw[g], blocks[g], graded[g], _ = raw_game(g, bundle)
    labels = {"raw": raw}
    if a.merged > 0:
        labels["merged"] = {g: merge_short_spells(raw[g], blocks[g], a.merged) for g in games}
    grid = list(itertools.product(*[[float(x) for x in part.split(",")] for part in a.grid_spec.split(";")]))
    out = {"runs": runs, "weights": w, "games": games}
    rows = []
    for r, o in zip(runs, oofs):
        m, c, per = best_over_grid({g: o[g].astype(np.float64) for g in games}, games, labels, blocks, graded, grid)
        rows.append((r, m, c, per))
    ens = {g: sum(wi * o[g].astype(np.float64) for wi, o in zip(w, oofs)) / sum(w) for g in games}
    m, c, per = best_over_grid(ens, games, labels, blocks, graded, grid)
    rows.append(("ensemble", m, c, per))
    base = rows[0][3]
    for name, m, c, per in rows:
        mm = float(np.mean([per[g]["merged"] for g in games])) if a.merged > 0 else float("nan")
        print("%-30s raw labels %.4f  merged %d frames %.4f  costs %s" % (name, m, a.merged, mm, ",".join("%g" % x for x in c)))
        out[name] = {"mean_raw": m, "mean_merged": mm, "costs": list(c), "per_game": per}
    best_single = max(rows[:-1], key=lambda t: t[1])
    d = np.array([rows[-1][3][g]["raw"] - best_single[3][g]["raw"] for g in games])
    print("ensemble minus best single run (%s), per match difference on raw labels %s, mean %+.4f, positive in %d/%d matches" % (best_single[0], " ".join("%+.3f" % x for x in d), d.mean(), (d > 0).sum(), len(d)))
    os.makedirs(os.path.join(RUNS, "_ensembles"), exist_ok=True)
    fn = os.path.join(RUNS, "_ensembles", "_".join(r.replace("/", "-") for r in runs)[:120] + ".json")
    json.dump(out, open(fn, "w"), indent=1, ensure_ascii=False)
    print("written to " + fn)


if __name__ == "__main__":
    main()
