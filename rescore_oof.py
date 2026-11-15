"""
rescore_oof, rescore the leave-one-match-out probabilities of mtah_v3 against the raw labels
============================
--min_spell_all of mtah_v3 and min_spell of IDSSE merge short spells in both the training and the evaluation labels, so the
leave-one-match-out scores of runs with different merge thresholds are not comparable. This script reads loso_oof_proba.npz in
the run directory (per frame probabilities of every held-out match), redoes the three-state Viterbi, scores every match against
the raw labels (no merging), and optionally also against labels merged with one common threshold for comparison.
Only summary numbers are printed.

Inputs  --run run name ($MTAH_RUNS/<run>/, default ~/Data/more_than_a_hack_2027/runs). The costs are read from model.joblib by default, --grid
        searches them again on the raw labels. --merged N also scores against labels with spells shorter than N frames merged
Outputs  per match and mean spellF1, macroF1, per frame accuracy split by ball visible and ball missing, and hit rate by
         spell length, written to rescore.json in the run directory

Usage
  python -u rescore_oof.py --run tree_loso
  python -u rescore_oof.py --run tree_loso --grid --merged 25

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
from mtah_v2 import to_hadx, viterbi3_tv  # noqa: E402
from mtah_v3 import IDSSE_ROOT  # noqa: E402
from possession.spell_metrics import binned_hit_rates, codes_from_hadx, evaluate, merge_short_spells  # noqa: E402

RUNS = os.path.expanduser(os.environ.get("MTAH_RUNS", "~/Data/more_than_a_hack_2027/runs"))   # can be redirected with the MTAH_RUNS environment variable


def raw_game(g, bundle):
    """Reload the raw labels and blocks of a held-out match by its name, and redo the degradation with the training settings
    (seed 0, noise, ball removal mode of the fixed degradation) to get the ball visible mask. Returns (gt_raw, blocks, graded, ball_visible)"""
    noise, realistic = bundle.get("noise", 0.3), bool(bundle.get("realistic", False))
    if bundle.get("eval_realistic"):
        noise, realistic = 0.01, True        # runs whose held-out matches were scored on the fixed realistic degradation; the ball visible mask is rebuilt with it
    if g.startswith("metrica"):
        from mtah_v3 import augment_metrica
        from possession.adapters.metrica_sample import load_game
        M, gt = load_game("~/Data/alt_training/metrica_sample", int(g[-1]))
        vis = augment_metrica(M, gt, 0, noise, 0.30, realistic)["ball_visible"]
    elif g.startswith("idsse_"):
        from mtah_v3 import degrade_like_wc
        from possession.adapters.idsse import load_match
        M, gt = load_match(IDSSE_ROOT, g[len("idsse_"):], min_spell=0)
        vis = degrade_like_wc(M, gt, 0, noise, realistic=realistic)["ball_visible"]
    elif g.startswith("skillcorner_"):
        from mtah_v3 import SC_ROOT, degrade_like_wc
        from possession.adapters.skillcorner import load_match as sc_load
        M, gt = sc_load(SC_ROOT, g[len("skillcorner_"):])
        vis = degrade_like_wc(M, gt, 0, noise, realistic=realistic, keep_missing=True)["ball_visible"]
    else:
        raise ValueError(g)
    return np.asarray(gt), M["blocks"], M["graded"], np.asarray(vis, bool)


def score(oof, games, labels, blocks, graded, costs, vis=None):
    per = {}
    for g in games:
        pred = to_hadx(viterbi3_tv(np.log(np.clip(oof[g], 1e-6, 1)), blocks[g], *costs))
        pred[~graded[g]] = 0
        G, Pc = codes_from_hadx(labels[g]), codes_from_hadx(pred)
        r = evaluate(G, Pc)
        per[g] = {"spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"], "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"],
                  "hit_by_length": {k: v[0] for k, v in binned_hit_rates(G, Pc).items()}}
        if vis is not None:
            lab = np.asarray(labels[g])
            m = graded[g] & (lab > 0)
            per[g]["acc_vis"] = float((pred[m & vis[g]] == lab[m & vis[g]]).mean())
            per[g]["acc_miss"] = float((pred[m & ~vis[g]] == lab[m & ~vis[g]]).mean())
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--grid", action="store_true", help="search the costs again on the raw labels")
    ap.add_argument("--grid_spec", default="4,8,16,32;16,32;12,24", help="grid for the search, the values of c_team;c_dead;c_alive")
    ap.add_argument("--merged", type=int, default=0, help="also score against labels with spells shorter than N frames merged")
    ap.add_argument("--oof_file", default="loso_oof_proba.npz", help="which held-out probability file of the run directory to read (the extra benchmarks of mtah_v3 --extra_eval are loso_oof_proba_eval_sw{sw}_bs{bs}.npz)")
    ap.add_argument("--out_tag", default="", help="suffix added to the output file name, such as _eval_sw0_bs0, so that the rescore result of the main benchmark is not overwritten")
    a = ap.parse_args()
    d = os.path.join(RUNS, a.run)
    bundle = joblib.load(os.path.join(d, "model.joblib"))
    oof = dict(np.load(os.path.join(d, a.oof_file)))
    games = list(oof.keys())
    raw, blocks, graded, vis = {}, {}, {}, {}
    for g in games:
        raw[g], blocks[g], graded[g], vis[g] = raw_game(g, bundle)
        oof[g] = oof[g].astype(np.float64)
    print("run %s, %d matches, probability file %s, costs stored in the model %s, trained with min_spell_all=%s idsse_min_spell=%s" % (
        a.run, len(games), a.oof_file, bundle["costs"], bundle.get("min_spell_all"), bundle.get("idsse_min_spell")))
    out = {"run": a.run, "games": games}
    if a.grid:
        grid = list(itertools.product(*[[float(x) for x in part.split(",")] for part in a.grid_spec.split(";")]))
        best = None
        for c in grid:
            per = score(oof, games, raw, blocks, graded, c, vis)
            m = float(np.mean([v["spell_f1"] for v in per.values()]))
            if best is None or m > best[0]:
                best = (m, c, per)
        costs, per_raw = best[1], best[2]
        print("costs found by the search on the raw labels %s" % (costs,))
    else:
        costs = tuple(bundle["costs"])
        per_raw = score(oof, games, raw, blocks, graded, costs, vis)
    out["costs"] = list(costs)

    def show(tag, per):
        ms, mm = float(np.mean([v["spell_f1"] for v in per.values()])), float(np.mean([v["macro_f1"] for v in per.values()]))
        bins_ = list(next(iter(per.values()))["hit_by_length"])
        hb = {k: float(np.nanmean([per[g]["hit_by_length"][k] for g in games])) for k in bins_}
        av = float(np.mean([per[g].get("acc_vis", np.nan) for g in games])); am = float(np.mean([per[g].get("acc_miss", np.nan) for g in games]))
        print("%s  mean spellF1 %.4f  macroF1 %.4f  per frame accuracy ball visible %.3f ball missing %.3f  hit rate by spell length %s" % (
            tag, ms, mm, av, am, " ".join("%s %.2f" % (k, hb[k]) for k in bins_)))
        for g in games:
            v = per[g]
            print("    %-16s spellF1 %.3f  macroF1 %.3f  true spells %4d  predicted spells %4d  visible %.3f missing %.3f  hit %s" % (
                g, v["spell_f1"], v["macro_f1"], v["n_gt"], v["n_pred"], v.get("acc_vis", np.nan), v.get("acc_miss", np.nan),
                " ".join("%.2f" % v["hit_by_length"][k] for k in bins_)))
        return {"mean_spell_f1": ms, "mean_macro_f1": mm, "mean_acc_vis": av, "mean_acc_miss": am, "mean_hit_by_length": hb, "per_game": per}

    out["raw"] = show("raw labels", per_raw)
    if a.merged > 0:
        lab = {g: merge_short_spells(raw[g], blocks[g], a.merged) for g in games}
        out["merged_%d" % a.merged] = show("spells under %d frames merged" % a.merged, score(oof, games, lab, blocks, graded, costs, vis))
    fn = ("rescore_grid" if a.grid else "rescore") + a.out_tag + ".json"
    with open(os.path.join(d, fn), "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("written to %s/%s" % (d, fn))


if __name__ == "__main__":
    main()
