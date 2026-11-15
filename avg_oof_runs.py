"""
avg_oof_runs, average the out-of-fold probabilities of several runs with weights and write them as a new run directory that rescore_oof.py and the decoding scripts can read by run name
============================
Reads loso_oof_proba.npz in every run directory (and the extra benchmark files listed in --extra), takes the weighted average
match by match (the weights are normalised), and writes npz files with the same names under runs/_ensembles/<name>/.
model.joblib holds metadata only (copied from the first run, model set to None, plus an ensemble field that records the weights).
Run names are relative to $MTAH_RUNS and may contain a subdirectory. Only shapes and weights are printed, never per frame content.

Inputs  --runs A,B,C (run names, relative to runs/), --weights optional (equal weights by default), --name output directory name,
        --extra npz file names of the extra benchmarks, --geometric switches to the geometric mean (weighted mean of the log
        probabilities, then normalised per frame)
Outputs  runs/_ensembles/<name>/{loso_oof_proba.npz, loso_oof_proba_eval_*.npz, model.joblib}

Usage
  python -u avg_oof_runs.py --runs tree_loso,seq_s0,seq_s1,seq_s2,seq_s3,seq_p0,seq_p1,seq_p2,seq_p3 \
      --weights 0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 --name ens \
      --extra loso_oof_proba_eval_rnd0.npz,loso_oof_proba_eval_rnd1.npz,loso_oof_proba_eval_rnd2.npz,loso_oof_proba_eval_sw3_bs15.npz,loso_oof_proba_eval_sw0_bs0.npz,loso_oof_proba_eval_sw4_bs21.npz

Last modified 2026-11-15
"""
import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rescore_oof import RUNS  # noqa: E402


def run_dir(name):
    return os.path.join(RUNS, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--extra", default="loso_oof_proba_eval_sw0_bs0.npz,loso_oof_proba_eval_sw4_bs21.npz")
    ap.add_argument("--geometric", action="store_true", help="geometric mean, weighted mean of the log probabilities then normalised per frame (the default is the arithmetic mean)")
    a = ap.parse_args()
    runs = a.runs.split(",")
    w = np.array([float(x) for x in a.weights.split(",")], float) if a.weights else np.ones(len(runs))
    w = w / w.sum()
    out = os.path.join(RUNS, "_ensembles", a.name)
    os.makedirs(out, exist_ok=True)
    files = ["loso_oof_proba.npz"] + [f for f in a.extra.split(",") if f]
    for f in files:
        acc = None
        games = None
        for r, wi in zip(runs, w):
            p = os.path.join(run_dir(r), f)
            if not os.path.exists(p):
                sys.exit("missing file %s" % p)
            z = np.load(p)
            if games is None:
                games = sorted(z.files)
                acc = {g: np.zeros(z[g].shape, np.float64) for g in games}
            elif sorted(z.files) != games:
                sys.exit("%s has different held-out matches from the first run" % r)
            for g in games:
                assert z[g].shape == acc[g].shape, (r, g, z[g].shape, acc[g].shape)
                if a.geometric:
                    acc[g] += wi * np.log(np.clip(z[g].astype(np.float64), 1e-6, None))
                else:
                    acc[g] += wi * z[g].astype(np.float64)
        if a.geometric:
            for g in games:
                e = np.exp(acc[g] - acc[g].max(1, keepdims=True))
                acc[g] = e / e.sum(1, keepdims=True)
        np.savez_compressed(os.path.join(out, f), **{g: acc[g].astype(np.float32) for g in games})
        print("%s  %d matches  shape %s" % (f, len(games), acc[games[0]].shape), flush=True)
    meta = joblib.load(os.path.join(run_dir(runs[0]), "model.joblib"))
    meta["model"] = None
    meta["boundary_model"] = None
    meta["ensemble"] = {r: float(wi) for r, wi in zip(runs, w)}
    meta["ensemble_mean"] = "geometric" if a.geometric else "arithmetic"
    joblib.dump(meta, os.path.join(out, "model.joblib"))
    print("written to %s  weights %s" % (out, meta["ensemble"]), flush=True)


if __name__ == "__main__":
    main()
