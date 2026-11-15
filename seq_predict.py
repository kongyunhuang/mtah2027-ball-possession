"""
seq_predict, run all fold weights saved by the leave-one-match-out training of seq_model on BAR-ATH and average their probabilities, optionally averaging again with the probabilities of the per frame gradient boosted tree model
============================
Training runs on a GPU, inference runs on the CPU (single torch thread, two runs on the same machine are bit-identical). The features are built the same way as in reproduce.py
(build_features with the ball gap and touch features; the feature names are checked one by one against those stored in the weight files). Every fold weight file carries its own standardisation mean and standard deviation.

Inputs  --seq_dir directory with the fold weight files seq_*.pt, --data_dir the data/ directory of the competition data package, optional --hgb_proba (proba_bar_ath.npy of the tree model) and its weight --w_hgb
Outputs  --out the (T, 3) float32 probabilities of the sequence model averaged over all fold weights in --seq_dir; with --hgb_proba the weighted average is also written to --out_mix

Usage
  python -u seq_predict.py --seq_dir models/seq_weights --setp_clip --shape_feats --comove_feats --pset_feats 6 \
      --data_dir <path to the bundle> --out proba_bar_ath_seq.npy
  (reproduce.py calls seq_probabilities and seq_head_probabilities directly)

Last modified 2026-11-15
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from possession.adapters.mtah2027 import GRADED_BLOCKS, load  # noqa: E402
from possession.mtah_features import build_features  # noqa: E402


def seq_probabilities(seq_dir, M, X, names, chunk=32768):
    """Run every fold weight file seq_*.pt in seq_dir and average, returns (T, 3) float32. M must have blocks (both ends inclusive)"""
    import torch
    from seq_model import make_model, predict_game, standardize
    torch.set_num_threads(1)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(seq_dir), "seq_*.pt")))
    if not files:
        sys.exit("no weight files %s/seq_*.pt" % seq_dir)
    acc = np.zeros((len(X), 3), np.float64)
    dev = torch.device("cpu")
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        fn = list(ck["feature_names"])
        if fn != list(names)[:len(fn)]:                  # equal, or a prefix (one directory may hold weights with and without per-player features; the feature table is built for the larger set)
            sys.exit("feature names do not match the weights %s" % os.path.basename(f))
        ar = ck["arch"]
        model = make_model(ar["n_in"], ar["channels"], ar["blocks"], ar["kernel"], ar["dropout"], ar.get("arch", "tcn"), ar.get("rnn_layers", 2), ar.get("n_out", 3), **{k: ar[k] for k in ("xf_layers", "xf_heads", "xf_stride", "pset") if k in ar}).to(dev)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        Z = standardize(X if len(fn) == X.shape[1] else X[:, :len(fn)], ck["mu"], ck["sd"])
        acc += predict_game(model, Z, M, dev, chunk).astype(np.float64)
        print("  %s done" % os.path.basename(f), flush=True)
    return (acc / len(files)).astype(np.float32)


def seq_head_probabilities(seq_dir, M, X, names, chunk=32768):
    """Sigmoid of the 4th output (change point) of the sequence model, averaged over all fold weights in the directory, returns (T,) float32. The weights must have n_out 4"""
    import torch
    from seq_model import make_model, predict_game, standardize
    torch.set_num_threads(1)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(seq_dir), "seq_*.pt")))
    if not files:
        sys.exit("no weight files %s/seq_*.pt" % seq_dir)
    acc = np.zeros(len(X), np.float64)
    dev = torch.device("cpu")
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        fn = list(ck["feature_names"])
        if fn != list(names)[:len(fn)]:                  # equal or a prefix, as in seq_probabilities
            sys.exit("feature names do not match the weights %s" % os.path.basename(f))
        ar = ck["arch"]
        if ar.get("n_out", 3) < 4:
            sys.exit("the weights %s have no 4th output" % os.path.basename(f))
        model = make_model(ar["n_in"], ar["channels"], ar["blocks"], ar["kernel"], ar["dropout"], ar.get("arch", "tcn"), ar.get("rnn_layers", 2), ar["n_out"], **{k: ar[k] for k in ("xf_layers", "xf_heads", "xf_stride", "pset") if k in ar})
        model.load_state_dict(ck["state_dict"])
        model.eval()
        b = np.zeros(len(X), np.float32)
        predict_game(model, standardize(X if len(fn) == X.shape[1] else X[:, :len(fn)], ck["mu"], ck["sd"]), M, dev, chunk, b)
        acc += b.astype(np.float64)
        print("  %s 4th output done" % os.path.basename(f), flush=True)
    return (acc / len(files)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setp_clip", action="store_true", help="the weights were trained with the outlier-clipped positioning features (seq_model --setp_clip), the features are built the same way")
    ap.add_argument("--shape_feats", action="store_true", help="the weights were trained with the team shape behaviour features (seq_model --shape_feats), the features are built the same way")
    ap.add_argument("--comove_feats", action="store_true", help="the weights were trained with the player-ball co-movement features (seq_model --comove_feats), the features are built the same way")
    ap.add_argument("--pset_feats", type=int, default=0, help="give K when the directory contains weights with per-player features (seq_model --pset_feats K --pset_pool); the feature table gets K times 11 more columns and weights without per-player features only use the leading columns (default 0)")
    ap.add_argument("--head_only", action="store_true", help="only compute the 4th output (change-point head), --out stores the (T,) probabilities")
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--data_dir", default="~/Data/more_than_a_hack_2027/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hgb_proba", default=None)
    ap.add_argument("--w_hgb", type=float, default=0.2, help="weight of the tree probabilities in --out_mix (0.2 in the submitted file)")
    ap.add_argument("--out_mix", default=None)
    ap.add_argument("--touch_set", default="clean")
    ap.add_argument("--kin_feats", action="store_true", help="for standalone use set it as in training; when called through reproduce.py the features are built there from the metadata of the tree model")
    a = ap.parse_args()
    t0 = time.time()
    M = load({"data_dir": os.path.expanduser(a.data_dir)})
    M["blocks"] = dict(GRADED_BLOCKS)
    X, names = build_features(M, gap_feats=True, touch_feats=True, touch_set=a.touch_set, kin_feats=a.kin_feats, setp_feats=("clip" if a.setp_clip else False), shape_feats=a.shape_feats, comove_feats=a.comove_feats, pset_feats=a.pset_feats)
    print("features %s built, %.0fs elapsed" % (X.shape, time.time() - t0), flush=True)
    if a.head_only:
        H = seq_head_probabilities(a.seq_dir, M, X.astype(np.float32), names)
        np.save(os.path.expanduser(a.out), H)
        print("sequence head written to %s, %d frames, %.0fs elapsed" % (a.out, len(H), time.time() - t0), flush=True)
        return
    P = seq_probabilities(a.seq_dir, M, X.astype(np.float32), names)
    np.save(os.path.expanduser(a.out), P)
    g = M["graded"]
    print("sequence model probabilities written to %s, argmax shares on graded frames H %.3f A %.3f D %.3f, %.0fs elapsed" % (
        a.out, (P[g].argmax(1) == 0).mean(), (P[g].argmax(1) == 1).mean(), (P[g].argmax(1) == 2).mean(), time.time() - t0), flush=True)
    if a.hgb_proba:
        Ph = np.load(os.path.expanduser(a.hgb_proba)).astype(np.float64)
        assert Ph.shape == P.shape, (Ph.shape, P.shape)
        Pm = (a.w_hgb * Ph + (1 - a.w_hgb) * P.astype(np.float64)).astype(np.float32)
        np.save(os.path.expanduser(a.out_mix), Pm)
        print("ensemble probabilities (tree weight %.2f) written to %s, argmax shares on graded frames H %.3f A %.3f D %.3f" % (
            a.w_hgb, a.out_mix, (Pm[g].argmax(1) == 0).mean(), (Pm[g].argmax(1) == 1).mean(), (Pm[g].argmax(1) == 2).mean()), flush=True)


if __name__ == "__main__":
    main()
