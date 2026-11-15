"""
reproduce, regenerates the submitted file from the competition bundle with the model files in models/
============================
One command, two steps. First the per frame tree classifier (model.joblib) computes H/A/D probabilities from the tactical camera tracking (ATD); with --seq_dir all fold weights
of the temporal convolutional network in that directory (80 in models/seq_weights) are run as well and averaged with the tree probabilities using --w_hgb. With --boundary_model and
--seq_head_dir the change-point probabilities of the tree change model and of the network are computed too. Then the candidate segment scoring decoder (decode_bundle.joblib)
decodes the probabilities into possession spells and writes a frame,label CSV.
No training data is read and the ground truth of the public window is not read. The network runs in one torch thread (seq_predict sets torch.set_num_threads(1)); the BLAS and OpenMP thread counts
are fixed by the environment variables that reproduce.sh exports before Python starts (OMP_NUM_THREADS is also set in main with setdefault, which only takes effect if it was not set and comes after numpy
is imported). With these settings two runs with the same arguments on the same machine give byte-identical files.

Inputs  --data_dir the directory of the competition bundle (contains tactical/), --model the per frame model, --bundle the decoder, --out the output CSV path, optional --seq_dir (weight directory of the sequence model) and --w_hgb;
        when the bundle uses change-point heads add --boundary_model (tree head) and --seq_head_dir (sequence head)
Outputs  the CSV (199,319 rows plus header) and a .json of the same name (arguments, md5, summary of the predictions)

Usage
  sh reproduce.sh <path to the bundle>
  (the full python command is in reproduce.sh and in the README)

Last modified 2026-11-15
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spell_decode_predict as SDP  # noqa: E402
from possession.adapters.mtah2027 import GRADED_BLOCKS, load  # noqa: E402
from possession.mtah_features import build_features  # noqa: E402


def build_inputs(bundle, data_dir):
    """Build the inputs of the per frame model from the competition data package, returns (M, X, names)"""
    M = load({"data_dir": data_dir})
    M["blocks"] = dict(GRADED_BLOCKS)
    X, names = build_features(M, gap_feats=bool(bundle.get("gap_feats", False)), touch_feats=bool(bundle.get("touch_feats", False)), touch_set=bundle.get("touch_set", "robust"), kin_feats=bool(bundle.get("kin_feats", False)),
                              setp_feats=bundle.get("setp_feats", False), shape_feats=bool(bundle.get("shape_feats", False)))
    if names != bundle["feature_names"]:
        sys.exit("feature names do not match the model")
    return M, X, names


def base_inputs(bundle, M, X, names):
    """When the per frame model has the extra positioning or team shape features, the change-point head and a sequence head trained without them still use the base features; when the model itself uses the base features the given arrays are returned"""
    if not (bundle.get("setp_feats", False) or bundle.get("shape_feats", False)):
        return X, names
    return build_features(M, gap_feats=bool(bundle.get("gap_feats", False)), touch_feats=bool(bundle.get("touch_feats", False)), touch_set=bundle.get("touch_set", "robust"), kin_feats=bool(bundle.get("kin_feats", False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="~/Data/more_than_a_hack_2027/data")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1, help="value given to OMP_NUM_THREADS when it is not already set (default 1); set the thread variables before Python starts, as reproduce.sh does, for bit-exact reproducibility")
    ap.add_argument("--seq_dir", default=None, help="directory of the fold weights of the sequence model (all seq_*.pt in it, 80 in models/seq_weights); when given, their mean probabilities are averaged with those of the per frame model using --w_hgb")
    ap.add_argument("--boundary_model", default=None, help="full-data model of the change-point head (boundary_model.joblib), required when the decoder bundle uses a change-point head")
    ap.add_argument("--seq_setp_clip", action="store_true", help="the weights in --seq_dir were trained with the outlier-clipped positioning features (seq_model --setp_clip), the sequence model features are built the same way")
    ap.add_argument("--seq_shape_feats", action="store_true", help="the weights in --seq_dir were trained with the team shape behaviour features (seq_model --shape_feats), the sequence model features are built the same way")
    ap.add_argument("--seq_comove_feats", action="store_true", help="the weights in --seq_dir were trained with the player-ball co-movement features (seq_model --comove_feats), the sequence model features are built the same way")
    ap.add_argument("--seq_pset_feats", type=int, default=0, help="give K when --seq_dir contains weights with per-player features (seq_model --pset_feats K --pset_pool); the feature table of the sequence model gets K times 11 more columns and weights without per-player features only use the leading columns (default 0)")
    ap.add_argument("--seq_head_same_feats", action="store_true", help="the sequence head weights were trained with the same features as --seq_dir, so the head also uses the sequence model features; without this flag the head uses the base features of the per frame model")
    ap.add_argument("--seq_head_dir", default=None, help="directory of the fold weights of the sequence head (seq_*.pt with a 4th output), required when the decoder bundle uses the tree head plus the sequence head")
    ap.add_argument("--w_hgb", type=float, default=0.2, help="weight of the per frame model in the ensemble, the sequence model gets 1 minus it; the submitted setting is 0.2 (chosen by a sweep on held-out matches)")
    ap.add_argument("--seq_tta", default="", choices=("", "swap"), help="with swap the sequence model (and a sequence head with the same features) is run once more on inputs with the home and away teams exchanged, the H and A columns are swapped back and the result is averaged with the normal pass (empty by default, not used for the submitted file)")
    ap.add_argument("--pre_kickoff_dead", action="store_true", help="the second half is graded from frame 110808 while the kick-off is at frame 113742; the official FAQ.md states that the 2934 frames in between are all D, and this switch writes them as D (off by default, used for the submitted file)")
    a = ap.parse_args()
    if a.seq_tta:
        assert a.seq_setp_clip or a.seq_shape_feats or a.seq_comove_feats, "--seq_tta is only supported when the sequence model uses its own feature set"
        assert (not a.seq_head_dir) or a.seq_head_same_feats, "with --seq_tta the sequence head must use the same features as the sequence model (--seq_head_same_feats)"
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    t0 = time.time()
    bundle = joblib.load(os.path.expanduser(a.model))
    M, X, names = build_inputs(bundle, a.data_dir)
    P = bundle["model"].predict_proba(X).astype(np.float32)
    print("per frame probabilities %s done, %.0fs elapsed" % (P.shape, time.time() - t0), flush=True)
    if a.seq_dir:
        from seq_predict import seq_probabilities
        if a.seq_setp_clip or a.seq_shape_feats or a.seq_comove_feats:   # the sequence model reads the positioning columns clipped to their physical range, the team shape columns and the 9 player-ball co-movement columns; the tree model reads unclipped positioning columns and no co-movement columns, so the sequence features are built separately
            Xs, names_s = build_features(M, gap_feats=bool(bundle.get("gap_feats", False)), touch_feats=bool(bundle.get("touch_feats", False)), touch_set=bundle.get("touch_set", "robust"), kin_feats=bool(bundle.get("kin_feats", False)), setp_feats=("clip" if a.seq_setp_clip else False), shape_feats=a.seq_shape_feats, comove_feats=a.seq_comove_feats, pset_feats=a.seq_pset_feats)
        else:
            Xs, names_s = base_inputs(bundle, M, X, names)
        Ps = seq_probabilities(a.seq_dir, M, Xs.astype(np.float32), names_s)
        if a.seq_tta == "swap":                          # the same exchange as the home/away swap augmentation used in training (mtah_v3._swap_teams), only the slot teams and the team names are swapped
            M_sw = dict(M)
            st_ = np.asarray(M["slot_team"])
            M_sw["slot_team"] = np.where(st_ == 1, 2, np.where(st_ == 2, 1, st_)).astype(st_.dtype)
            M_sw["team_names"] = tuple(reversed(M.get("team_names", ("home", "away"))))
            Xs_sw, names_sw = build_features(M_sw, gap_feats=bool(bundle.get("gap_feats", False)), touch_feats=bool(bundle.get("touch_feats", False)), touch_set=bundle.get("touch_set", "robust"), kin_feats=bool(bundle.get("kin_feats", False)), setp_feats=("clip" if a.seq_setp_clip else False), shape_feats=a.seq_shape_feats, comove_feats=a.seq_comove_feats, pset_feats=a.seq_pset_feats)
            assert names_sw == names_s
            Ps_sw = seq_probabilities(a.seq_dir, M_sw, Xs_sw.astype(np.float32), names_sw)[:, [1, 0, 2]]
            Ps = ((Ps.astype(np.float64) + Ps_sw.astype(np.float64)) / 2).astype(np.float32)
            print("sequence model pass with home and away exchanged done and averaged, %.0fs elapsed" % (time.time() - t0), flush=True)
        P = (a.w_hgb * P.astype(np.float64) + (1 - a.w_hgb) * Ps.astype(np.float64)).astype(np.float32)
        print("sequence model probabilities done and averaged with tree weight %.2f, %.0fs elapsed" % (a.w_hgb, time.time() - t0), flush=True)
    tmp = tempfile.mkdtemp(prefix="mtah_reproduce_")
    try:
        run = "reproduce"
        os.makedirs(os.path.join(tmp, run))
        np.save(os.path.join(tmp, run, "proba_bar_ath.npy"), P)
        if a.boundary_model:
            bm = joblib.load(os.path.expanduser(a.boundary_model))
            Xb, names_b = base_inputs(bundle, M, X, names) if list(bm["feature_names"]) != list(names) else (X, names)
            assert bm["feature_names"] == list(names_b), "the feature columns of the change-point head match neither the base features nor those of the per frame model"
            np.save(os.path.join(tmp, run, "proba_boundary_bar_ath.npy"), bm["boundary_model"].predict_proba(Xb)[:, 1].astype(np.float32))
            print("change-point probabilities done, %.0fs elapsed" % (time.time() - t0), flush=True)
        if a.seq_head_dir:
            from seq_predict import seq_head_probabilities
            Xh, names_h = (Xs, names_s) if a.seq_head_same_feats else base_inputs(bundle, M, X, names)
            Bh = seq_head_probabilities(a.seq_head_dir, M, Xh.astype(np.float32), names_h)
            if a.seq_tta == "swap":                      # change points carry no team, so the swapped pass needs no column swap and is averaged directly
                Bh = ((Bh.astype(np.float64) + seq_head_probabilities(a.seq_head_dir, M_sw, Xs_sw.astype(np.float32), names_sw).astype(np.float64)) / 2).astype(np.float32)
            np.save(os.path.join(tmp, run, "proba_boundary_bar_ath_h1.npy"), Bh)
            print("sequence head done, %.0fs elapsed" % (time.time() - t0), flush=True)
        SDP.RUNS = tmp
        SDP.DATA_DIR = a.data_dir
        SDP.cmd_predict(argparse.Namespace(run=run, bundle=os.path.expanduser(a.bundle), out_csv=os.path.expanduser(a.out), pre_kickoff_dead=a.pre_kickoff_dead))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("total time %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
