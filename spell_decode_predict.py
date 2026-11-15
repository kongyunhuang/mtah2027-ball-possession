"""
spell_decode_predict, apply the candidate segment scoring decoder to BAR-ATH and write the CSV in submission format
============================
Same candidates, segment features and semi-Markov dynamic programme as spell_decode.py, with two differences.
  1 The scorer is trained on all public matches of a run, with the out-of-fold probabilities for the segment features and the ground truth of all public matches as targets (nothing is held out). By default one scorer is trained for each of the IoU targets 0.3, 0.5, 0.7 and the probabilities are averaged.
  2 The candidate boundaries and segment features of BAR-ATH use the probabilities of the full-data model of the run (proba_bar_ath.npy in the run directory) and the BAR-ATH tracking.
    The touch thresholds follow the noise of the benchmark (same as spell_decode.default_touch_set; clean for a run with noise 0.01 or with eval_realistic).
θ and γ can be set in two ways. With --from_eval (the evaluation JSON written by spell_decode.py for the same run) the pair with the highest mean spellF1 of the leave-one-match-out decoding over all public matches is taken,
otherwise --theta and --gamma are used (defaults 0.3 and 0, the control without selection of the evaluation).
The output is a frame,label CSV (199,319 rows, X outside the graded frames) that passes spell_metrics.validate; a JSON of the same name next to it records the arguments, the md5 and summary numbers of the predictions.
There are three subcommands, and reproduction only needs predict.
  train    trains the scorer on the public matches (--label_min_spell makes the targets use IDSSE labels with short spells merged) and saves it together with θ, γ, kmax, the touch threshold set, the IoU targets,
           the IDSSE merge length in frames and the Viterbi cost grid as decode_bundle.joblib in the run directory;
           it accepts the extra benchmark options of spell_decode.py (--extra_benches, --score_benches, --report_benches; a benchmark may be a randomised benchmark rnd{k}) and its change-point head options (--switch_run and related);
  predict  reads only decode_bundle.joblib, proba_bar_ath.npy of the run directory and the BAR-ATH tracking, and writes the CSV (no public training data needed);
  all      runs both steps.
With --bundle_out, train saves to that path and leaves decode_bundle.joblib of the run directory untouched; predict reads it with --bundle.
The public 5 minute ground truth is not read (adapters.mtah2027.load is called without official_repo); only summary numbers of the predictions are printed (class shares, number of possession spells, mean spell length).
Everything is deterministic, two runs with the same arguments should give byte-identical files.

Inputs  --run run name (loso_oof_proba.npz, model.joblib, metrics.json, proba_bar_ath.npy under $MTAH_RUNS/<run>/, default ~/Data/more_than_a_hack_2027/runs),
        the BAR-ATH tracking (~/Data/more_than_a_hack_2027/data), the public source data (to train the scorer)
Outputs  the CSV at --out_csv and a .json of the same name

Usage
  OMP_NUM_THREADS=1 python -u spell_decode_predict.py train --run ens --label_min_spell 0 --from_eval ens_decode.json --scorer max_iter=500 \
      --thetas 0.4,0.45,0.5,0.55,0.6,0.7 --gammas 0.1,0.15,0.2,0.3,0.4 --extra_benches rnd1,rnd2 --score_benches \
      --report_benches sw3_bs15,sw0_bs0,sw4_bs21 --switch_run bnd_loso,seq_avg --switch_main_tag rnd0 --bundle_out models/decode_bundle.joblib
  reproduce.py runs the predict step on the competition match; the all subcommand runs both steps and takes the arguments of both

Last modified 2026-11-15
"""
import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import spell_decode as SD  # noqa: E402
from mtah_v2 import viterbi3  # noqa: E402
from possession.adapters.mtah2027 import GRADED_BLOCKS, N_FRAMES, SECOND_HALF_KICKOFF, load  # noqa: E402
from possession.spell_metrics import codes_from_hadx, spells, valid_runs, validate  # noqa: E402

RUNS = SD.RUNS
DATA_DIR = "~/Data/more_than_a_hack_2027/data"
T0 = time.time()


def log(msg):
    print("[%5.0fs] %s" % (time.time() - T0, msg), flush=True)


def prep(P, M, touch_set, kmax):
    """Candidate boundaries, segment features and mean dead-ball probability of every segment for one match (no ground truth needed)"""
    logp = np.log(np.clip(P, 1e-6, 1))
    blocks = {int(k): (int(v[0]), int(v[1])) for k, v in M["blocks"].items()}
    graded = np.asarray(M["graded"], bool)
    vis = np.asarray(M["ball_visible"], bool)
    decs = [viterbi3(logp, blocks, *c) for c in SD.COST_GRID]
    sw_fr, sw_tm, tt = SD.switch_touches(M, touch_set)
    cands = {}
    for h, (bA, bB) in blocks.items():
        pts = SD.build_candidates(bA, bB, decs, vis, sw_fr, extra=SD.extra_boundaries(P, tt, bA, bB))
        cands[h] = SD.merge_close(pts, sw_fr, sw_tm, bA, bB)
    seg = SD.segment_table(P, decs, vis, tt, cands, blocks, kmax)
    mD = {h: seg[h][2][:, SD.FEAT_NAMES.index("m_D")].astype(np.float64) for h in blocks}
    return {"P": P, "blocks": blocks, "graded": graded, "cands": cands, "seg": seg, "mD": mD, "n_switch_touch": int(len(sw_fr))}


def pick_theta_gamma(a):
    if not a.from_eval:
        return a.theta, a.gamma, "fixed values (--theta, --gamma)"
    ev = json.load(open(os.path.expanduser(a.from_eval)))
    for k, want in (("run", a.run), ("kmax", a.kmax)):
        if ev.get(k) != want:
            raise SystemExit("%s of the evaluation JSON is %s, which differs from %s of this call" % (k, ev.get(k), want))
    if [float(x) for x in ev.get("target_ious", [])] != [float(x) for x in a.ious.split(",")]:
        raise SystemExit("IoU targets %s of the evaluation JSON differ from %s of this call" % (ev.get("target_ious"), a.ious))
    if int(ev.get("label_min_spell_idsse") or 0) != int(a.label_min_spell):
        raise SystemExit("IDSSE short spell merge %s of the evaluation JSON differs from %s of this call" % (ev.get("label_min_spell_idsse"), a.label_min_spell))
    # the scorer parameters and the negative sampling used in the evaluation must match those used to train this bundle (both steps receive them separately, this is the safety check)
    sp = SD.scorer_params(a.scorer)
    if "scorer_params" in ev and {k: float(v) for k, v in ev["scorer_params"].items()} != {k: float(v) for k, v in sp.items()}:
        raise SystemExit("scorer parameters %s of the evaluation JSON differ from %s of this call" % (ev["scorer_params"], sp))
    if "neg_frac" in ev and float(ev["neg_frac"]) != float(a.neg_frac):
        raise SystemExit("negative sampling %s of the evaluation JSON differs from %s of this call" % (ev["neg_frac"], a.neg_frac))
    if ev.get("feat_set", "v1") != a.feat_set:
        raise SystemExit("feature set %s of the evaluation JSON differs from %s of this call" % (ev.get("feat_set", "v1"), a.feat_set))
    if int(ev.get("scorer_seeds", 1)) != int(a.scorer_seeds):
        raise SystemExit("number of scorer seeds %s of the evaluation JSON differs from %s of this call" % (ev.get("scorer_seeds", 1), a.scorer_seeds))
    if dict(ev.get("src_weight", {})) != SD.src_weight_map(a.src_weight):
        raise SystemExit("source weights %s of the evaluation JSON differ from %s of this call" % (ev.get("src_weight", {}), SD.src_weight_map(a.src_weight)))
    if (ev.get("extra_benches") or "") != (a.extra_benches or ""):
        raise SystemExit("extra benchmarks %s of the evaluation JSON differ from %s of this call" % (ev.get("extra_benches") or "none", a.extra_benches or "none"))
    if bool(ev.get("score_benches", False)) != bool(a.score_benches):
        raise SystemExit("score_benches %s of the evaluation JSON differs from %s of this call" % (ev.get("score_benches", False), a.score_benches))
    if (ev.get("cand_extra") or "") != (a.cand_extra or ""):
        raise SystemExit("extra candidate boundaries %s of the evaluation JSON differ from %s of this call" % (ev.get("cand_extra") or "none", a.cand_extra or "none"))
    gm = ev["grid_mean"]
    if a.thetas or a.gammas:                                 # the grid used in the evaluation must match the one given here
        want = {(float(t), float(g)) for t in a.thetas.split(",") for g in a.gammas.split(",")}
        have = {tuple(float(x) for x in k.split(",")) for k in gm}
        if want != have:
            raise SystemExit("the θγ grid of the evaluation JSON (%d cells) differs from the --thetas/--gammas given here (%d cells)" % (len(have), len(want)))
    key = max(sorted(gm), key=lambda k: gm[k])
    th, ga = (float(x) for x in key.split(","))
    return th, ga, "highest mean spellF1 over all public matches in the evaluation JSON (%.4f)" % gm[key]


def spell_summary(pred, graded):
    pc = codes_from_hadx(pred)
    runs = valid_runs(codes_from_hadx(np.where(graded, 1, 0)))
    sp = spells(pc, runs)
    g = graded
    return {"frac_H": float((pred[g] == 1).mean()), "frac_A": float((pred[g] == 2).mean()), "frac_D": float((pred[g] == 3).mean()),
            "n_spells": len(sp), "mean_spell_s": float(np.mean([b - a for a, b, _ in sp]) / SD.FPS) if sp else 0.0}


def cmd_train(a):
    target_ious = [float(x) for x in a.ious.split(",")]
    d = os.path.join(RUNS, a.run)
    bundle = joblib.load(os.path.join(d, "model.joblib"))
    oof = dict(np.load(os.path.join(d, "loso_oof_proba.npz")))
    games = list(oof)
    touch_set = a.touch_set or SD.default_touch_set(bundle)
    SD.LABEL_MIN_SPELL = int(a.label_min_spell)
    SD.FEAT_SET = a.feat_set
    SD.CAND_EXTRA = a.cand_extra
    SD.SWITCH_TAU, SD.SWITCH_HW = a.switch_tau, a.switch_hw
    sw_oof = [dict(np.load(os.path.join(RUNS, r, "loso_oof_boundary%s.npz" % ("_eval_" + a.switch_main_tag if a.switch_main_tag else "")))) for r in a.switch_run.split(",")] if a.switch_run else None   # change-point heads; several heads work as in spell_decode.py
    src_w = SD.src_weight_map(a.src_weight)
    theta, gamma, how = pick_theta_gamma(a)
    sp = SD.scorer_params(a.scorer)
    log("run %s, scorer trained on %d public matches, touch thresholds %s, kmax %d, IoU targets %s, IDSSE short spells merged below %d frames, θ %.2f γ %.2f (%s), scorer %s, feature set %s" % (
        a.run, len(games), touch_set, a.kmax, target_ious, a.label_min_spell, theta, gamma, how, sp, SD.FEAT_SET))
    data = {}
    for g in games:
        M, gt = SD.load_game_full(g, bundle)
        SD.SWITCH_CUR = SD.pick_switch(sw_oof, g)
        gd = prep(oof[g].astype(np.float64), M, touch_set, a.kmax)
        gd["tgt"] = {h: (SD.spell_iou_targets(gt, gd["seg"][h][4], gd["seg"][h][5], 1), SD.spell_iou_targets(gt, gd["seg"][h][4], gd["seg"][h][5], 2))
                     for h in gd["blocks"]}
        data[g] = gd
        log("%s loaded, %d candidate segments" % (g, sum(len(gd["seg"][h][0]) for h in gd["blocks"])))
    # --extra_benches, the out-of-fold probabilities of other benchmarks of the same run also give candidate segments, which are only added to the training rows of the scorer, as in spell_decode.py
    data_x = []
    for tag in [t for t in a.extra_benches.split(",") if t]:
        bx = SD.bench_bundle(bundle, tag)                    # sw{wander}_bs{smooth} or rnd{k}
        oof_x = dict(np.load(os.path.join(d, "loso_oof_proba_eval_%s.npz" % tag)))
        sw_x = [dict(np.load(os.path.join(RUNS, r, "loso_oof_boundary_eval_%s.npz" % tag))) for r in a.switch_run.split(",")] if a.switch_run else None
        dx = {}
        for g in games:
            M, gt = SD.load_game_full(g, bx)
            assert oof_x[g].shape[0] == len(gt), "%s %s, the number of frames of the out-of-fold probabilities does not match the rebuilt degradation" % (tag, g)
            SD.SWITCH_CUR = SD.pick_switch(sw_x, g)
            gd = prep(oof_x[g].astype(np.float64), M, touch_set, a.kmax)
            gd["tgt"] = {h: (SD.spell_iou_targets(gt, gd["seg"][h][4], gd["seg"][h][5], 1), SD.spell_iou_targets(gt, gd["seg"][h][4], gd["seg"][h][5], 2))
                         for h in gd["blocks"]}
            dx[g] = gd
        log("extra benchmark %s loaded (scorer training only), %d candidate segments" % (tag, sum(len(dx[g]["seg"][h][0]) for g in games for h in dx[g]["blocks"])))
        data_x.append(dx)
    # train the scorer on all public matches, sampling as in the evaluation script (fixed seeds, in match and block order)
    rngs = [np.random.default_rng(0)] + [np.random.default_rng(100 + sd) for sd in range(1, a.scorer_seeds)]   # same as in spell_decode.py, seed 0 for the first scorer
    clfs = []
    for thr in target_ious:
      for sd in range(a.scorer_seeds):
        rng = rngs[sd]
        Xs, ys, ws = [], [], []
        for g in games:
          for dd in [data] + data_x:
            for h in dd[g]["blocks"]:
                _, _, fH, fA, _, _ = dd[g]["seg"][h]
                for f, t in ((fH, dd[g]["tgt"][h][0]), (fA, dd[g]["tgt"][h][1])):
                    y = (t >= thr)
                    keep = y | (rng.random(len(y)) < a.neg_frac)
                    Xs.append(f[keep]); ys.append(y[keep]); ws.append(np.where(y[keep], 1.0, 1.0 / a.neg_frac) * SD.src_weight_of(g, src_w))
        X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
        clf = HistGradientBoostingClassifier(random_state=sd, **sp)
        clf.fit(X, y, sample_weight=w)
        clfs.append(clf)
        log("scorer IoU>=%.1f seed %d trained, %d rows, %d positives" % (thr, sd, len(y), int(y.sum())))
    import sklearn
    db = {"clfs": clfs, "feat_names": list(SD.FEAT_NAMES), "theta": theta, "gamma": gamma, "theta_gamma_from": how, "kmax": a.kmax,
          "touch_set": touch_set, "target_ious": target_ious, "cost_grid": [list(c) for c in SD.COST_GRID], "neg_frac": a.neg_frac, "scorer_params": sp, "feat_set": SD.FEAT_SET, "scorer_seeds": int(a.scorer_seeds), "src_weight": src_w, "cand_extra": a.cand_extra,
          "run": a.run, "train_games": games, "label_min_spell_idsse": int(a.label_min_spell),
          "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}}
    if a.extra_benches:                                      # recorded only when given
        db["extra_benches"] = a.extra_benches
    if bundle.get("rand_main") or a.score_benches:           # randomised benchmark setup, also recorded only when used
        db["main_bench"] = SD.bundle_main_tag(bundle)
        db["score_benches"] = bool(a.score_benches)
    if a.switch_run:                                         # recorded only when used; predict then reads proba_boundary_bar_ath.npy from the run directory
        db["switch"] = {"run": a.switch_run, "main_tag": a.switch_main_tag, "tau": a.switch_tau, "hw": a.switch_hw}
    path = os.path.expanduser(a.bundle_out) if a.bundle_out else os.path.join(d, "decode_bundle.joblib")
    joblib.dump(db, path)
    log("scorer and parameters saved to %s" % path)
    return path


def cmd_predict(a):
    d = os.path.join(RUNS, a.run)
    db = joblib.load(os.path.expanduser(a.bundle) if a.bundle else os.path.join(d, "decode_bundle.joblib"))
    if [list(c) for c in SD.COST_GRID] != db["cost_grid"]:
        raise SystemExit("the Viterbi cost grid in the code differs from the one in decode_bundle, the candidates would change")
    Mb = load({"data_dir": DATA_DIR})
    Mb["blocks"] = dict(GRADED_BLOCKS)
    SD.FEAT_SET = db.get("feat_set", "v1")   # the segment features follow the feature set recorded in the bundle
    SD.CAND_EXTRA = db.get("cand_extra", "") or ""   # the extra candidate boundaries also follow the bundle
    Pb = np.load(os.path.join(d, "proba_bar_ath.npy")).astype(np.float64)
    assert len(Pb) == N_FRAMES == len(Mb["graded"]), (len(Pb), N_FRAMES, len(Mb["graded"]))
    if db.get("switch"):                                     # change-point heads, the BAR-ATH change-point probabilities come from the full-data boundary model and sit in the run directory
        # the file of the first head is proba_boundary_bar_ath.npy, that of head i (from 1) is proba_boundary_bar_ath_h<i>.npy
        n_heads = len(db["switch"]["run"].split(","))
        arrs = [np.load(os.path.join(d, "proba_boundary_bar_ath%s.npy" % ("_h%d" % i if i else ""))).astype(np.float64) for i in range(n_heads)]
        assert all(len(x) == N_FRAMES for x in arrs)
        SD.SWITCH_CUR = arrs[0] if n_heads == 1 else arrs
        SD.SWITCH_TAU, SD.SWITCH_HW = db["switch"]["tau"], db["switch"]["hw"]
    else:
        SD.SWITCH_CUR = None
    gb = prep(Pb, Mb, db["touch_set"], db["kmax"])
    if list(SD.FEAT_NAMES) != db["feat_names"]:
        raise SystemExit("the segment feature columns differ from those in decode_bundle")
    theta, gamma, kmax = db["theta"], db["gamma"], db["kmax"]
    log("decode_bundle (θ %.2f γ %.2f kmax %d, touch thresholds %s, %d scorers, feature set %s)" % (theta, gamma, kmax, db["touch_set"], len(db["clfs"]), SD.FEAT_SET))
    pH, pA = {}, {}
    for h in gb["blocks"]:
        _, _, fH, fA, _, _ = gb["seg"][h]
        pH[h] = np.mean([c.predict_proba(fH)[:, 1] for c in db["clfs"]], axis=0)
        pA[h] = np.mean([c.predict_proba(fA)[:, 1] for c in db["clfs"]], axis=0)
    pred = SD.decode_game(gb, pH, pA, theta, gamma, kmax)
    n_pre = None
    if getattr(a, "pre_kickoff_dead", False):                # the official FAQ.md states that the second half is graded from frame 110808 while the kick-off is at frame 113742, and that the 2934 frames in between are all D; they are written as D here, no ground truth is read
        s2 = GRADED_BLOCKS[2][0]
        pred = np.array(pred, copy=True)
        n_pre = int((pred[s2:SECOND_HALF_KICKOFF] != 3).sum())
        pred[s2:SECOND_HALF_KICKOFF] = 3
        log("frames %d to %d before the second-half kick-off written as D, %d frames changed" % (s2, SECOND_HALF_KICKOFF - 1, n_pre))
    labels = np.array(["X", "H", "A", "D"])[pred]
    reason = validate(np.arange(N_FRAMES), labels, N_FRAMES)
    if reason is not None:
        raise SystemExit("format validation failed, %s" % reason)
    out_csv = os.path.expanduser(a.out_csv)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)   # when --out is a bare file name, dirname is an empty string and makedirs would fail
    text = "frame,label\n" + "".join("%d,%s\n" % (i, l) for i, l in enumerate(labels))
    open(out_csv, "w", newline="\n").write(text)   # newline="\n" keeps the bytes identical on every platform
    md5 = hashlib.md5(text.encode()).hexdigest()
    # only summary numbers of the predictions are printed; the number of spells is compared with the Viterbi submission file of the same run (no ground truth involved)
    S = spell_summary(pred, gb["graded"])
    base_path = os.path.join(d, a.run + ".csv")
    S_base = None
    if os.path.isfile(base_path):
        bl = np.loadtxt(base_path, delimiter=",", skiprows=1, dtype=str)[:, 1]
        bp = np.select([bl == "H", bl == "A", bl == "D"], [1, 2, 3], 0).astype(np.int8)
        S_base = spell_summary(bp, gb["graded"])
    log("BAR-ATH candidate segments %d, team-switch touches %d" % (sum(len(gb["seg"][h][0]) for h in gb["blocks"]), gb["n_switch_touch"]))
    print("predicted (graded frames)  H %.3f  A %.3f  D %.3f  possession spells %d  mean %.1f s" % (S["frac_H"], S["frac_A"], S["frac_D"], S["n_spells"], S["mean_spell_s"]))
    if S_base:
        print("Viterbi of the same run    H %.3f  A %.3f  D %.3f  possession spells %d  mean %.1f s" % (S_base["frac_H"], S_base["frac_A"], S_base["frac_D"], S_base["n_spells"], S_base["mean_spell_s"]))
    import sklearn
    meta = {k: db.get(k) for k in ("kmax", "neg_frac", "target_ious", "touch_set", "theta", "gamma", "train_games", "label_min_spell_idsse")}
    meta.update({"md5": md5, "summary": S, "summary_viterbi_same_run": S_base, "bundle_versions": db["versions"],
                 "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}})
    if n_pre is not None:
        meta["pre_kickoff_dead"] = {"frames": [GRADED_BLOCKS[2][0], SECOND_HALF_KICKOFF - 1], "changed": n_pre}
    json.dump(meta, open(os.path.splitext(out_csv)[0] + ".json", "w"), indent=1, ensure_ascii=False)
    log("written to %s (md5 %s), format validation passed" % (out_csv, md5))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "predict", "all"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        if name in ("train", "all"):
            p.add_argument("--kmax", type=int, default=30)
            p.add_argument("--neg_frac", type=float, default=0.25)
            p.add_argument("--ious", default="0.3,0.5,0.7")
            p.add_argument("--touch_set", default=None)
            p.add_argument("--from_eval", default=None)
            p.add_argument("--theta", type=float, default=0.3)
            p.add_argument("--gamma", type=float, default=0.0)
            p.add_argument("--label_min_spell", type=int, default=0, help="the scorer targets use IDSSE labels in which possession spells shorter than this many frames are merged, 0 for the raw labels; stored in decode_bundle")
            p.add_argument("--bundle_out", default=None, help="alternative path for the scorer and parameters, default decode_bundle.joblib in the run directory (which is overwritten)")
            p.add_argument("--scorer", default="", help="scorer hyperparameter overrides as a comma separated k=v list, same as --scorer of spell_decode.py, default SCORER_DEFAULT")
            p.add_argument("--feat_set", default="v1", choices=["v1", "v2", "v3"], help="segment feature set, same as --feat_set of spell_decode.py, must match the evaluation JSON; stored in the bundle")
            p.add_argument("--src_weight", default="", help="source weights for the scorer training samples, for example metrica=3, same as in spell_decode.py, must match the evaluation JSON; stored in the bundle")
            p.add_argument("--cand_extra", default="", choices=["", "touch", "argmax", "both"], help="one more kind of candidate boundary, same as in spell_decode.py, must match the evaluation JSON; stored in the bundle")
            p.add_argument("--scorer_seeds", type=int, default=1, help="number of scorers with different random seeds per IoU target, same as in spell_decode.py, must match the evaluation JSON; stored in the bundle")
            p.add_argument("--thetas", default="", help="θ grid as a comma separated list, accepted only so that the same extra arguments can be passed as to spell_decode.py; when given, the grid of the evaluation JSON is checked against it, and θγ themselves are still picked from grid_mean of the evaluation JSON")
            p.add_argument("--gammas", default="", help="γ grid, see --thetas")
            p.add_argument("--extra_benches", default="", help="candidate segments of other benchmarks of the same run also enter the scorer training, same as in spell_decode.py, must match the evaluation JSON")
            p.add_argument("--score_benches", action="store_true", help="same as in spell_decode.py, must match the evaluation JSON; only checked here, θγ are still picked from grid_mean of the evaluation JSON (which is then already the mean over the scored benchmarks)")
            p.add_argument("--switch_run", default="", help="run names of the change-point probability heads, same as in spell_decode.py, must match the evaluation JSON; stored in the bundle")
            p.add_argument("--switch_main_tag", default="", help="same as in spell_decode.py")
            p.add_argument("--switch_tau", type=float, default=0.5, help="same as in spell_decode.py")
            p.add_argument("--switch_hw", type=int, default=6, help="same as in spell_decode.py")
            p.add_argument("--report_benches", default="", help="accepted only so that the same extra arguments can be passed as to spell_decode.py, not used when training the bundle")
        if name in ("predict", "all"):
            p.add_argument("--out_csv", required=True)
            p.add_argument("--pre_kickoff_dead", action="store_true", help="write the frames from the start of second-half grading to the kick-off (110808 to 113741) as D, following the official FAQ.md; off by default")
            p.add_argument("--bundle", default=None, help="path of decode_bundle.joblib, default the one in the run directory")
    a = ap.parse_args()
    if a.cmd in ("train", "all"):
        path = cmd_train(a)
        if a.cmd == "all":
            a.bundle = path
    if a.cmd in ("predict", "all"):
        cmd_predict(a)


if __name__ == "__main__":
    main()
