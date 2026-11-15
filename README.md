# Rondo Lab, More than a Hack 2027, Phase 1

Frame-level ball possession (H, A, D, X) for the BAR-ATH match, computed from the tactical ATD of the organisers' bundle only. Nothing from the bundle is in this repository. The models are trained on public data only, and the parameters that the degradation of the training data draws at random come from ranges set from public sources only. Label-free summary statistics of the competition tracking entered the work early in the project in three places, all listed in Section 9 of the technical document, namely the three fixed degradation settings (used for the three fixed evaluation benches and, early on, to degrade the training copies), the choice and form of the tracking defects that the degradation simulates, including the constants that are not drawn at random, and the touch-detection thresholds for low-jitter tracking with their assignment radius and step back. The method is described in the technical document submitted with this repository, `Rondo Lab.pdf`.

## What is here

- `reproduce.py` regenerates the submitted file from the model files in `models/` in one command.
- `models/model.joblib` is the gradient-boosted tree classifier (scikit-learn HistGradientBoostingClassifier, 1500 iterations, learning rate 0.03, 63 leaves, at least 100 samples per leaf) over 282 per-frame features.
- `models/boundary_model.joblib` is the tree change model, a second HistGradientBoostingClassifier that gives for every frame the probability that a possession starts or ends within 6 frames, over 180 features.
- `models/seq_weights/` holds the 80 fold weights of the temporal convolutional network, 4 seeds times ten leave-one-match-out folds for each of two variants. The first variant reads 291 features, the 282 of the trees and 9 co-movement features; the second also reads 66 values that describe the six players nearest to the ball, through a small sub-network shared by the six. The class probabilities of the 80 networks are averaged with equal weights and combined with the tree probabilities with weights 0.8 and 0.2. `reproduce.py` reads the same files a second time for the fourth output of the network, its change probability (`--seq_head_dir models/seq_weights`).
- `models/decode_bundle.joblib` holds the segment scorers and the decoding parameters (theta 0.5, gamma 0.3, candidate segments of at most 30 boundary intervals).
- `possession/` is the feature code (`mtah_features.py`, with the time-window helpers in `features.py`), the touch detector (`touch_events.py`), the degradation functions (`degrade.py`), the scoring code (`spell_metrics.py`) and the readers for the competition bundle (`adapters/mtah2027.py`), for the two public training sources (`adapters/metrica_sample.py`, `adapters/idsse.py`) and for the SkillCorner open data (`adapters/skillcorner.py`).
- `degrade_random.py` draws the degradation of every training copy and of the three randomised benches at random from ranges set from public data only, with the degradation functions in `possession/degrade.py`; `external_stats.py` measures the statistics of the public sources (SkillCorner open data, IDSSE, Metrica sample) behind those ranges.
- `mtah_v3.py` (tree classifier and tree change model), `seq_model.py` (network), `spell_decode.py` and `spell_decode_predict.py` (segment decoder) are the training and evaluation entry points; `seq_predict.py` runs the network on the competition match; `mtah_v2.py` holds the Viterbi decoder; `rescore_oof.py`, `avg_oof_runs.py` and `ensemble_oof.py` score and combine held-out probabilities.

`mtah_v2.py` holds the three-state Viterbi decoder, the change-point labels and the tree fitting helpers used by the other scripts. The supported inputs are the competition bundle and the public sources `metrica` and `idsse`; `skillcorner` can also be read, as an optional training-only source, and the submitted models do not use it. In this repository the SkillCorner open data are read by `external_stats.py`, to measure the statistics of that source; put the folder of the SkillCorner open data repository that contains `matches/` under `~/Data/alt_training/skillcorner_opendata`, or point the environment variable `MTAH_SC_ROOT` to it. The check of the submitted pipeline on the 20 SkillCorner matches reported in the technical document was run with scripts that are not part of this repository; it took no part in computing the submitted file and served as a check on the last decisions, as the technical document describes. `external_stats.py` also needs `matplotlib` for the SkillCorner detection statistics (players detected per frame, runs of undetected players and camera view coverage); nothing else does.

## Reproduce the submitted file

1. Put the organisers' bundle under a directory of your choice, keeping its layout (`tactical/BAR-ATH_tactical_atd.xml`, `tactical/BAR-ATH_tactical_atd.txt`, and the rest).
2. Install the pinned dependencies in Python 3.10 (the submitted file was produced with Python 3.10.19; CPU only, no GPU is needed).

```
pip install -r requirements.txt
```

3. Run, from the repository root,

```
sh reproduce.sh <path to the bundle>
```

which sets four thread variables (below) and runs

```
python reproduce.py --data_dir <path to the bundle> --model models/model.joblib --seq_dir models/seq_weights --w_hgb 0.2 --boundary_model models/boundary_model.joblib --seq_head_dir models/seq_weights --seq_setp_clip --seq_shape_feats --seq_comove_feats --seq_pset_feats 6 --seq_head_same_feats --bundle models/decode_bundle.joblib --pre_kickoff_dead --out "Rondo Lab.csv"
```

`--pre_kickoff_dead` writes D on the 2,934 graded frames between the start of the second graded block (frame 110,808) and the second-half kick-off (frame 113,742), as the challenge FAQ states.

The output has 199,319 rows plus a header and is byte-identical to the submitted file on the library versions in `requirements.txt`; its md5 is `ee4b87654cd5cb809713f02a2c1972fc`. `reproduce.sh` runs the same command and takes the path to the bundle as its only argument; it sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to 1 before Python starts, and the network runs in one torch thread, so that the result is bitwise reproducible (running the python command directly, without these variables, gave the same file on an Apple M4 Pro). The same file was also obtained in new Python 3.10 virtual environments with only `requirements.txt` installed, on the Apple M4 Pro and on an Intel x86-64 machine under Windows 11 (there run the python command, since Windows has no `sh`). It takes 7 to 9 minutes of wall-clock time in one thread on an Apple M4 Pro. The md5 of every model file is in `MD5SUMS.txt`. `reproduce.py` also writes `Rondo Lab.json` next to the CSV (the arguments, the md5 and a summary of the predictions) and a parse cache of the tracking in `cache/` inside the bundle directory, so that directory must be writable.

## Retrain from scratch

Training uses ten public matches, three from the Metrica Sports sample data and seven from the IDSSE dataset, and selects every setting by leave-one-match-out on those ten matches. Put the Metrica sample data under `~/Data/alt_training/metrica_sample` (the three folders `Sample_Game_1`, `Sample_Game_2` and `Sample_Game_3` of the `data` folder of the Metrica repository) and the IDSSE matches under `~/Data/alt_training/idsse` (the `DFL_02_01_matchinformation` and `DFL_04_03_positions_raw_observed` XML files of the seven matches in one folder), as the two adapters expect. The steps that also predict the competition match read the bundle from `~/Data/more_than_a_hack_2027/data`. Run everything from the repository root with the settings below. The first three apply the corrections of the public data described in the technical document. The referee and the assistant referees are dropped from the IDSSE position files. The Metrica event chain starts a dead ball at goals, offsides, hand balls, shots out of play and the ends of the halves. After a ball out of play the dead ball starts at the end of that event, not earlier. `MTAH_RUNS` is where the run directories go.

In the commands below `--rand_degrade` makes every training copy draw its degradation from the public ranges, and `--rand_bench 3 --rand_main` builds the three randomised benches and selects on them. With these flags `--realistic` and `--noise` affect neither training nor any bench (the three fixed benches added by `--extra_eval` always use realistic ball gaps and 0.01 m of noise); `--noise 0.01` is stored with the run and makes the segment decoder use the touch thresholds for low-jitter tracking. The commands without `--no_bar` train on all ten matches and then predict the competition match. If a copy of the official challenge repository is at `~/Data/more_than_a_hack_2027/official_repo` (with `evaluation/gt/BAR-ATH_possession_gt_public.csv`), these commands also print the official metrics on the public five minutes of ground truth as a check of the pipeline and store them in `metrics.json`; no later step reads these numbers.

```
export MTAH_IDSSE_DROP_OFFICIALS=1 MTAH_METRICA_FIX_DEAD=1 MTAH_METRICA_FIX_OVERLAP=1 MTAH_RUNS=~/Data/more_than_a_hack_2027/runs
```

1. Tree classifier, leave-one-match-out probabilities on the three randomised and the three fixed benches, and the model trained on all ten matches (which also writes the probabilities for the competition match).

```
python mtah_v3.py --name tree_loso --sources metrica,idsse --seeds 1 --realistic --noise 0.01 --touch_set clean --touch_feats --gap_feats --idsse_min_spell 25 --no_bar --augment swap --hgb max_iter=1500,learning_rate=0.03,max_leaf_nodes=63,min_samples_leaf=100 --rand_bench 3 --rand_degrade --rand_main --extra_eval 3:15,0:0,4:21 --setp_feats --shape_feats
python mtah_v3.py --name tree_full --sources metrica,idsse --seeds 1 --realistic --noise 0.01 --touch_set clean --touch_feats --gap_feats --idsse_min_spell 25 --augment swap --hgb max_iter=1500,learning_rate=0.03,max_leaf_nodes=63,min_samples_leaf=100 --rand_degrade --setp_feats --shape_feats --skip_loso --costs 8,32,24
```

2. Tree change model, held out and on all ten matches. `mtah_v3.py` always trains the per-frame classifier as well, which is kept small here with `--hgb max_iter=50,learning_rate=0.08,max_leaf_nodes=31` because only the change model is kept from these runs; the hyperparameters of the change model are given by `--bhgb`.

```
python mtah_v3.py --name bnd_loso --sources metrica,idsse --seeds 1 --realistic --noise 0.01 --touch_set clean --touch_feats --gap_feats --idsse_min_spell 25 --no_bar --augment swap --rand_bench 3 --rand_degrade --rand_main --extra_eval 3:15,0:0,4:21 --hgb max_iter=50,learning_rate=0.08,max_leaf_nodes=31 --boundary --kgrid 0 --bhgb max_iter=1500,learning_rate=0.03,max_leaf_nodes=63,min_samples_leaf=100
python mtah_v3.py --name bnd_full --sources metrica,idsse --seeds 1 --realistic --noise 0.01 --touch_set clean --touch_feats --gap_feats --idsse_min_spell 25 --augment swap --rand_degrade --skip_loso --costs 8,32,24 --hgb max_iter=50,learning_rate=0.08,max_leaf_nodes=31 --boundary --kgrid 0 --boundary_keep --bhgb max_iter=1500,learning_rate=0.03,max_leaf_nodes=63,min_samples_leaf=100
```

3. Temporal convolutional network, two variants, one run per variant and seed, for the seeds 0 to 3 (45 to 70 minutes per run on the GPUs we used). Each run writes ten fold weights `seq_<match>.pt` with the fourth output, and the held-out class and change probabilities on all six benches. The second command trains the variant that also reads the six players nearest to the ball.

```
python seq_model.py --name seq_s<seed> --seed <seed> --sources metrica,idsse --realistic --noise 0.01 --touch_set clean --touch_feats --gap_feats --idsse_min_spell 25 --channels 128 --blocks 8 --window 4096 --batch 8 --augment swap --no_inner_val --steps 2500 --trans_weight 2 --trans_win 12 --ema 0.998 --rand_degrade 3 --rand_bench 3 --rand_main --extra_eval 3:15,0:0,4:21 --lazy_norm --slim_m --aux_win 6 --save_boundary --aux_boundary 1 --setp_clip --shape_feats --comove_feats
python seq_model.py --name seq_p<seed> --seed <seed> --sources metrica,idsse --realistic --noise 0.01 --touch_set clean --touch_feats --gap_feats --idsse_min_spell 25 --channels 128 --blocks 8 --window 4096 --batch 8 --augment swap --no_inner_val --steps 2500 --trans_weight 2 --trans_win 12 --ema 0.998 --rand_degrade 3 --rand_bench 3 --rand_main --extra_eval 3:15,0:0,4:21 --lazy_norm --slim_m --aux_win 6 --save_boundary --aux_boundary 1 --setp_clip --shape_feats --comove_feats --pset_feats 6 --pset_pool
```

4. Average the held-out class probabilities (trees 0.2, each of the eight runs an equal share of 0.8) into a run directory `ens`, and the change output of the eight runs into `seq_avg`.

```
python avg_oof_runs.py --runs tree_loso,seq_s0,seq_s1,seq_s2,seq_s3,seq_p0,seq_p1,seq_p2,seq_p3 --weights 0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 --name ens --extra loso_oof_proba_eval_rnd0.npz,loso_oof_proba_eval_rnd1.npz,loso_oof_proba_eval_rnd2.npz,loso_oof_proba_eval_sw3_bs15.npz,loso_oof_proba_eval_sw0_bs0.npz,loso_oof_proba_eval_sw4_bs21.npz
mkdir -p $MTAH_RUNS/ens && cp $MTAH_RUNS/_ensembles/ens/* $MTAH_RUNS/ens/ && echo '{"costs_used": "8,32,24"}' > $MTAH_RUNS/ens/metrics.json
python - <<'EOF'
import glob, os
import numpy as np
R = os.path.expanduser(os.environ["MTAH_RUNS"]); seeds = ["seq_s0", "seq_s1", "seq_s2", "seq_s3", "seq_p0", "seq_p1", "seq_p2", "seq_p3"]
os.makedirs(os.path.join(R, "seq_avg"), exist_ok=True)
for f in glob.glob(os.path.join(R, seeds[0], "loso_oof_boundary*.npz")):
    arrs = [np.load(os.path.join(R, s, os.path.basename(f))) for s in seeds]
    np.savez_compressed(os.path.join(R, "seq_avg", os.path.basename(f)), **{g: np.mean([a[g].astype(np.float64) for a in arrs], axis=0).astype(np.float32) for g in arrs[0].files})
EOF
```

5. Segment scorers and decoding parameters, evaluated by leave-one-match-out on the ten matches (theta and gamma chosen for each match on the other nine) and then trained on all ten. The second command writes the new bundle over the shipped `models/decode_bundle.joblib`; give `--bundle_out` another path to keep the shipped file.

```
python spell_decode.py --run ens --ious 0.3,0.5,0.7 --label_min_spell 0 --scorer max_iter=500 --thetas 0.4,0.45,0.5,0.55,0.6,0.7 --gammas 0.1,0.15,0.2,0.3,0.4 --extra_benches rnd1,rnd2 --score_benches --report_benches sw3_bs15,sw0_bs0,sw4_bs21 --switch_run bnd_loso,seq_avg --switch_main_tag rnd0 --save_pred ens_decode_pred.npz --out ens_decode.json
python spell_decode_predict.py train --run ens --label_min_spell 0 --from_eval ens_decode.json --scorer max_iter=500 --thetas 0.4,0.45,0.5,0.55,0.6,0.7 --gammas 0.1,0.15,0.2,0.3,0.4 --extra_benches rnd1,rnd2 --score_benches --report_benches sw3_bs15,sw0_bs0,sw4_bs21 --switch_run bnd_loso,seq_avg --switch_main_tag rnd0 --bundle_out models/decode_bundle.joblib
```

6. Copy `tree_full/model.joblib` and `bnd_full/boundary_model.joblib` into `models/`, and the fold weights of all eight runs into `models/seq_weights/` with the run in the file name (`seq_s0_metrica1.pt` for `seq_s0/seq_metrica1.pt`, and `seq_s4_` to `seq_s7_` for the four runs `seq_p0` to `seq_p3` of the second variant, as in the shipped folder), then run the command under "Reproduce the submitted file".

Retraining reproduces the method but not the exact network weights, because GPU training is not bitwise deterministic across devices; the submitted seeds were trained on an Apple GPU and on an NVIDIA laptop GPU, and their weights are the ones in `models/seq_weights/`.

The scripts keep switches for variants tested during development; the commands above are the submitted configuration.

## Training data

- Metrica Sports sample data, three matches with tracking and events, from `github.com/metrica-sports/sample-data`, whose README asks that the data be used responsibly and that the source be acknowledged in anything public. Frame-level possession labels are derived from the event stream by `possession/adapters/metrica_sample.py`.
- IDSSE, seven matches of German professional football with synchronised tracking, events and official frame-level possession and ball status, released under CC BY 4.0 with the authorisation of the DFL. Data on figshare, DOI 10.6084/m9.figshare.28196177. Please cite Bassek, M., Rein, R., Weber, H. and Memmert, D. An integrated dataset of spatiotemporal and event data in elite soccer. Scientific Data 12, 195 (2025), DOI 10.1038/s41597-025-04505-y.

The public tracking is cleaner than tracking produced automatically from video, so every training copy is degraded with parameters drawn at random from ranges set from public sources (`--rand_degrade`), and every setting is selected on three held-out benches degraded the same way (`--rand_bench 3 --rand_main`); `--extra_eval 3:15,0:0,4:21` adds three fixed benches that the technical document reports for comparison. The settings of these three benches were calibrated in the early part of the project on label-free summary statistics of the competition tracking. The choices made on them at that stage were kept (the hyperparameters of the trees and of the segment scorers, the training recipe of the network and the first six feature groups, 180 features), and the benches take no part in training the submitted models or in any later choice.

## Licence

The code and the model files in this repository are released under the MIT licence, see `LICENSE`. The training datasets keep their own licences, given above. The competition data are not part of this repository and may not be redistributed.
