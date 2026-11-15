#!/bin/sh
# Regenerates the submitted file "Rondo Lab.csv" from the organisers' bundle.
# Usage  sh reproduce.sh <path to the bundle>
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python reproduce.py --data_dir "$1" --model models/model.joblib --seq_dir models/seq_weights --w_hgb 0.2 \
  --boundary_model models/boundary_model.joblib --seq_head_dir models/seq_weights \
  --seq_setp_clip --seq_shape_feats --seq_comove_feats --seq_pset_feats 6 --seq_head_same_feats \
  --bundle models/decode_bundle.joblib --pre_kickoff_dead --out "Rondo Lab.csv"
# md5 of the output with Python itself, so that it works without md5sum or md5 on the PATH
python -c "import hashlib; print(hashlib.md5(open('Rondo Lab.csv', 'rb').read()).hexdigest(), ' Rondo Lab.csv')"
