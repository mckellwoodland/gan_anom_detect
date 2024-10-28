#!/bin/bash

# Required arguments.
ORIG_DIR="/workspace/images/original/MIDRC/missing_lung/"
RECON_DIR="/workspace/images/reconstructed/MIDRC/missing_lung/"
OUT_PATH="/workspace/logs/dists/missing_lung_mse_full.csv"

# Optional arguments.
DIST="MSE"

# Run script.
docker run -it --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/eval_recon.py \
											--orig_dir $ORIG_DIR \
											--recon_dir $RECON_DIR \
											--out_path $OUT_PATH \
											--distance $DIST
