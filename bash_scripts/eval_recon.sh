#!/bin/bash

# Required arguments.
ORIG_DIR="/workspace/"
RECON_DIR="/workspace/"
OUT_PATH="/workspace/"

# Optional arguments.
DIST="MSE"
REG="full"
PSIZE="32"
PPATH="None"

# Run script.
docker run -it --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/eval_recon.py \
											--orig_dir $ORIG_DIR \
											--recon_dir $RECON_DIR \
											--out_path $OUT_PATH \
											--distance $DIST \
											--region $REG \
											--patch_size $PSIZE \
											--patch_path $PPATH
