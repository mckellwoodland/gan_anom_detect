#!/bin/bash

# Required arguments.
ORIG_DATA="/workspace/"
RECON_DIR="/workspace/"

# Default arguments.
DISTANCE="MSE"
PATCH_SIZE="32"

# Run script.
docker run -it --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/eval_recon_patch.py \
												-o ORIG_DATA \
												-r RECON_DIR \
												-d DISTANCE \
												-s PATCH_SIZE
