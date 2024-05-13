#!/bin/bash

# Required arguments.
ORIG_DATA=""
RECON_DIR=""

# Default arguments.
DISTANCE="MSE"
PATCH_SIZE="32"

# Run script.
docker run -it --rm -v $(pwd):/workspace gan_anom_detect python scripts/eval_recon_patch.py \
									-o ORIG_DATA \
									-r RECON_DIR \
									-d DISTANCE \
									-s PATCH_SIZE
