#!/bin/bash

# Required arugments.
MODEL_PKL="/workspace/"
IN_DIR="/workspace/"
OUT_DIR="/workspace/"

# Optional arguments.
SAVE_VIDEO="False"

# Run script.
docker run --rm -v $(pwd):/workspace sg2ada python /workspace/stylegan2-ada-pytorch/projector.py \
									--network $MODEL_PKL \
									--target $IN_DIR \
									--save-video $SAVE_VIDEO \
									--outdir $OUT_DIR
