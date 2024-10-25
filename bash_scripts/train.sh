#!/bin/bash

# Required arguments.
OUT_DIR=""
DATA=""

# Optional arguments.
GPUS="8"
MIRROR="1"
CFG="stylegan2"
RESUME="ffhq512"
FP32="1"
BETA0="0.9"

# Run script.
docker run --rm -v $(pwd):/workspace sg2ada python /workspace/stylegan2-ada-pytorch/train.py \
							--outdir $OUT_DIR \
							--gpus $GPUS \
							--data $DATA \
							--mirror $MIRROR \
							--cfg $CFG \
							--resume $RESUME \
							--fp32 $FP32 \
							--beta0 $BETA0
