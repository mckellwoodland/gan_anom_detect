#!/bin/bash

# Required arguments.
IN="/workspace/"
OUT="/workspace/"

# Optional arguments.
KERNEL_SIZE="(11,11)"
MEAN="0."
SIGMA="0."
TYPE="n"
VAR="0.01"

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/add_noise.py \
								-i $IN \
								-o $OUT \
								-k $KERNEL_SIZE \
								-m $MEAN \
								-s $SIGMA \
								-t $TYPE \
								-v $VAR
