#!/bin/bash

# Set arguments.
INPUT="/workspace/"
OUTPUT="/workspace/"
KERNEL_SIZE="(5,5)"
MEAN="0."
SIGMA="0."
TYPE="n"
VAR="0.01"

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python /workspace/scripts/add_noise.py \
								-i $INPUT \
								-o $OUTPUT \
								-k $KERNEL_SIZE \
								-m $MEAN \
								-s $SIGMA \
								-t $TYPE \
								-v $VAR
