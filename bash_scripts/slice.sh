#!/bin/bash

# Required arguments.
IN="/workspace/"
OUT="/workspace/"

# Optional arguments.
MASK="/workspace/"

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python python_scripts/slice.py \
										--in_dir=$IN \
										--out_dir=$OUT \
										--mask_dir=$MASK \
