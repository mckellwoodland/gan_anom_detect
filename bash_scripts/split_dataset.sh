#!/bin/bash

# Required arguments.
IN_DIR="/workspace/"
OUT_DIR1="/workspace/"
OUT_DIR2="/workspace/"

# Run script.
docker run -it --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/split_dataset.py \
												--in_dir=$IN_DIR \
												--out_dir1=$OUT_DIR1 \
												--out_dir2=$OUT_DIR2
