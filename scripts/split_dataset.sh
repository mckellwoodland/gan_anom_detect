#!/bin/bash

# Set arguments.
IN_DIR="/workspace/images/splits/chestxray14_2/class0/"
OUT_DIR1="/workspace/images/splits/chestxray14_1_2/class0/"
OUT_DIR2="/workspace/images/splits/chestxray14_2_2/class0/"

# Run script.
docker run -it --rm -v $(pwd):/workspace gan_anom_detect python /workspace/scripts/split_dataset.py \
									--in_dir=$IN_DIR \
									--out_dir1=$OUT_DIR1 \
									--out_dir2=$OUT_DIR2
