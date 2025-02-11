#!/bin/bash

# Required arguments.
IN="/workspace/"
OUT="/workspace/"

# Optional arguments.
WID="350."
LEV="50."

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/window_level.py \
											--in_dir $IN \
											--out_dir $OUT \
											--window_width $WID \
											--window_level $LEV
