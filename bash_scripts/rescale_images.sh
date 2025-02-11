#!/bin/bash

# Required arguments.
IN="/workspace/"
OUT="/workspace/"

# Optional arguments.
RES="512"

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/rescale_images.py \
								-i $IN \
								-o $OUT \
								-r $RES
