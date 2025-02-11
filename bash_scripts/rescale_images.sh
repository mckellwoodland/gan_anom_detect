#!/bin/bash

# Required arguments.
IN="/workspace/images/original/chestxray14/github_testing/"
OUT="/workspace/images/original/chestxray14/github_testing512/"

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/rescale_images.py \
								-i $IN \
								-o $OUT
