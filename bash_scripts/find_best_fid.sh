#!/bin/bash

# Required arguments.
FNAME="/workspace/"

# Run script.
docker run -it --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/find_best_fid.py \
												-f $FNAME
