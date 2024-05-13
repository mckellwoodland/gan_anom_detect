#!/bin/bash

# Set arguments.
FNAME=""

# Run script.
docker run -it --rm -v $(pwd):/workspace gan_anom_detect python scripts/find_best_fid.py \
									-f $FNAME
