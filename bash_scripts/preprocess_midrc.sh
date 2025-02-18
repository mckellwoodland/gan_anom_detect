#!/bin/bash

# Required arguments.
IN='/workspace/'
OUT='/workspace/'

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/preprocess_midrc.py \
											--in_dir=$IN \
											--out_dir=$OUT
