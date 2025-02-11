#!/bin/bash

# Required arguments.
ANOM='/workspace/'
BASE='/workspace/'
OUT='/workspace/'

# Optional arguments.
NUM='50'

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/eval_ad.py \
										--anomaly=$ANOM \
										--baseline=$BASE \
										--out_path=$OUT \
										--num_samp=$NUM
