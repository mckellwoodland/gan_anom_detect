#!/bin/bash

# Required arguments.
IN="/workspace/"
OUT="/workspace/"

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python python_scripts/png_conversion.py \
								--in_dir=$IN \
								--out_dir=$OUT
