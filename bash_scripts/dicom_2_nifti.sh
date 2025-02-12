#!/bin/bash

# Arguments.
IN="/workspace/"
OUT="/workspace/"

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python python_scripts/dicom_2_nifti.py \
       										--in_dir=$IN \
										--out_dir=$OUT
