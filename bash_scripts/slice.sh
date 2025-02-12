#!/bin/bash

# Required arguments.
IN="/workspace/images/original/computed_tomography/github_testing/cervix_nifti_windowed/"
OUT="/workspace/images/original/computed_tomography/github_testing/cervix_nifti_windowed_masked_sliced/"

# Optional arguments.
MASK="/workspace/images/original/computed_tomography/github_testing/cervix_masks/"

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python python_scripts/slice.py \
										--in_dir=$IN \
										--out_dir=$OUT \
										--mask_dir=$MASK \
