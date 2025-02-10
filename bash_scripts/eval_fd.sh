#!/bin/bash

# Required arguments.
DSET1="/workspace/"
DSET2="/workspace/"
BACKBONE="_torch"
OUTPATH="/workspace/"

# Default arguments.
BATCH_SIZE="64"

# Run script.
# studiogan can be changed to alex4727/experiment:pytorch113_cuda116 for different CUDA compatibility.
docker run -it --rm -v $(pwd):/workspace studiogan python /workspace/PyTorch-StudioGAN/src/evaluate.py \
                                                                                -metrics fid \
                                                                                --dset1 $DSET1 \
                                                                                --dset2 $DSET2 \
                                                                                --post_resizer clean \
                                                                                --eval_backbone $BACKBONE \
                                                                                --out_path $OUTPATH \
                                                                                --batch_size $BATCH_SIZE
