#!/bin/bash

# Set arguments.
DSET1="/workspace/"
DSET2="/workspace/"
BACKBONE="_torch"
OUTPATH="/workspace/"
BATCH_SIZE="64"

# Run script.
docker run -it --rm -v $(pwd):/workspace alex4727/experiment:pytorch113_cuda116 python /workspace/PyTorch-StudioGAN/src/evaluate.py \
                                                                                -metrics fid \
                                                                                --dset1 $DSET1 \
                                                                                --dset2 $DSET2 \
                                                                                --post_resizer clean \
                                                                                --eval_backbone $BACKBONE \
                                                                                --out_path $OUTPATH \
                                                                                --batch_size $BATCH_SIZE
