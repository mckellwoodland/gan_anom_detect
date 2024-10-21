#!/bin/bash

# Set arguments.
DSET1 = ""
DSET2 = ""
BACKBONE = ""
OUTPATH = ""
BATCH_SIZE = ""

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python Pytorch-StudioGAN/src/evaluate.py \
                                                                                -metrics fid \
                                                                                --dset1 $DSET1 \
                                                                                --dset2 $DSET2 \
                                                                                --post_resizer clean \
                                                                                --eval_backbone $BACKBONE \
                                                                                --out_path $OUTPATH \
                                                                                --batch_size $BATCH_SIZE
