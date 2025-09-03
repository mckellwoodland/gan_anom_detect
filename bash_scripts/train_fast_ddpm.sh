#!/bin/bash

# Required arguments.
CONFIG="Fast-DDPM/configs/liver_ct_linear.yml" # Path to the config file
DATASET="LIVER_CT" # Name of dataset
OUT_PTH="generative_modeling/diffusion_modeling/fast-ddpm/CT/" # Path for saving running related data
DOC="liver_ct"

# Optional arguments.
SCHED="uniform"
STEPS="100"

# Run script.
docker run --rm -v $(pwd):/workspace fastddpm python /workspace/Fast-DDPM/fast_ddpm_main.py \
							--config $CONFIG \
							--dataset $DATASET \
							--exp $OUT_PTH \
							--doc $DOC \
							--scheduler_type $SCHED \
							--timesteps $STEPS
