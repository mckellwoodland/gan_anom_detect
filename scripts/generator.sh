#~/bin/bash

# Set arguments here
MODEL_PKL=""
SEEDS=""
OUTPUT_DIR=""

stylegan2-ada-pytorch/docker_run.sh python stylegan2-ada-pytorch/generate.py \
						--network $MODEL_PKL \
						--seeds $SEEDS \
						--outdir $OUTPUT_DIR
