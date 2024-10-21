#~/bin/bash

# Set arguments here
MODEL_PKL=""
SEEDS="0-49999"
OUTPUT_DIR=""

stylegan2-ada-pytorch/docker_run.sh python stylegan2-ada-pytorch/generate.py \
						--network $MODEL_PKL \
						--seeds $SEEDS \
						--outdir $OUTPUT_DIR
