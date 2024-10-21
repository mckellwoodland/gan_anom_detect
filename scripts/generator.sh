#~/bin/bash

# Set arguments here
MODEL_PKL=""
OUTPUT_DIR=""

stylegan2-ada-pytorch/docker_run.sh python stylegan2-ada-pytorch/generate.py \
						--network $MODEL_PKL \
						--outdir $OUTPUT_DIR
