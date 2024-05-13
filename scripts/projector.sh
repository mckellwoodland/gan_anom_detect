#!/bin/bash

# Set arguments here
MODEL_PKL=""
INPUT_DIR=""
OUTPUT_DIR=""

stylegan2-ada-pytorch/docker_run.sh python stylegan2-ada-pytorch/projector.py \
						--network MODEL_PKL \
						--target INPUT_DIR \
						--save-video False \
						--outdir OUTPUT_DIR
