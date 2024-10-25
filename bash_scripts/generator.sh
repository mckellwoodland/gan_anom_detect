#~/bin/bash

# Required arguments.
MODEL_PKL="/workspace/"
OUT_DIR="/workspace/"

# Optional arguments.
SEEDS="0-49999"

# Run script.
docker run --rm -v $(pwd):/workspace sg2ada python /workspace/stylegan2-ada-pytorch/generate.py \
									--network $MODEL_PKL \
									--seeds $SEEDS \
									--outdir $OUT_DIR
