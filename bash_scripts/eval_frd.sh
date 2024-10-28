#!/bin/bash

# Required arguments.
DSET1="/workspace/"
DSET2="/workspace/"

# Run script.
docker run --rm -v $(pwd):/workspace frd python -m frd_score $DSET1 $DSET2
