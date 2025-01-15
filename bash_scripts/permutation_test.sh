#!/bin/bash

# Required arguments.
CSV1=""
CSV2=""
NAME1=""
NAME2=""

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/permutation_test.py \
											--csv1=$CSV1 \
											--csv2=$CSV2 \
											--col_name1=$NAME1 \
											--col_name2=$NAME2
