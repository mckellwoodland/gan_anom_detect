#!/bin/bash

# Required arguments.
CSV1="/workspace/logs/aucs/needles_wd_body.csv"
CSV2="/workspace/logs/aucs/needles_wd.csv"
NAME1="AUROC"
NAME2="AUROC"

# Run script.
docker run --rm -v $(pwd):/workspace gan_anom_detect python /workspace/python_scripts/permutation_test.py \
											--csv1=$CSV1 \
											--csv2=$CSV2 \
											--col_name1=$NAME1 \
											--col_name2=$NAME2
