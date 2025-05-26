#!/bin/bash

python scripts/split_parquet.py \
    --input data/NYUDepthv2_seg/data/train.parquet \
    --output-train data/NYUDepthv2_seg/data/split_train.parquet \
    --output-val data/NYUDepthv2_seg/data/split_val.parquet \
    --val-size 0.3 \
    --seed 42 \