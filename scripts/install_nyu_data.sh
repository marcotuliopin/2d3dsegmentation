#!/bin/bash

mkdir -p data/NYUDepthv2_seg

git clone --depth 1 https://huggingface.co/datasets/wyrx/NYUDepthv2_seg data/nyuv2/old
rm -rf data/nyuv2/old/.git data/nyuv2/old/.gitattributes

mv data/nyuv2/old/data/train-00000-of-00001.parquet data/nyuv2/data/train.parquet
mv data/nyuv2/old/data/test-00000-of-00001.parquet data/nyuv2/data/test.parquet