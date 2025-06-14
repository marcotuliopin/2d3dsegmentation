#!/bin/bash

mkdir -p data/NYUDepthv2_seg

git clone --depth 1 https://huggingface.co/datasets/wyrx/NYUDepthv2_seg data/NYUDepthv2_seg
rm -rf data/NYUDepthv2_seg/.git data/NYUDepthv2_seg/.gitattributes

mv data/NYUDepthv2_seg/data/train-00000-of-00001.parquet data/NYUDepthv2_seg/data/train.parquet
mv data/NYUDepthv2_seg/data/test-00000-of-00001.parquet data/NYUDepthv2_seg/data/test.parquet