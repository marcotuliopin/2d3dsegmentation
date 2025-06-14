#!/bin/bash

python train.py -n unet_attention \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model unet_hha_attention \

echo "Evaluating experiment..."
python evaluate.py -n unet_attention
