#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python train.py -n unet_resnet50_hha \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model unet_hha_concatenate \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_hha

# echo "unet_resnet50_w experiment started..."
# python train.py -n unet_resnet50_w \
#     --optimizer sgd \
#     --scheduler step \
#     --loss weighted_cross_entropy \
#     --model unet_resnet50 \

# echo "Evaluating experiment..."
# python evaluate.py -n unet_resnet50_w

# echo "unet_resnet50_d experiment started..."
# python train.py -n unet_resnet50_d \
#     --optimizer sgd \
#     --scheduler step \
#     --loss weighted_cross_entropy \
#     --model unet_depth_concatenate \

# echo "Evaluating experiment..."
# python evaluate.py -n unet_resnet50_d

# echo "Experiments completed."