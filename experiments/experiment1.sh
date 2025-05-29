#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# python train.py -n unet_resnet50 \
#     --optimizer adam \
#     --lr 1e-4 \
#     --scheduler plateau \
#     --loss cross_entropy \
#     --model unet_resnet50 \

# echo "Evaluating experiment..."
# python evaluate.py -n unet_resnet50

# python train.py -n unet_resnet50_d \
#     --optimizer adam \
#     --lr 1e-4 \
#     --scheduler plateau \
#     --loss cross_entropy \
#     --model unet_depth_concatenate \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_d

echo "Experiments completed."