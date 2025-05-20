#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# echo "Running experiment: DeepLabV3-Resnet101 pretrained..."
# python train.py -m deeplabv3_resnet101 \
#     -p \
#     -b 4 \
#     -e 100 \
#     --lr 1e-4 \
#     -n deeplabv3_resnet101_pretrained_37labels \

# echo "Evaluating experiment..."
# python evaluate.py -n deeplabv3_resnet101_pretrained_37labels

# echo "Running experiment: FCN-Resnet101 pretrained..."
# python train.py -m fcn_resnet101 \
#     -p \
#     -b 4 \
#     -e 100 \
#     --lr 1e-4 \
#     -n fcn_resnet101_pretrained_square_img \

# echo "Evaluating experiment..."
# python evaluate.py -n fcn_resnet101_pretrained_square_img

# echo "Running experiment: FCN-Resnet50 pretrained..."
# python train.py -m fcn_resnet50 \
#     -p \
#     -b 4 \
#     -e 100 \
#     --lr 1e-4 \
#     -n fcn_resnet50_pretrained_square_img \

# echo "Evaluating experiment..."
# python evaluate.py -n fcn_resnet50_pretrained_square_img

echo "Running experiment: DeepLabV3-Resnet101 Pretrained in NYUDepth V2..."
python train.py -m deeplabv3_resnet101 \
    -p \
    -b 4 \
    -e 70 \
    --lr 5e-5 \
    -n deeplabv3_resnet101_pretrained_nyu_depth_v2_focal_loss \

echo "Evaluating experiment..."
python evaluate.py -n deeplabv3_resnet101_pretrained_nyu_depth_v2_focal_loss

echo "Experiments completed."