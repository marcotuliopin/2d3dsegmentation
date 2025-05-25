#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# echo "Running experiment: UNet-Resnet50 With Dice Loss and Polynomial Scheduler..."
# python train.py --model unet_resnet50_dice_polynomial \
#     --optimizer adam \
#     --lr 5e-3 \
#     --scheduler polynomial \
#     --loss dice_loss \
#     -n unet_resnet101 \

# echo "Evaluating experiment..."
# python evaluate.py -n unet_resnet50_dice_polynomial

# echo "Running experiment: DeepLabV3-Resnet101 With Dice Loss and Polynomial Scheduler..."
# python train.py -n deeplabv3_resnet101_dice_polynomial \
#     --optimizer adam \
#     --lr 5e-3 \
#     --scheduler polynomial \
#     --loss dice_loss \
#     --model deeplabv3_resnet101 \

# echo "Evaluating experiment..."
# python evaluate.py -n deeplabv3_resnet101_dice_polynomial

# echo "Running experiment: UNet-Resnet50 With Dice Loss and Plateau scheduler..."
# python train.py -n unet_resnet50_dice_plateau \
#     --optimizer adam \
#     --lr 1e-4 \
#     --scheduler plateau \
#     --loss dice_loss \
#     --model unet_resnet50 \

# echo "Evaluating experiment..."
# python evaluate.py -n unet_resnet50_dice_plateau

# echo "Running experiment: DeepLabV3-Resnet101 With Dice Loss and Plateau scheduler..."
# python train.py -n deeplabv3_resnet101_dice_plateau \
#     --optimizer adam \
#     --lr 1e-4 \
#     --scheduler plateau \
#     --loss dice_loss \
#     --model deeplabv3_resnet101 \

# echo "Evaluating experiment..."
# python evaluate.py -n deeplabv3_resnet101_dice_plateau

echo "Running experiment: UNet-Resnet50 With CE and Plateau scheduler..."
python train.py -n unet_resnet50_ce_plateau \
    --optimizer adam \
    --lr 1e-4 \
    --scheduler plateau \
    --loss cross_entropy \
    --model unet_resnet50 \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_ce_plateau

echo "Running experiment:  With Cross Entropy Loss and Plateau scheduler..."
python train.py -n deeplabv3_resnet101_ce_plateau \
    --optimizer adam \
    --lr 1e-4 \
    --scheduler plateau \
    --loss cross_entropy \
    --model deeplabv3_resnet101 \

echo "Evaluating experiment..."
python evaluate.py -n deeplabv3_resnet101_ce_plateau

echo "Experiments completed."