#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running experiment: UNet-Resnet50 With Dice Loss and Polynomial Scheduler..."
python train.py --model unet_resnet50_polynomial \
    --optimizer adam \
    --lr 5e-3 \
    --scheduler polynomial \
    --loss dice_loss \
    -n unet_resnet101 \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_polynomial

echo "Running experiment: UNet-Resnet50 With Dice Loss and Plateau scheduler..."
python train.py --model unet_resnet50_plateau \
    --optimizer adam \
    --lr 1e-4 \
    --scheduler plateau \
    --loss dice_loss \
    -n unet_resnet101 \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_plateau

echo "Running experiment: UNet-Resnet50 With CE and Plateau scheduler..."
python train.py --model unet_resnet50_plateau_ce \
    --optimizer adam \
    --lr 1e-4 \
    --scheduler plateau \
    --loss cross_entropy \
    -n unet_resnet101 \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_plateau_ce

echo "Experiments completed."