#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python train.py -n unet_resnet50_2enc_hha \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model unet_hha_dual_encoder \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_2enc_hha

python train.py -n unet_resnet50_2enc_d \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model unet_depth_dual_encoder \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_2enc_d

python train.py -n unet_resnet50_hha_1 \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model unet_hha_concatenate \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_hha_1

python train.py -n unet_resnet50_hha_fl \
    --optimizer sgd \
    --scheduler step \
    --loss focal_loss \
    --model unet_hha_concatenate \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_hha_fl

echo "unet_resnet50_w experiment started..."
python train.py -n unet_resnet50_w_1 \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model unet_resnet50 \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_w_1

echo "unet_resnet50_d experiment started..."
python train.py -n unet_resnet50_d_1 \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model unet_depth_concatenate \

echo "Evaluating experiment..."
python evaluate.py -n unet_resnet50_d_1

echo "Experiments completed."