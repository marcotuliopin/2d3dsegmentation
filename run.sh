#!/bin/bash


python train.py -n unet_mid_fusion_hha_1906 \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model unet_mid_fusion_hha \

echo "Evaluating experiment..."
python evaluate.py -n unet_mid_fusion_hha_1906


# python train.py -n unet_hha_2enc_1806 \
#     --optimizer sgd \
#     --scheduler step \
#     --loss weighted_cross_entropy \
#     --model unet_hha_dual_encoder \

# echo "Evaluating experiment..."
# python evaluate.py -n unet_hha_2enc_1806


# python train.py -n unet_depth_2enc_1806 \
#     --optimizer sgd \
#     --scheduler step \
#     --loss weighted_cross_entropy \
#     --model unet_depth_dual_encoder \

# echo "Evaluating experiment..."
# python evaluate.py -n unet_depth_2enc_1806


# python train.py -n unet_depth_2enc_1806 \
#     --optimizer sgd \
#     --scheduler step \
#     --loss weighted_cross_entropy \
#     --model unet_depth_dual_encoder \

# echo "Evaluating experiment..."
# python evaluate.py -n unet_depth_2enc_1806


# python train.py -n unet_hha_concatenate_1806 \
#     --optimizer sgd \
#     --scheduler step \
#     --loss weighted_cross_entropy \
#     --model unet_hha_concatenate \

# echo "Evaluating experiment..."
# python evaluate.py -n unet_hha_concatenate_1806


# python train.py -n unet_depth_concatenate_1806 \
#     --optimizer sgd \
#     --scheduler step \
#     --loss weighted_cross_entropy \
#     --model unet_depth_concatenate \

# echo "Evaluating experiment..."
# python evaluate.py -n unet_depth_concatenate_1806


# python train.py -n unet_1806 \
#     --optimizer sgd \
#     --scheduler step \
#     --loss weighted_cross_entropy \
#     --model unet \

# echo "Evaluating experiment..."
# python evaluate.py -n unet_1806
