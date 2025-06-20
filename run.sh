#!/bin/bash


python train.py -n early_fusion_hha_att_2006 \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model early_fusion_hha_att \

echo "Evaluating experiment..."
python evaluate.py -n early_fusion_hha_att_2006


python train.py -n late_fusion_hha_att_2006 \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model late_fusion_hha_att \

echo "Evaluating experiment..."
python evaluate.py -n late_fusion_hha_att_2006


python train.py -n late_fusion_d_2006 \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model late_fusion_d \

echo "Evaluating experiment..."
python evaluate.py -n late_fusion_d_2006


python train.py -n late_fusion_hha_2006 \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model late_fusion_hha \

echo "Evaluating experiment..."
python evaluate.py -n late_fusion_hha_2006


python train.py -n early_fusion_hha_2006 \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model early_fusion_hha \

echo "Evaluating experiment..."
python evaluate.py -n early_fusion_hha_2006


python train.py -n early_fusion_d_2006 \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model early_fusion_d \

echo "Evaluating experiment..."
python evaluate.py -n early_fusion_d_2006


python train.py -n rgb_only_2006 \
    --optimizer sgd \
    --scheduler step \
    --loss weighted_cross_entropy \
    --model rgb_only \

echo "Evaluating experiment..."
python evaluate.py -n rgb_only_2006
