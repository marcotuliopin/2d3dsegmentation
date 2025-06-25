# !/bin/bash


# python train.py -n mid_late_fusion_hha_att_2306 \
#     --optimizer adam \
#     --scheduler cosine \
#     --loss weighted_cross_entropy \
#     --model mid_fusion_hha_att \

# echo "Evaluating experiment..."
# python evaluate.py -n mid_late_fusion_hha_att_2306


python train.py -n late_fusion_hha_att_2306 \
    --optimizer adam \
    --scheduler cosine \
    --loss weighted_cross_entropy \
    --model late_fusion_hha_att \

echo "Evaluating experiment..."
python evaluate.py -n late_fusion_hha_att_2306


# python train.py -n early_fusion_hha_att_2306 \
#     --optimizer adam \
#     --scheduler cosine \
#     --loss weighted_cross_entropy \
#     --model early_fusion_hha_att \

# echo "Evaluating experiment..."
# python evaluate.py -n early_fusion_hha_att_2306


# python train.py -n late_fusion_d_2306 \
#     --optimizer adam \
#     --scheduler cosine \
#     --loss weighted_cross_entropy \
#     --model late_fusion_d \

# echo "Evaluating experiment..."
# python evaluate.py -n late_fusion_d_2306


# python train.py -n early_fusion_hha_2306 \
#     --optimizer adam \
#     --scheduler cosine \
#     --loss weighted_cross_entropy \
#     --model early_fusion_hha \

# echo "Evaluating experiment..."
# python evaluate.py -n early_fusion_hha_2306


# python train.py -n early_fusion_d_2306 \
#     --optimizer adam \
#     --scheduler cosine \
#     --loss weighted_cross_entropy \
#     --model early_fusion_d \

# echo "Evaluating experiment..."
# python evaluate.py -n early_fusion_d_2306


# python train.py -n rgb_only_2306 \
#     --optimizer adam \
#     --scheduler cosine \
#     --loss weighted_cross_entropy \
#     --model rgb_only \

# echo "Evaluating experiment..."
# python evaluate.py -n rgb_only_2306


# python train.py -n late_fusion_hha_2306 \
#     --optimizer adam \
#     --scheduler cosine \
#     --loss weighted_cross_entropy \
#     --model late_fusion_hha \

# echo "Evaluating experiment..."
# python evaluate.py -n late_fusion_hha_2306

