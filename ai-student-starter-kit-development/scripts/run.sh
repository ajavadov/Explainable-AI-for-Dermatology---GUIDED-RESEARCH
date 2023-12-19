#!/bin/sh

CUDA_VISIBLE_DEVICES=2 python3 ./src/train.py \
    --train 'test' \
    --batch_size 32 \
    --max_epochs 250 \
    --learning_rate 3e-5
 