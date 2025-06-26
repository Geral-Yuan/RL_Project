#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python run.py \
    --env_name Boxing-v5 \
    --train \
    --save_ckpt \
    --store_gif