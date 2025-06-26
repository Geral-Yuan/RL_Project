#!/bin/bash

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

python run.py \
    --env_name HalfCheetah-v4