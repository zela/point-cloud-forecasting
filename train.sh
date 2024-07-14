#!/bin/bash

python train.py --dataset nuscenes \
    --model-dir models/nusc/1s_forecasting \
    --n-input 2 \
    --n-output 2 \
    --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
    --voxel-size 0.2 \
    --batch-size 4 --num-workers 4 \
    --num-epoch 15

python train.py --dataset nuscenes \
    --model-dir models/nusc/3s_forecasting \
    --n-input 6 \
    --n-output 6 \
    --pc-range -70.0 -70.0 -4.5 70.0 70.0 4.5 \
    --voxel-size 0.2 \
    --batch-size 8 --num-workers 8 \
    --num-epoch 15
