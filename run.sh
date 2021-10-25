#!/bin/bash
NUM_NODE=4
EPOCHS=200
MODEL=resnet56
BS=128
LR=0.01
# Static topology
bfrun -np ${NUM_NODE} python train_cifar10.py \
    --epochs ${EPOCHS} \
    --model ${MODEL} \
    --batch-size ${BS} \
    --base-lr ${LR} \
    --disable-dynamic-topology
# Dynamic One-peer topology
bfrun -np ${NUM_NODE} python train_cifar10.py \
    --epochs ${EPOCHS} \
    --model ${MODEL} \
    --batch-size ${BS} \
    --base-lr ${LR} \