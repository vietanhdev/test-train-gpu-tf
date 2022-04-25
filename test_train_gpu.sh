#!/bin/bash

echo $'WARNING: files created within this docker will be owned by root.\n'
docker run --gpus all -it -v $PWD/:/tf/work -w /tf/work tensorflow/tensorflow:2.7.1-gpu bash -c "python train.py"
