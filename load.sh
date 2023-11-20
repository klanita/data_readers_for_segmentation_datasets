#!/bin/bash

# srun --time 60 --account=staff --partition=gpu.debug --gres=gpu:1 --pty bash -i
# source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
# conda activate pytcu11

python ./main.py 