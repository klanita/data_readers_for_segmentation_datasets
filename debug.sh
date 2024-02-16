#!/bin/bash

# srun --time 480 --account=staff --partition=gpu.debug --gres=gpu:1 --pty bash -i
# source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
# conda activate pytcu11

# source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
# conda activate mic

python ./convert_h5_to_png.py dataset=wmh_nuhs 
# tgtpath=/usr/bmicnas02/data-biwi-01/contrastive_dg/data/da_data/brain/

