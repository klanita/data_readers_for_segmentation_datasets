#!/bin/bash

#SBATCH  --output=./LOGS/%j.out
#SBATCH  --error=./LOGS/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
#SBATCH --constraint='titan_xp|geforce_rtx_2080_ti|geforce_gtx_1080_ti'
#SBATCH  --account=staff

source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
conda activate pytcu11

python -u convert_h5_to_png.py "$@"


