#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=4:00:00

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate mil

python main.py -m model=attention,additive dataset.bag_size=16,49,100 blank_ratio=10-90,10-30,40-60,70-90