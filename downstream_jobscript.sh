#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo "Running script..."
python3 downstream_only.py
