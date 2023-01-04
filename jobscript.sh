#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 20:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo "Running script..."
python3 script_gpu_cluster.py
