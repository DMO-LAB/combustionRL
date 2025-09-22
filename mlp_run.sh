#!/bin/bash
#SBATCH -N 1                    # request one node
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -p single
#SBATCH -A hpc_les_gt
#SBATCH -o slurm-%j.out-%N     # stdout file
#SBATCH -e slurm-%j.err-%N     # stderr file
# below are job commands

# Load Python and activate environment
module load python/3.11.5-anaconda
source activate sundialEnv

# Run PPO LSTM training with real-time logging
python ppo_training.py \
    --mechanism large_mechanism/n-dodecane.yaml \
    --neptune \
    2>&1 | tee run_mlp.log

