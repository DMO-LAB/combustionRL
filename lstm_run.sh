#!/bin/bash
#SBATCH -N 1                    # request one node
#SBATCH -n 1
#SBATCH -t 15:00:00
#SBATCH -p single
#SBATCH -A hpc_les_gt
#SBATCH -o slurm-%j.out-%N     # stdout file
#SBATCH -e slurm-%j.err-%N     # stderr file
# below are job commands

# Load Python and activate environment
module load python/3.11.5-anaconda
source activate sundialEnv

# Run PPO LSTM training with real-time logging
python train_ppo_lstm.py \
    --from_scratch \
    --mechanism large_mechanism/n-dodecane.yaml \
    --fuel nc12h26 \
    --oxidizer "O2:0.21, N2:0.79" \
    --epsilon 1e-4 \
    --horizon 100 \
    --rollout_steps 2048 \
    --total_updates 300 \
    --eval_interval 10 \
    --eval_time 7e-2 \
    --eval_temperatures 650 700 1100 \
    --eval_pressures 3.0 10.0 1.0 \
    --eval_phis 1 1.66 1.0 \
    --out_dir ppo_runs/run_lstm \
    2>&1 | tee run_lstm.log
