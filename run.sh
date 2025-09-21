#!/bin/bash
#SBATCH -N 1                    # request two nodes
#SBATCH -n 1
#SBATCH --cpus-per-task=32                  # specify 16 MPI processes (8 per node)
#SBATCH -t 15:00:00
#SBATCH -p single
#SBATCH -A hpc_llm_mech
#SBATCH -o slurm-%j.out-%N # optional, name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # optional, name of the stderr, using job and first node values
# below are job commands
module load python/3.9.7-anaconda
source ../lit-llama/llmenv/bin/activate
# run lora fine-tune
#./run_experiments.sh > run.log
python test.py > run.log

