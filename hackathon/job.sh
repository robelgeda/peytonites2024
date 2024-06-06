#!/bin/bash -l
#SBATCH --job-name=gpu_test
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=openhack
#SBATCH -o output.out
#SBATCH -e output.err


module load anaconda3/2023.3
source /usr/licensed/anaconda3/2023.3/etc/profile.d/conda.sh
conda activate peytonites

python jax_nbody.py
