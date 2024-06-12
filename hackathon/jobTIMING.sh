#!/bin/bash -l
#SBATCH --job-name=GPU_JAX
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=openhack
#SBATCH -o output2.out
#SBATCH -e output2.err
#SBATCH --mem=200G


module load anaconda3/2023.3
source /usr/licensed/anaconda3/2023.3/etc/profile.d/conda.sh
conda activate peytonites

python jax_nbody.py
