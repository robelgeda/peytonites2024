#!/usr/bin/env bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -t 03:00:00
#SBATCH -o output.out
#SBATCH -e output.err
#SBATCH --gres=gpu:1
#SBATCH --reservation=openhack

set -euo pipefail

module load cudatoolkit/12.4
module list

hatch run jax:python jax_nbody.py

echo "done"
