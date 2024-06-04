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

hatch run cuda12:python main.py init_conds/plummer_100p_1000step_init.dat plummer_100p_simout/
hatch run cuda12:python main.py init_conds/plummer_1000p_1000step_init.dat plummer_1000p_simout/

hatch run cuda12:python main.py init_conds/plummer_collision_200p_1000step_init.dat plummer_collision_200p_simout/
hatch run cuda12:python main.py init_conds/plummer_collision_1000p_1000step_init.dat plummer_collision_1000p_simout/

hatch run cuda12:python main.py init_conds/solar_system_100000step_init.dat solar_system_simout/

echo "done"
