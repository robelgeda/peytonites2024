#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -t 03:00:00
#SBATCH -J peytonites
#SBATCH -o output.out
#SBATCH -e output.err
#SBATCH -A astro
#SBATCH --gres=gpu:1
#SBATCH --reservation=openhack

module load anaconda3/2024.2
module list 

conda init
conda activate peytonites
conda list

#Original Code
#python main.py init_conds/plummer_100p_1000step_init.dat plummer_100p_simout/
#python main.py init_conds/plummer_1000p_1000step_init.dat plummer_1000p_simout/

#python main.py init_conds/plummer_collision_200p_1000step_init.dat plummer_collision_200p_simout/
#python main.py init_conds/plummer_collision_1000p_1000step_init.dat plummer_collision_1000p_simout/

#python main.py init_conds/solar_system_100000step_init.dat solar_system_simout/

#Vectorized Code
#python mainVec.py init_conds/plummer_100p_1000step_init.dat plummer_100p_simout/
#python mainVec.py init_conds/plummer_1000p_1000step_init.dat plummer_1000p_simout/

#python mainVec.py init_conds/plummer_collision_200p_1000step_init.dat plummer_collision_200p_simout/
#python mainVec.py init_conds/plummer_collision_1000p_1000step_init.dat plummer_collision_1000p_simout/

#python mainVec.py init_conds/solar_system_100000step_init.dat solar_system_simout/

#GPU Code
#python mainGpu.py init_conds/plummer_100p_1000step_init.dat plummer_100p_simout/
#python mainGpu.py init_conds/plummer_1000p_1000step_init.dat plummer_1000p_simout/

#python mainGpu.py init_conds/plummer_collision_200p_1000step_init.dat plummer_collision_200p_simout/
#python mainGpu.py init_conds/plummer_collision_1000p_1000step_init.dat plummer_collision_1000p_simout/

#python mainGpu.py init_conds/solar_system_100000step_init.dat solar_system_simout/

#GPU Code
#python mainGpuOpt1.py init_conds/plummer_100p_1000step_init.dat plummer_100p_simout/
#python mainGpuOpt1.py init_conds/plummer_1000p_1000step_init.dat plummer_1000p_simout/

#python mainGpuOpt1.py init_conds/plummer_collision_200p_1000step_init.dat plummer_collision_200p_simout/
#python mainGpuOpt1.py init_conds/plummer_collision_1000p_1000step_init.dat plummer_collision_1000p_simout/

#python mainGpuOpt1.py init_conds/solar_system_100000step_init.dat solar_system_simout/

python mainGpu.py init_conds/plummer_100p_1000step_init.dat plummer_100p_simout/

python mainGpuOpt1.py init_conds/plummer_100p_1000step_init.dat plummer_100p_simout/
