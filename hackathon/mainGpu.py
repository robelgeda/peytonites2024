import os
from os import path
import sys
from time import time

import numpy as np
import cupy as cp

from astropy import units as u
from astropy import constants as const


import peytonites
from peytonites import (
    Distribution, SimState,
    kpc_to_cm, cm_to_kpc,
    lyr_to_cm, cm_to_lyr,
    au_to_cm, cm_to_au
)

N = 25000
nsteps = 50000

###############

total_mass = N * (1*u.M_sun).to('g').value
radius = (10*u.lyr).to('cm').value
dynamical_time = peytonites.dynamical_time(total_mass, radius)

dist_1 = peytonites.plummer(
    N, radius,
    x0=0, y0=0, z0=0,
    total_mass=total_mass,
    vx0=0.0, vy0=0.0, vz0=0.0,
    max_radius=radius
)

#dist_combo.plot()
#plt.show()

# Estimate softening param based on number density and mean length:
soft = peytonites.estimate_softening_length(N, radius, fraction=0.5)

sim_init_cond = SimState(
    distribution=dist_1, # `Distribution` object
    nsteps=nsteps, # Number of steps in the sim
    dt=dynamical_time / 100, # Time interval for each time-step
    soft=soft, # Softening parameter
    out_interval=100 # Output sim every out_interval
)

load_cuda_kernels = r'''
extern "C"{

__global__ void calcForce(float* x_arr, float* y_arr, float* z_arr, float* ax_arr, \
                          float* ay_arr, float* az_arr, float* mass_arr, unsigned int N, float soft, float G)
{
    unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    float ax, ay, az; float dx, dy, dz; float mj, a;
    float xi, yi, zi; float r_ij_squared, r_ij, r_ij_cubed;
    if(i<N){
        ax = 0.0; ay = 0.0; az = 0.0;
        xi = x_arr[i]; yi = y_arr[i]; zi = z_arr[i];
        for(int j=0;j<N;j++){
            if(i!=j){
                dx = xi - x_arr[j];
                dy = yi - y_arr[j];
                dz = zi - z_arr[j];

                r_ij_squared = (dx*dx) + (dy*dy) + (dz*dz) + (soft*soft);
                r_ij = sqrt(r_ij_squared);
                r_ij_cubed = r_ij_squared * r_ij;

                mj = mass_arr[j];
                a = - (G * mj) / r_ij_cubed;

                ax += a * dx;
                ay += a * dy;
                az += a * dz;
            }
        }
        ax_arr[i] = ax;
        ay_arr[i] = ay;
        az_arr[i] = az;
    }
}

__global__ void updatePosition(float* vx_arr, float* vy_arr, float* vz_arr, \
                               float* x_arr, float* y_arr, float* z_arr, float* ax_arr, \
                               float* ay_arr, float* az_arr, unsigned int N, float dt)
{
    unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(i<N){
        vx_arr[i] += ax_arr[i] * dt;
        vy_arr[i] += ay_arr[i] * dt;
        vz_arr[i] += az_arr[i] * dt;

        x_arr[i] += vx_arr[i] * dt;
        y_arr[i] += vy_arr[i] * dt;
        z_arr[i] += vz_arr[i] * dt;
    }
}

}'''


def simulation(sim_init_cond, out_dir, verbose=True):

    # This part you need
    G = sim_init_cond.G # cm^3 / (g s^2)
    G = np.float32(G)
    dt = sim_init_cond.dt
    dt = np.float32(dt)
    nsteps = sim_init_cond.nsteps
    out_interval = sim_init_cond.out_interval
    soft = sim_init_cond.soft
    soft = np.float32(soft)

    init_dist = sim_init_cond.distribution

    number_particles = init_dist.N

    x_arr = init_dist.x.copy()
    y_arr = init_dist.y.copy()
    z_arr = init_dist.z.copy()

    x_arr = np.float32(x_arr)
    y_arr = np.float32(y_arr)
    z_arr = np.float32(z_arr)

    vx_arr = init_dist.vx.copy()
    vy_arr = init_dist.vy.copy()
    vz_arr = init_dist.vz.copy()

    vx_arr = np.float32(vx_arr)
    vy_arr = np.float32(vy_arr)
    vz_arr = np.float32(vz_arr)

    mass_arr = init_dist.m.copy()
    mass_arr = np.float32(mass_arr)

    #cuPy Arrays
    cu_x_arr = cp.array(x_arr, dtype=cp.float32, copy=True)
    cu_y_arr = cp.array(y_arr, dtype=cp.float32, copy=True)
    cu_z_arr = cp.array(z_arr, dtype=cp.float32, copy=True)

    cu_vx_arr = cp.array(vx_arr, dtype=cp.float32, copy=True)
    cu_vy_arr = cp.array(vy_arr, dtype=cp.float32, copy=True)
    cu_vz_arr = cp.array(vz_arr, dtype=cp.float32, copy=True)

    cu_mass_arr = cp.array(mass_arr, dtype=cp.float32, copy=True)

    module = cp.RawModule(code=load_cuda_kernels)
    calcForceKernel = module.get_function('calcForce')
    updatePositionKernel = module.get_function('updatePosition')

#Computation Start
    tstart = time()
    for step in range(nsteps):
        cu_ax_arr = cp.zeros_like(cu_x_arr)
        cu_ay_arr = cp.zeros_like(cu_x_arr)
        cu_az_arr = cp.zeros_like(cu_x_arr)

        xThreads = 256
        grid = (int((number_particles+xThreads-1)/xThreads),)
        block = (xThreads,)
        calcForceKernel(grid,block,(cu_x_arr,cu_y_arr,cu_z_arr,cu_ax_arr,cu_ay_arr,cu_az_arr,cu_mass_arr,number_particles,soft,G))
        grid = (int((number_particles+xThreads-1)/xThreads),)
        block = (xThreads,)
        updatePositionKernel(grid,block,(cu_vx_arr,cu_vy_arr,cu_vz_arr,cu_x_arr,cu_y_arr,cu_z_arr,cu_ax_arr,cu_ay_arr,cu_az_arr,number_particles,dt))

    tend = time()
#Computation End

    x_arr = cp.asnumpy(cu_x_arr)
    y_arr = cp.asnumpy(cu_y_arr)
    z_arr = cp.asnumpy(cu_z_arr)

    run_time = (tend-tstart)*u.s
    if run_time > 3600*u.s:
        run_time = run_time.to(u.hour)
        print('run_time for N =',N,'and steps = ',nsteps,':', run_time)
    elif run_time > 60*u.s:
        run_time = run_time.to(u.minute)
        print('run_time for N =',N,'and steps = ',nsteps,':', run_time)
    else:
        print('run_time for N =',N,'and steps = ',nsteps,':', run_time)
    #----------------------------------------
        # Output code every out_interval:
        if step % out_interval == 0:
            if verbose:
                print(step)
            step_params = sim_init_cond.copy()
            if step > 0:
                step_dist = Distribution.from_arrays(
                    x_arr, y_arr, z_arr,
                    vx_arr, vy_arr, vz_arr,
                    mass_arr, name=init_dist.name)

                step_params = sim_init_cond.copy()
                step_params.distribution = step_dist

            step_filename = 'step_{:08d}.dat'.format(step)
            step_path = path.join(out_dir, step_filename)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            step_params.write(step_path)

    if verbose:
        print(step)
    return


if __name__ == '__main__':
#    if len(sys.argv) != 3:
#        print("Usage: python main.py <path_to_initial_conditions> <path_to_output_dir>")
#        sys.exit(1)
    
    init_cond_path = sys.argv[1]
    out_dir = sys.argv[2]

#    assert 'simout' in out_dir, "'simout' needs to be a part of the output dir name."

#    print('Running ',init_cond_path)
#    sim_init_cond = SimState.read(init_cond_path)
    simulation(sim_init_cond, out_dir, verbose=False)
    print('Done!')

