import os
from os import path
import sys
from time import time

import numpy as np
#import cupy as np

from astropy import units as u
from astropy import constants as const


import peytonites
from peytonites import (
    Distribution, SimState,
    kpc_to_cm, cm_to_kpc,
    lyr_to_cm, cm_to_lyr,
    au_to_cm, cm_to_au
)


def simulation(sim_init_cond, out_dir, verbose=True):

    # This part you need
    G = sim_init_cond.G # cm^3 / (g s^2)
    dt = sim_init_cond.dt
    nsteps = sim_init_cond.nsteps
    out_interval = sim_init_cond.out_interval
    soft = sim_init_cond.soft

    init_dist = sim_init_cond.distribution

    number_particles = init_dist.N

    x_arr = init_dist.x.copy()
    y_arr = init_dist.y.copy()
    z_arr = init_dist.z.copy()

    vx_arr = init_dist.vx.copy()
    vy_arr = init_dist.vy.copy()
    vz_arr = init_dist.vz.copy()

    mass_arr = init_dist.m.copy()

#Computation Start
    tstart = time()
    for step in range(nsteps):
        dx = x_arr[:, np.newaxis] - x_arr
        print('Szie of dx is ',dx.shape)
        dy = y_arr[:, np.newaxis] - y_arr
        dz = z_arr[:, np.newaxis] - z_arr

        # Avoid division by zero by adding softening length
        r_squared = dx**2 + dy**2 + dz**2 + soft**2
        np.fill_diagonal(r_squared, 1)  # Avoid self-interaction

        r = np.sqrt(r_squared)
        r_cubed = r_squared * r

        # Compute accelerations
        a = -G * mass_arr / r_cubed
        np.fill_diagonal(a, 0)
        ax_arr = np.sum(a * dx, axis=1)
        ay_arr = np.sum(a * dy, axis=1)
        az_arr = np.sum(a * dz, axis=1)


        # Update velocities and positions
        vx_arr += ax_arr * dt
        vy_arr += ay_arr * dt
        vz_arr += az_arr * dt

        x_arr += vx_arr * dt
        y_arr += vy_arr * dt
        z_arr += vz_arr * dt

    tend = time()
#Computation End
    run_time = (tend-tstart)*u.s
    if run_time > 3600*u.s:
        run_time = run_time.to(u.hour)
        print('run_time for ',init_cond_path,':', run_time)
    elif run_time > 60*u.s:
        run_time = run_time.to(u.minute)
        print('run_time for ',init_cond_path,':', run_time)
    else:
        print('run_time for ',init_cond_path,':', run_time)
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
    if len(sys.argv) != 3:
        print("Usage: python main.py <path_to_initial_conditions> <path_to_output_dir>")
        sys.exit(1)
    
    init_cond_path = sys.argv[1]
    out_dir = sys.argv[2]

    assert 'simout' in out_dir, "'simout' needs to be a part of the output dir name."

    print('Running ',init_cond_path)
    sim_init_cond = SimState.read(init_cond_path)
    simulation(sim_init_cond, out_dir, verbose=False)
    print('Done!')

