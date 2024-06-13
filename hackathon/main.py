import os
from os import path
import sys
from time import time

import numpy as np

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

        ax_arr = np.zeros_like(x_arr)
        ay_arr = np.zeros_like(x_arr)
        az_arr = np.zeros_like(x_arr)

        for i in range(number_particles):
            ax, ay, az = 0., 0., 0.
            xi, yi, zi = x_arr[i], y_arr[i], z_arr[i]

            for j in range(number_particles):
                if i == j:
                    continue

                dx = xi - x_arr[j]
                dy = yi - y_arr[j]
                dz = zi - z_arr[j]

                r_ij_squared = dx**2 + dy**2 + dz**2 + soft**2
                r_ij = np.sqrt(r_ij_squared)
                r_ij_cubed = r_ij_squared * r_ij

                mj = mass_arr[j]
                a = - (G * mj) / r_ij_cubed

                ax += a * dx
                ay += a * dy
                az += a * dz

            ax_arr[i] = ax
            ay_arr[i] = ay
            az_arr[i] = az


        for i in range(number_particles):

            vx_arr[i] += ax_arr[i] * dt
            vy_arr[i] += ay_arr[i] * dt
            vz_arr[i] += az_arr[i] * dt

            x_arr[i] += vx_arr[i] * dt
            y_arr[i] += vy_arr[i] * dt
            z_arr[i] += vz_arr[i] * dt

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

