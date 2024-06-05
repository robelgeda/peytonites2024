import os
import sys
import time

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
    for step in range(nsteps):

        #----------------#
        #                #
        # YOUR CODE HERE #
        #                #
        #----------------#

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

            step_filename = f'step_{step:08d}.dat'
            step_path = os.path.join(out_dir, step_filename)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            step_params.write(step_path)

    if verbose:
        print(step)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python main.py <path_to_initial_conditions> <path_to_output_dir>")
        sys.exit(1)

    init_cond_path = sys.argv[1]
    out_dir = sys.argv[2]

    assert 'simout' in out_dir, "'simout' needs to be a part of the output dir name."

    sim_init_cond = SimState.read(init_cond_path)

    tstart = time.monotonic()
    simulation(sim_init_cond, out_dir, verbose=False)
    tend = time.monotonic()
    print(f'Done with {init_cond_path}!')

    run_time = (tend-tstart)*u.s
    if run_time > 3600*u.s:
        run_time = run_time.to(u.hour)
    elif run_time > 60*u.s:
        run_time = run_time.to(u.minute)
    print('run_time:', run_time)
