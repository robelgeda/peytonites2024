# ------------------------- #
# Jax version of nbody code #
# ------------------------- #
# imports

import os
from functools import partial
from os import path
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as u

# turn on 64 bit operations
jax.config.update("jax_enable_x64", True)

# import peytonites package things
import peytonites
from peytonites import Distribution, SimState


# --------------------
# the simulation code
@partial(jax.jit, static_argnums=(7,8,9,))
def step_fun(x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr, mass_arr, soft, G, dt):
    dx = x_arr[:, None] - x_arr
    dy = y_arr[:, None] - y_arr
    dz = z_arr[:, None] - z_arr


    # Avoid division by zero by adding softening length
    r_squared = dx**2 + dy**2 + dz**2 + soft**2
    r_squared = jnp.fill_diagonal(r_squared, 1, inplace=False)  # Avoid self-interaction
    #np.fill_diagonal(r_squared, 1)  # Avoid self-interaction
    #r_squared = jax_fill_diag(r_squared, 1)

    r = jnp.sqrt(r_squared)
    r_cubed = r_squared * r

    # Compute accelerations
    a = -G * mass_arr / r_cubed
    a = jnp.fill_diagonal(a, 0, inplace=False)
    #np.fill_diagonal(a, 0)
    #a = jax_fill_diag(a, 0)

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
    #----------------------------------------
    return x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr,

def jax_simulation(n_steps, sim_init_cond, out_dir, verbose=False):

    G = sim_init_cond.G # cm^3 / (g s^2)
    dt = sim_init_cond.dt
    nsteps = n_steps #sim_init_cond.nsteps
    out_interval = sim_init_cond.out_interval
    soft = sim_init_cond.soft

    init_dist = sim_init_cond.distribution

    number_particles = init_dist.N

    x_arr = jnp.array(init_dist.x.copy())
    y_arr = jnp.array(init_dist.y.copy())
    z_arr = jnp.array(init_dist.z.copy())

    vx_arr = jnp.array(init_dist.vx.copy())
    vy_arr = jnp.array(init_dist.vy.copy())
    vz_arr = jnp.array(init_dist.vz.copy())

    mass_arr = jnp.array(init_dist.m.copy())

    for step in range(nsteps):
        x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr, = step_fun(x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr, mass_arr, soft, G, dt)

        if step % out_interval == 0:

            step_params = sim_init_cond.copy()
            if step > 0:
                step_dist = Distribution.from_arrays(
                    x_arr, y_arr, z_arr,
                    vx_arr, vy_arr, vz_arr,
                    mass_arr, name=init_dist.name)

                step_params = sim_init_cond.copy()
                step_params.distribution = step_dist

            step_filename = f'step_{step:08d}.dat'
            step_path = path.join(out_dir, step_filename)

            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            step_params.write(step_path)

def main():

    # --------------------
    # run some speed tests
    N = [1000, 10000, 50000]
    STEPS = [4000, 8000, 16000, 32000, 64000]
    for n_parts in N:

        for n_step in STEPS:

            # setup the simulation parameters
            N = n_parts
            total_mass = N * (1*u.M_sun).to('g').value
            radius = (10*u.lyr).to('cm').value
            dynamical_time = peytonites.dynamical_time(total_mass, radius)

            dist_1 = peytonites.plummer(
                N // 2, radius,
                x0=0, y0=0, z0=0,
                total_mass=total_mass,
                vx0=0.0, vy0=0.0, vz0=0.0,
                max_radius=radius
            )

            dist_2 = peytonites.plummer(
                N // 2, radius,
                x0=radius*8, y0=radius*8, z0=radius*8,
                total_mass=total_mass,
                vx0=0.0, vy0=0.0, vz0=0.0,
                max_radius=radius
            )


            dist_combo = dist_1 + dist_2

            # Estimate softening param based on number density and mean length:
            soft = peytonites.estimate_softening_length(N, radius, fraction=0.5)

            sim_init_cond = SimState(
                distribution=dist_combo, # `Distribution` object
                nsteps=n_step, # Number of steps in the sim
                dt=dynamical_time / 100, # Time interval for each time-step
                soft=soft, # Softening parameter
                out_interval=n_step // 10 # Output sim every out_interval
            )

            out_dir = './solar_system_simout_GPU_TESTING'
            jitted_sim = jax.jit(jax_simulation, static_argnums=(0,1,2,3))
            steps = n_step

            # run the simulation
            tstart = time()
            jax_simulation(steps, sim_init_cond, out_dir, verbose=False)
            tend = time()

            vector_run_time = (tend-tstart)*u.s
            print(f'N {n_parts}, steps {steps}', 'jax compile_and_run_time', vector_run_time)

# -----------------------
# code entry point
if __name__ == "__main__":
    main()

# -----------------------
# end of file
