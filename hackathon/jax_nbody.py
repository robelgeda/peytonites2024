# ------------------------- # 
# Jax version of nbody code #
# ------------------------- #
# imports

import os
from os import path 
from time import time
import jax
import jax.numpy as jnp
import numpy as np 
from astropy import units as u
from astropy import constants as const 
from functools import partial

# plot parameters
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['font.size'] = 12
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["xtick.minor.size"] = 4.5
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["ytick.minor.size"] = 4.5
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["xtick.minor.width"] = 1.5
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["ytick.minor.width"] = 1.5
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({"text.usetex": True})

# import peytonites package things
import peytonites
from peytonites import (
    Distribution, SimState,
    kpc_to_cm, cm_to_kpc, 
    lyr_to_cm, cm_to_lyr, 
    au_to_cm, cm_to_au
)

# the jax simulation code
@jax.jit
def body(x_arr, 
        y_arr, 
        z_arr, 
        mass_arr,
        vx_arr, 
        vy_arr,
        vz_arr,
        G,
        dt):
    
    dx = x_arr[:, jnp.newaxis] - x_arr
    dy = y_arr[:, jnp.newaxis] - y_arr
    dz = z_arr[:, jnp.newaxis] - z_arr
        
    # Avoid division by zero by adding softening length
    r_squared = dx**2 + dy**2 + dz**2 + soft**2
    jnp.fill_diagonal(r_squared, 1, inplace=False)  # Avoid self-interaction
        
    r = jnp.sqrt(r_squared)
    r_cubed = r_squared * r
        
    # Compute accelerations
    a = -G * mass_arr / r_cubed
    jnp.fill_diagonal(a, 0, inplace=False)  
    ax_arr = jnp.sum(a * dx, axis=1)
    ay_arr = jnp.sum(a * dy, axis=1)
    az_arr = jnp.sum(a * dz, axis=1)
        
        
    # Update velocities and positions
    vx_arr += ax_arr * dt
    vy_arr += ay_arr * dt
    vz_arr += az_arr * dt
        
    x_arr += vx_arr * dt
    y_arr += vy_arr * dt
    z_arr += vz_arr * dt
    #----------------------------------------
        
    return x_arr, y_arr, z_arr, mass_arr,vx_arr, vy_arr,vz_arr,G,dt


def jax_simulation(n_steps, sim_init_cond, out_dir, verbose=False):
    
    G = sim_init_cond.G # cm^3 / (g s^2)
    dt = sim_init_cond.dt
    nsteps = n_steps  #sim_init_cond.nsteps
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
    #----------------------------------------
    for step in range(nsteps): 
        x_arr, y_arr, z_arr, mass_arr,vx_arr, vy_arr,vz_arr,G,dt = body(x_arr, 
        y_arr, 
        z_arr, 
        mass_arr,
        vx_arr, 
        vy_arr,
        vz_arr,
        G,
        dt)
        
        if step % out_interval == 0:
            
            step_params = sim_init_cond.copy()
            if step > 0:
                step_dist = Distribution.from_arrays(
                    np.array(x_arr), np.array(y_arr), np.array(z_arr), 
                    np.array(vx_arr), np.array(vy_arr), np.array(vz_arr),
                    np.array(mass_arr), name=init_dist.name)

                step_params = sim_init_cond.copy()
                step_params.distribution = step_dist                
            
            step_filename = 'step_{:08d}.dat'.format(step)
            step_path = path.join(out_dir, step_filename)
            
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            
            step_params.write(step_path)

    return 

# setup the simulation parameters
N = 50000
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
    nsteps=20000, # Number of steps in the sim
    dt=dynamical_time / 100, # Time interval for each time-step
    soft=soft, # Softening parameter
    out_interval=1000 # Output sim every out_interval
)

out_dir = './solar_system_simout_GPU'
jitted_sim = jax.jit(jax_simulation, static_argnums=(0,1,2,3))
steps = 20000

# run the simulation
print('           JIT compiled')
tstart = time()
jax_simulation(steps, sim_init_cond, out_dir, verbose=False)
tend = time()

vector_run_time = (tend-tstart)*u.s
print('done', 'jax compile_and_run_time', vector_run_time)

# plot the stuff
file_path = path.join(out_dir, 'step_00000090.dat')
# Load simulation state
step_state = SimState.read(file_path)
# Get particle distribution 
dist = step_state.distribution

# make gifs
peytonites.simulation_to_gif_2d(
    out_dir, 
    gif_filename='solar_sys_2d.gif', 
    extent=1e7*u.AU, 
    unit='AU'
)

peytonites.simulation_to_gif_3d(out_dir, gif_filename='solar_sys_3d.gif', extent=1e7*u.AU, unit='AU')
