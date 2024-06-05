import re
from glob import glob
from os import path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.animation import FuncAnimation, PillowWriter

from .core import SimState

__all__ = [
    'simulation_to_gif_2d', 'simulation_to_gif_3d',
]


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)



def simulation_to_gif_3d(simulation_dir, gif_filename='simulation_3d.gif',
                         scale=1, alpha=1., unit='cm', extent=None, fps=25):

    gif_path = path.join(simulation_dir, gif_filename)
    fb = natural_sort(glob(path.join(simulation_dir, '*step*.dat')))

    if len(fb) < 1:
        return

    init_cond = SimState.read(fb[0]).distribution.points
    init_cond = np.array(init_cond, float)
    x_arr, y_arr, z_arr, vx_arr ,vy_arr ,vz_arr, masses = init_cond.T
    number_particles = len(x_arr)

    init_xmax = abs(x_arr).max()*scale
    init_ymax = abs(y_arr).max()*scale
    init_zmax = abs(z_arr).max()*scale

    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')

    def animate(i):
        axs.clear()

        max_range = max([init_xmax, init_ymax, init_zmax])
        if extent is not None:
            max_range = extent.to(unit).value if isinstance(extent, u.Quantity) else extent

        axs.set_xlim(-max_range, max_range)
        axs.set_ylim(-max_range, max_range)
        axs.set_zlim(-max_range, max_range)

        # w_axis_value = 1.0
        # axs.w_xaxis.set_pane_color((w_axis_value, w_axis_value, w_axis_value, 1.0))
        # axs.w_yaxis.set_pane_color((w_axis_value, w_axis_value, w_axis_value, 1.0))
        # axs.w_zaxis.set_pane_color((w_axis_value, w_axis_value, w_axis_value, 1.0))

        f = fb[i]
        dist = SimState.read(f).distribution

        x = (dist.x * u.cm).to(unit).value
        y = (dist.y * u.cm).to(unit).value
        z = (dist.z * u.cm).to(unit).value

        scatter1 = axs.scatter3D(x, y, z, alpha=alpha, color='black')
        return [scatter1]
    ani = FuncAnimation(fig, animate, interval=1, blit=True, repeat=True, frames=len(fb))
    ani.save(gif_path, dpi=100, writer=PillowWriter(fps=fps))
    plt.show()


def simulation_to_gif_2d(simulation_dir, gif_filename='simulation_2d.gif',
                         scale=1., alpha=1., unit='cm', extent=None, fps=25):

    gif_path = path.join(simulation_dir, gif_filename)
    fb = natural_sort(glob(path.join(simulation_dir, '*step*.dat')))

    if len(fb) < 1:
        return

    init_cond = SimState.read(fb[0]).distribution.points
    init_cond = np.array(init_cond, float)
    x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr, masses = init_cond.T
    number_particles = len(x_arr)

    init_xmax = abs(x_arr).max() * scale
    init_ymax = abs(y_arr).max() * scale
    init_zmax = abs(z_arr).max() * scale

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    def animate(i):
        axs[0].clear()
        axs[1].clear()

        f = fb[i]
        dist = SimState.read(f).distribution

        x = (dist.x * u.cm).to(unit).value
        y = (dist.y * u.cm).to(unit).value
        z = (dist.z * u.cm).to(unit).value

        coordinates = [(x, y), (x, z)]
        labels = [('X', 'Y'), ('X', 'Z')]

        max_range = max([init_xmax, init_ymax, init_zmax])
        if extent is not None:
            max_range = extent.to(unit).value if isinstance(extent, u.Quantity) else extent

        scatter_list = []
        for ax, (data_x, data_y), (label_x, label_y) in zip(axs, coordinates, labels, strict=False):
            scatter = ax.scatter(data_x, data_y, alpha=alpha, color='black')
            ax.set_xlabel(f'{label_x} [{unit}]')
            ax.set_ylabel(f'{label_y} [{unit}]')
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            scatter_list.append(scatter)

        return scatter_list

    ani = FuncAnimation(fig, animate, interval=100, blit=False, repeat=True, frames=len(fb))
    ani.save(gif_path, dpi=100, writer=PillowWriter(fps=fps))
    plt.show()
