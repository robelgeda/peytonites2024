import random as rn

import numpy as np

from .core import Distribution, G

__all__ = [
    'collapse_shpere',
    'disk',
    'plummer',
    'solar_system',
]

def solar_system():
    M_sun = 1.988e+33 # in grams
    masses = np.array([M_sun, 3.300e+26, 4.870e+27,
                       5.970e+27, 6.420e+26, 1.898e+30,
                       5.680e+29, 8.680e+28, 1.020e+29]) # in grams

    r_dist = np.array([0., 5.790e+12, 1.082e+13,
                       1.496e+13, 2.279e+13, 7.785e+13,
                       1.434e+14, 2.871e+14, 4.495e+14]) # in cm

    x_arr = r_dist.copy()
    y_arr = np.zeros_like(r_dist)
    z_arr = np.zeros_like(r_dist)

    v_kep = np.zeros_like(r_dist)
    v_kep[1:] = np.sqrt(G * M_sun / r_dist[1:])

    vx_arr = np.zeros_like(r_dist)
    vy_arr = v_kep.copy()
    vz_arr = np.zeros_like(r_dist)

    return Distribution.from_arrays(
        x_arr, y_arr, z_arr,
        vx_arr, vy_arr, vz_arr,
        masses, name='Solar')


def normal(phi, psi):
    v1 = np.array([np.cos(phi), 0, np.sin(phi)])
    v2 = np.array([0, np.cos(psi), np.sin(psi)])
    return np.cross(v1, v2)


# Only works when phi0=0.0, psi0=0.0 for now
def disk(
        N, radius,
        x0, y0, z0, disk_mass, main_mass,
        vx0=0., vy0=0., vz0=0.,
        phi0=0.0, psi0=0.0):
    points = []

    n = a, b, c = normal(phi0, psi0)

    mass = disk_mass / N
    dens = N / (radius * radius)
    r_list = []
    for _ in range(N):
        r = rn.uniform(0.0, radius) + 4e3
        thet = rn.uniform(0.0000001, 2.0 * np.pi) #+ phi0

        i = r * np.cos(thet)
        j = r * np.sin(thet)

        x = i * np.cos(phi0)
        y = j * np.cos(psi0)
        #z = np.sqrt((sin(phi0) ** 2 * i ** 2) + (sin(psi0) ** 2 * j ** 2))
        z = (-(a * x) - (b * y)) / c

        m0 = 0 #len(np.where(np.array(r_list) < r)[0]) * mass
        v = 1 * np.sqrt((G * (main_mass + m0)) / r)  # / np.sqrt(2)

        vi = v * -np.sin(thet)
        vj = v * np.cos(thet)

        vx = vi * np.cos(phi0)
        vy = vj * np.cos(psi0)
        #vz = np.sqrt((sin(phi0) ** 2 * vi ** 2) + (sin(psi0) ** 2 * vj ** 2))
        vz = (-(a * vx) - (b * vy)) / c

        line = [str(j) for j in [x+x0, y+y0, z+z0, vx+vx0, vy+vy0, vz+vz0, mass]]
        points.append(line)
        r_list.append(r)

    points[0] = [str(j) for j in [x0, y0, z0, vx0, vy0, vz0, main_mass]]
    return Distribution(points, name='Disk')


def collapse_shpere(N, radius,
                    x, y, z, m,
                    vx=0., vy=0., vz=0.):
    points = []

    for _ in range(N):
        r = rn.uniform(0.0, radius)
        phi = rn.uniform(0.0000001, 2.0 * np.pi)
        costheta = rn.uniform(-1,1)
        thet = np.arccos(costheta)

        xi = r*np.sin(thet)*np.cos(phi)
        yi = r*costheta
        zi = r*np.sin(thet)*np.sin(phi)

        vxi = 0.
        vyi = 0.
        vzi = 0.

        line = [str(j) for j in [x+xi,y+yi,z+zi,vxi+vx,vyi+vy,vzi+vz, m]]
        points.append(line)
    return Distribution(points, name='CollapseShpere')


def scalar_to_sph(r, thet=None, phi=None):
    if phi is None:
        phi = rn.uniform(0.0000001, 2.0 * np.pi)
    if thet is None:
        costheta = rn.uniform(-1, 1)
        thet = np.arccos(costheta)

    x = r * np.sin(thet) * np.cos(phi)
    y = r * np.sin(thet) * np.sin(phi)
    z = r * costheta

    return x, y, z, thet, phi


def plummer(N, radius,
            x0, y0, z0, total_mass,
            vx0=0., vy0=0., vz0=0,
            max_radius=np.inf):

    points = []
    if N == 0:
        return points

    a = radius
    M = total_mass
    mass = M/N

    for _ in range(N):
        r = None
        while r is None or r > max_radius:
            uni_draw = np.random.uniform(0, 1)
            r = a * (uni_draw**(-2/3) - 1)**(-1/2)

        x, y, z, thet, phi = scalar_to_sph(r)

        # Von Neumann acceptance-rejection technique
        q = 0
        g_of_q = 0.1
        while g_of_q > q**2 * (1 - q**2)**3.5:
            q = np.random.uniform(0.1, 1)
            g_of_q = np.random.uniform(0, 0.1)
        v_esc = ((2*G*M)/(r**2 + a**2)**(1/2))**(1/2)
        v = q * v_esc

        vx, vy, vz, *_ = scalar_to_sph(v)

        line = [str(j) for j in [x+x0, y+y0, z+z0, vx+vx0, vy+vy0, vz+vz0, mass]]
        points.append(line)

    points[0] = [str(j) for j in [x0, y0, z0, vx0, vy0, vz0, mass]]

    return Distribution(points, name='Plummer')
