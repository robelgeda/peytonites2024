from os import path 

import numpy as np

from astropy import units as u

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['font.size'] = 12

__all__ = [
    'SimState', 'Distribution',
    'dynamical_time', 'estimate_softening_length'
]


G = 6.67e-08 # cm^3 / (g s^2)


class SimState:
    def __init__(self, distribution, nsteps, dt, soft, out_interval):
        """
        - `distribution`: Particle distributions stored in a `Distribution` object (see next main bullet)
        - `nsteps`: Number of timesteps to run the simulation
        - `dt`: How many seconds a single time-step is. 
        - `soft`: Softening parameter.
        - `out_interval`: Output and save to file every `out_interval` (output is saved `SimState.write(filename)`.
        """
        self.distribution = distribution
        self.nsteps = int(nsteps)
        self.dt = dt
        self.soft = soft
        self.out_interval = out_interval
        self.G = G

    def write(self, filename):
        self.distribution._write_init_file(
            filename,
            self.nsteps,
            self.dt,
            self.soft,
            self.out_interval,
            self.G
        )

    @staticmethod
    def read(filename):
        with open(filename, "r") as f:
            line = f.readline().replace('\n', '').split(' ')
            N, nsteps, dt, soft, out_interval, G_in = [float(i) for i in line]
            nsteps = int(nsteps)
            out_interval = int(out_interval)
            points = []
            for line in f:
                points.append(line.split(' '))
        basename = path.splitext(filename)[0]
        dist = Distribution(points, name=basename)
        sim_init_cond = SimState(dist, nsteps, dt, soft, out_interval)
        sim_init_cond.G = G_in
        return sim_init_cond
    
    def copy(self):
        return SimState(
            self.distribution, self.nsteps, self.dt, 
            self.soft, self.out_interval)


class Distribution:
    def __init__(self, points, name=''):
        """
        points is a list of particles with each particle 
        being a list with the following:

        points = [
            [x, y, z, vx, vy, vz, mass],
            [x, y, z, vx, vy, vz, mass],
            .
            .
            .
            [x, y, z, vx, vy, vz, mass]
        ]
        """

        self.name = name

        self._points = points
        self.points = np.array(self._points, float) 
        self.points_T = self.points.T
        self.N = len(self.points)

        self.x, self.y, self.z, self.vx, self.vy, self.vz, self.m  = self.points_T


    def __str__(self):
        return 'Points: {}, Distribution: {}'.format(len(self.points), self.name)


    def __add__(self, other):
        '''Combine two Distribution points'''
        if not isinstance(other, Distribution):
            raise TypeError('You can only sum Distribution to another Distribution')
        
        if self.name and other.name:
            name = self.name + '+' + other.name 
        else:
            name = self.name if self.name else other.name

        return Distribution(self._points +  other._points, name)
    
    
    def _write_lines(self, f):
        for line in self.points:
                line = " ".join([str(j) for j in line])
                f.write(line+"\n")

    def _write_init_file(self, filename, nsteps, dt, soft, out_interval, G_out):
        with open(filename, "w") as f:
            f.write(str(self.N)+" "+str(nsteps)+" "+str(dt)+" "+str(soft)+" "+str(out_interval)+" "+str(G_out)+"\n")
            self._write_lines(f)

    def write(self, filename):
        with open(filename, "w") as f:
            self._write_lines(f)

    @staticmethod
    def read(filename):
        with open(filename, "r") as f:
            points = []
            for line in f:
                points.append(line.split(' '))
        basename = path.splitext(filename)[0]
        return Distribution(points, name=basename)


    def plot(self, unit='cm'):
        fig, axs = plt.subplots(1, 3, figsize=[6*3, 6])

        x = (self.x * u.cm).to(unit).value
        y = (self.y * u.cm).to(unit).value
        z = (self.z * u.cm).to(unit).value

        extent_min = min([x.min(), y.min(), z.min()])
        extent_max = max([x.max(), y.max(), z.max()])

        coordinates = [(x, y), (x, z), (y, z)]
        labels = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]

        for ax, (data_x, data_y), (label_x, label_y) in zip(axs, coordinates, labels):
            plt.sca(ax)
            plt.scatter(data_x, data_y)
            plt.xlabel(f'{label_x} [{unit}]')
            plt.ylabel(f'{label_y} [{unit}]')
            plt.xlim(extent_min, extent_max)
            plt.ylim(extent_min, extent_max)

        plt.show()

    @staticmethod
    def from_arrays(
        x_arr, y_arr, z_arr, 
        vx_arr, vy_arr, vz_arr,
        mass_arr, name=''):

        points = np.array([
            x_arr, y_arr, z_arr, 
            vx_arr, vy_arr, vz_arr,
            mass_arr
        ]).T

        return Distribution(points, name=name)


def dynamical_time(mass, radius):
    volume = (4/3) * np.pi * radius**3
    density = mass / volume
    return np.sqrt(1 / (G * density))


def estimate_softening_length(N, radius, fraction=0.1):
    volume = (4/3) * np.pi * radius**3
    number_density = N / volume

    mean_interparticle_distance = (1 / number_density) ** (1/3)

    softening_length = fraction * mean_interparticle_distance

    return softening_length
