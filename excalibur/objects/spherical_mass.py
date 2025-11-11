import numpy as np
import matplotlib.pyplot as plt

from excalibur.core.constants import *

class spherical_mass:
    def __init__(self, mass, radius, center=np.array([0.5, 0.5, 0.5])):
        """
        Initialise une masse sphérique uniforme.
        """
        self.G = G
        self.mass = mass
        self.radius = radius
        self.center = center
        self.x0, self.y0, self.z0 = self.center

    def contains(self, x, y, z):
        distance_squared = (x - self.x0)**2 + (y - self.y0)**2 + (z - self.z0)**2
        return distance_squared <= self.radius**2

    def density(self, x, y, z):
        distance_squared = (x - self.x0)**2 + (y - self.y0)**2 + (z - self.z0)**2
        rho_uniforme = self.mass / (4/3 * np.pi * self.radius**3)
        return np.where(distance_squared <= self.radius**2, rho_uniforme, 0.0)
    
    def potential(self, x, y, z):
        # Accept either 1D coordinate arrays (x,y,z) -> build a 3D grid,
        # or already-broadcast/batched coordinate arrays (e.g. X,Y,Z of shape (nx,ny,nz)).
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        if x.ndim == 1 and y.ndim == 1 and z.ndim == 1:
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        else:
            # Broadcast arrays to a common shape (works for scalars, 1D, or 3D inputs)
            X, Y, Z = np.broadcast_arrays(x, y, z)

        r = np.sqrt((X - self.x0)**2 + (Y - self.y0)**2 + (Z - self.z0)**2)
        r = np.where(r == 0, 1e-10, r)

        potential = np.empty_like(r, dtype=float)
        inside = r < self.radius
        potential[inside] = -self.G * self.mass * (3 * self.radius**2 - r[inside]**2) / (2 * self.radius**3)
        potential[~inside] = -self.G * self.mass / r[~inside]
        return potential

    def mass_enclosed(self, x, y, z):
        """
        Calcule la masse contenue jusqu'à r(x, y, z).
        """
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        if x.ndim == 1 and y.ndim == 1 and z.ndim == 1:
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        else:
            X, Y, Z = np.broadcast_arrays(x, y, z)

        r = np.sqrt((X - self.x0)**2 + (Y - self.y0)**2 + (Z - self.z0)**2)

        mass_r = np.empty_like(r, dtype=float)
        inside = r <= self.radius
        mass_r[inside] = self.mass * (r[inside] / self.radius)**3
        mass_r[~inside] = self.mass
        return mass_r

    def plot_profiles(self, rmin=1e-3, rmax=None, log=True, N=500):
        """
        Trace les profils radiaux de densité et de potentiel.
        """
        if rmax is None:
            rmax = 2 * self.radius

        r = np.logspace(np.log10(rmin), np.log10(rmax), N) if log else np.linspace(rmin, rmax, N)

        x = r + self.x0
        y = np.full_like(r, self.y0)
        z = np.full_like(r, self.z0)

        rho = self.density(x, y, z)
        phi = self.potential(x, y, z)

        plt.figure(figsize=(10, 4.5))

        plt.subplot(1, 2, 1)
        plt.plot(r, rho)
        plt.xscale('log' if log else 'linear')
        plt.yscale('log')
        plt.xlabel("r")
        plt.ylabel("Densité ρ(r)")
        plt.title("Profil de densité (uniform)")

        plt.subplot(1, 2, 2)
        plt.plot(r, phi)
        plt.xscale('log' if log else 'linear')
        plt.xlabel("r")
        plt.ylabel("Potentiel Φ(r)")
        plt.title("Potentiel gravitationnel (uniform)")

        plt.tight_layout()
        plt.show()


class spherical_mass_nfw:
    def __init__(self, mass, radius, concentration, center=np.array([0.5, 0.5, 0.5])):
        """
        Initialise une masse sphérique avec profil NFW.
        """
        self.G = G
        self.mass = mass
        self.radius = radius
        self.center = center
        self.x0, self.y0, self.z0 = self.center
        self.concentration = concentration
        self.Rs = self.radius / self.concentration
        self.rho0 = self.compute_rho0_nfw()

    def compute_rho0_nfw(self):
        c = self.concentration
        Rs = self.Rs
        factor = np.log(1 + c) - c / (1 + c)
        rho0 = self.mass / (4 * np.pi * Rs**3 * factor)
        return rho0

    def contains(self, x, y, z):
        distance_squared = (x - self.x0)**2 + (y - self.y0)**2 + (z - self.z0)**2
        return distance_squared <= self.radius**2

    def density(self, x, y, z):
        r = np.sqrt((x - self.x0)**2 + (y - self.y0)**2 + (z - self.z0)**2)
        r = np.where(r == 0, 1e-10, r)
        return self.rho0 / ((r / self.Rs) * (1 + r / self.Rs)**2)

    def potential(self, x, y, z):
        r = np.sqrt((x - self.x0)**2 + (y - self.y0)**2 + (z - self.z0)**2)
        r = np.where(r == 0, 1e-10, r)

        xrel = r / self.Rs
        factor = np.log(1 + xrel) / xrel
        return -4 * np.pi * self.G * self.rho0 * self.Rs**2 * factor

    def mass_enclosed(self, x, y, z):
        """
        Calcule la masse contenue jusqu'à r(x, y, z).
        """
        r = np.sqrt((x - self.x0)**2 + (y - self.y0)**2 + (z - self.z0)**2)

        xrel = r / self.Rs
        factor = np.log(1 + xrel) - xrel / (1 + xrel)
        return 4 * np.pi * self.rho0 * self.Rs**3 * factor

    def plot_profiles(self, rmin=1e-3, rmax=None, log=True, N=500):
        """
        Trace les profils radiaux de densité et de potentiel.
        """
        if rmax is None:
            rmax = 2 * self.radius

        r = np.logspace(np.log10(rmin), np.log10(rmax), N) if log else np.linspace(rmin, rmax, N)

        x = r + self.x0
        y = np.full_like(r, self.y0)
        z = np.full_like(r, self.z0)

        rho = self.density(x, y, z)
        phi = self.potential(x, y, z)

        plt.figure(figsize=(10, 4.5))

        plt.subplot(1, 2, 1)
        plt.plot(r, rho)
        plt.xscale('log' if log else 'linear')
        plt.yscale('log')
        plt.xlabel("r")
        plt.ylabel("Densité ρ(r)")
        plt.title("Profil de densité (NFW)")

        plt.subplot(1, 2, 2)
        plt.plot(r, phi)
        plt.xscale('log' if log else 'linear')
        plt.xlabel("r")
        plt.ylabel("Potentiel Φ(r)")
        plt.title("Potentiel gravitationnel (NFW)")

        plt.tight_layout()
        plt.show()

