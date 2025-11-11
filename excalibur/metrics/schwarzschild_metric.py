import numpy as np
from .base_metric import Metric
from excalibur.core.constants import *


class SchwarzschildMetric(Metric):
    """
    Métrique de Schwarzschild en coordonnées sphériques :
    ds² = -(1 - 2M/r) dt² + (1 - 2M/r)^(-1) dr² + r² dΩ²
    """
    def __init__(self, mass, radius, center = (0.5,0.5,0.5)):
        self.mass = mass
        self.radius = radius
        self.center = center
        self.schwarzschild_radius = 2*G*self.mass/c**2

    def metric_tensor(self, x):
        t, pos = x[0], x[1:]
        g = np.zeros((4,4))

        g[0,0] = -(1-2*G*self.mass/pos[0])
        g[1,1] = 1/(1-2*G*self.mass/pos[0])
        g[2,2] = pos[0]**2
        g[3,3] = (pos[0]**2) * (np.sin(pos[1])**2)

        return g

    def christoffel(self, x):
        t, r_origin, theta_origin, phi_origin = x

        cartesian_origin_position = r_origin * np.array([
            np.sin(theta_origin) * np.cos(phi_origin),
            np.sin(theta_origin) * np.sin(phi_origin),
            np.cos(theta_origin)
        ])
        
        shifted_cartesian_position = cartesian_origin_position - np.array(self.center)

        r = np.linalg.norm(shifted_cartesian_position)
        theta = np.arccos(shifted_cartesian_position[2] / r)
        phi = np.arctan2(shifted_cartesian_position[1], shifted_cartesian_position[0])

        M = self.mass
        Γ = np.zeros((4,4,4))

        if r <= self.radius :
            raise ValueError("Photon inside the massive object, not described by Schwarzschild metric")
        if r <= self.schwarzschild_radius :
            raise ValueError("Photon inside the Schwarzschild radius, stopping integration")

        # Composantes non nulles des symboles de Christoffel
        Γ[0,1,0] = G*M / (r * (c**2 * r - 2*G*M))
        Γ[0,0,1] = G*M / (r * (c**2 * r - 2*G*M))
        Γ[1,0,0] = G*M * (c**2 * r - 2*G*M) / (c**2 * r**3)
        Γ[1,1,1] = -G*M / (r * (c**2 * r - 2*G*M))
        Γ[1,2,2] = -(r - 2*G*M/c**2)
        Γ[1,3,3] = -(r - 2*G*M/c**2) * np.sin(theta)**2
        Γ[2,1,2] = 1 / r
        Γ[2,2,1] = 1 / r        
        Γ[2,3,3] = -np.sin(theta) * np.cos(theta)
        Γ[3,1,3] = 1 / r
        Γ[3,3,1] = 1 / r
        Γ[3,2,3] = np.cos(theta) / np.sin(theta)
        Γ[3,3,2] = np.cos(theta) / np.sin(theta)
        return Γ
    
    def geodesic_equations(self, state):
        x, u = state[:4], state[4:]
        Γ = self.christoffel(x)
        du = np.zeros(4)
        for μ in range(4):
            du[μ] = -np.sum(Γ[μ,:,:] * np.outer(u,u))
        return np.concatenate([u, du])
    
    def metric_physical_quantities(self, state):
        """ return relevant physical quantities from the metric at given state """
        t, r_origin, theta_origin, phi_origin = state

        cartesian_origin_position = r_origin * np.array([
            np.sin(theta_origin) * np.cos(phi_origin),
            np.sin(theta_origin) * np.sin(phi_origin),
            np.cos(theta_origin)
        ])
        
        shifted_cartesian_position = cartesian_origin_position - np.array(self.center)

        r = np.linalg.norm(shifted_cartesian_position)

        quantities = np.array([r, self.mass, self.radius, self.schwarzschild_radius])
        return quantities
    