import numpy as np
from .base_metric import Metric
from excalibur.core.constants import *
from excalibur.core.coordinates import cartesian_to_spherical, spherical_to_cartesian, cartesian_velocity_to_spherical


class SchwarzschildMetric(Metric):
    """
    Métrique de Schwarzschild en coordonnées sphériques :
    ds² = -(1 - 2M/r) dt² + (1 - 2M/r)^(-1) dr² + r² dΩ²

    Must be given mass, radius, and center position in chosen coordinate system.
    Default center is at (0.5,0.5,0.5) Gpc.
    Coordinates are assumed to be given in cartesian by default, can be changed to spherical. This change only affects the reading of the position vectors,
    all computations are done in spherical coordinates, so basically you're asking the metric to convert cartesian to spherical before computing any quantity.
    ️Args:
        mass: Total mass in kg
        radius: Characteristic radius (for potential cutoff) in meters
        center: 3D position of mass center in meters (array-like)
        coords: Coordinate system of input positions ("cartesian" or "spherical")
        analytical_geodesics: Whether to use analytical geodesic equations or tensor-based ones
        free_time_geodesics: Choose between forcing the normalisation of the time geodesic at each step, or monitor the drift from the null condition if you let it integrate freely
    """
    def __init__(
        self,
        mass,
        radius,
        center=np.array([0.5, 0.5, 0.5]) * one_Gpc,
        analytical_geodesics=False,
        free_time_geodesic=True,
        coords="cartesian",  # coords = INPUT coords
    ):
        self.analytical_geodesics = analytical_geodesics
        self.free_time_geodesic = free_time_geodesic

        # NEW: explicit coordinate conventions
        self.input_coords = coords              # "cartesian" or "spherical"
        self.internal_coords = "spherical"      # ALWAYS spherical internally

        self.mass = mass
        self.radius = radius
        self.center = np.asarray(center, dtype=float)
        self.r_s = 2 * G * self.mass / c**2


    def metric_tensor(self, x):
        t, pos = x[0], x[1:]
        g = np.zeros((4,4))

        # Use SI-consistent units: the time component carries a factor c^2.
        # With signature (-,+,+,+):
        #   ds^2 = -(1 - r_s/r) c^2 dt^2 + (1 - r_s/r)^{-1} dr^2 + r^2 dΩ^2
        # where r_s = 2GM/c^2.
        r = pos[0]
        f = 1.0 - self.r_s / r

        g[0,0] = -f * c**2
        g[1,1] = 1.0 / f
        g[2,2] = pos[0]**2
        g[3,3] = (pos[0]**2) * (np.sin(pos[1])**2)

        return g

    def christoffel(self, x):
        """
        IMPORTANT: x is expected to be (t, r, theta, phi) with r measured FROM THE MASS CENTER.
        If you pass cartesian states, geodesic_equations will convert them using cartesian_state_to_spherical(),
        which already subtracts self.center.
        """
        t, r, theta, phi = x
        r = float(r)
        theta = float(theta)

        if (not np.isfinite(r)) or (not np.isfinite(theta)) or (not np.isfinite(phi)):
            raise ValueError("Non-finite (r,theta,phi) in SchwarzschildMetric")

        if r <= self.radius:
            r = self.radius
        if r <= self.r_s:
            raise ValueError("Photon inside the Schwarzschild radius, stopping integration")

        M = self.mass
        rs = self.r_s

        # Avoid division by extremely small denominators (numerical guard)
        denom = c**2 * r - 2 * G * M
        if abs(denom) < 1e-300:
            raise ValueError("Denominator too small in Christoffel computation (near horizon)")

        Γ = np.zeros((4, 4, 4), dtype=float)

        # Non-zero Christoffels for Schwarzschild (standard form)
        Γ[0, 1, 0] = G * M / (r * denom)
        Γ[0, 0, 1] = Γ[0, 1, 0]

        Γ[1, 0, 0] = G * M * (c**2 * r - 2 * G * M) / (c**2 * r**3)
        Γ[1, 1, 1] = -G * M / (r * denom)

        Γ[1, 2, 2] = -(r - rs)
        Γ[1, 3, 3] = -(r - rs) * (np.sin(theta) ** 2)

        Γ[2, 1, 2] = 1.0 / r
        Γ[2, 2, 1] = 1.0 / r
        Γ[2, 3, 3] = -np.sin(theta) * np.cos(theta)

        Γ[3, 1, 3] = 1.0 / r
        Γ[3, 3, 1] = 1.0 / r
        Γ[3, 2, 3] = np.cos(theta) / np.sin(theta)
        Γ[3, 3, 2] = Γ[3, 2, 3]

        return Γ
    
    def geodesic_equations(self, state):
        # Convert to INTERNAL spherical only if input is cartesian
        if self.input_coords == "cartesian":
            state = self.cartesian_state_to_spherical(state)
        elif self.input_coords == "spherical":
            # already spherical: (t,r,theta,phi, dt,dr,dtheta,dphi)
            pass
        else:
            raise ValueError("input_coords must be 'cartesian' or 'spherical'")

        if self.analytical_geodesics:
            return self.geodesic_equations_analytical(state)
        else:
            return self.geodesic_equations_tensor(state)

    def geodesic_equations_tensor_1(self, state):
        x, u = state[:4], state[4:]
        Γ = self.christoffel(x)
        du = np.zeros(4)
        for μ in range(4):
            du[μ] = -np.sum(Γ[μ,:,:] * np.outer(u,u))
        return np.concatenate([u, du])
    
    def geodesic_equations_tensor(self, state):
        x, u = state[:4], state[4:]
        Γ = self.christoffel(x)
        du = -np.einsum('mij,i,j->m', Γ, u, u)
        
        return np.concatenate([u, du])
    
    def geodesic_equations_analytical(self, state):
        t, r, theta, phi, dtdl, drdl, dthetadl, dphidl = state

        if r <= self.radius:
            raise ValueError("Photon inside the massive object")
        if r <= self.r_s:
            raise ValueError("Photon inside the Schwarzschild radius")

        equations = np.zeros(8)

        if self.free_time_geodesic :
            equations[4] = - self.r_s / (r * (r - self.r_s)) * dtdl * drdl
        else : 
            dtdl = - 1/c * np.sqrt((1/(1-self.r_s/r))**2 * drdl**2 + r**2 / (1 - self.r_s/r) * dthetadl**2 + r**2 * np.sin(theta)**2 / (1 - self.r_s/r) * dphidl**2)
            equations[4] = 0.0


        equations[0] = dtdl
        equations[1] = drdl
        equations[2] = dthetadl
        equations[3] = dphidl

        #equations[4] = - self.r_s / (r * (r - self.r_s)) * dtdl * drdl

        equations[5] = (
            self.r_s / (2 * r * (r - self.r_s)) * drdl**2
            - (r - self.r_s) / (2 * r**3)
            * (c**2 * self.r_s * dtdl**2
            - 2 * r**3 * (dthetadl**2 + np.sin(theta)**2 * dphidl**2))
        )

        equations[6] = (
            -2 / r * drdl * dthetadl
            + np.sin(theta) * np.cos(theta) * dphidl**2
        )

        equations[7] = (
            -2 / r * drdl * dphidl
            - 2 * np.cos(theta) / np.sin(theta) * dthetadl * dphidl
        )

        return equations


    def metric_physical_quantities(self, state):
        """ return relevant physical quantities from the metric at given state """
        t, r, theta, phi = state
        r = float(r)
        return np.array([r, self.mass, self.radius, self.r_s], dtype=float)
    
    def cartesian_state_to_spherical(self, state):
        """
        Convert a full cartesian state
        (t, x, y, z, dt, dx, dy, dz)
        to a spherical internal state
        (t, r, theta, phi, dt, dr, dtheta, dphi)
        """
        t, x, y, z, dtdl, dxdl, dydl, dzdl = state

        X = x - self.center[0]
        Y = y - self.center[1]
        Z = z - self.center[2]

        r, theta, phi = cartesian_to_spherical(X, Y, Z)
        drdl, dthetadl, dphidl = cartesian_velocity_to_spherical(
            X, Y, Z, dxdl, dydl, dzdl
        )

        return np.array([
            t, r, theta, phi,
            dtdl, drdl, dthetadl, dphidl
        ])
    