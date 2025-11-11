# metrics/perturbed_flrw_metric_fast.py
import numpy as np
from .base_metric import Metric
from excalibur.core.constants import *
from numba import njit

@njit(cache=True, fastmath=True)
def compute_geodesic_acceleration(u0, u1, u2, u3, a, adot, phi, grad_phi_x, grad_phi_y, grad_phi_z, phi_dot, c_val):
    """
    Optimized geodesic acceleration calculation.
    Inlines all Christoffel symbol calculations.
    
    Args:
        u0, u1, u2, u3: 4-velocity components
        a, adot: scale factor and its derivative
        phi: potential value (normalized by c^2, dimensionless)
        grad_phi_x, grad_phi_y, grad_phi_z: potential gradient in m/s^2 (NOT normalized!)
        phi_dot: time derivative of potential (normalized by c^2)
        c_val: speed of light
    """
    c2 = c_val * c_val
    c4 = c2 * c2
    a2 = a * a
    adot_a = adot / a
    
    # Common terms
    u_spatial_sq = u1*u1 + u2*u2 + u3*u3
    
    # For FLRW perturbed metric, assuming psi = phi (Newtonian gauge)
    psi = phi
    
    # du0/dlambda (temporal acceleration)
    Gamma_000_term = 0.0  # psi_dot is zero for static field
    # Corrected: use phi (potential value), not grad_phi!
    Gamma_0ij_term = (a*adot/c2 + 2*a*adot/c4 * (phi + psi) - a2/c4 * phi_dot) * u_spatial_sq
    # grad_psi in m/s^2, so normalize by c^2 to make dimensionless
    Gamma_00i_term = 2 * ((grad_phi_x/c2) * u0 * u1 +
                          (grad_phi_y/c2) * u0 * u2 +
                          (grad_phi_z/c2) * u0 * u3)
    
    du0 = -(Gamma_000_term + Gamma_0ij_term + Gamma_00i_term)
    
    # du1/dlambda (x acceleration)
    # Gamma^1_00 = grad_psi_x / a^2, grad in m/s^2, normalize by c^2
    Gamma_100_term = (grad_phi_x / c2) / a2 * u0*u0
    Gamma_10i_term = 2 * (adot_a - phi_dot/c2) * u0 * u1
    # grad_phi in m/s^2, normalize by c^2 for dimensionless
    Gamma_1ii_term = (-(grad_phi_x/c2) * u1*u1 +
                      (grad_phi_x/c2) * (u2*u2 + u3*u3) +
                      (-(grad_phi_y/c2)) * u1*u2 +
                      (-(grad_phi_z/c2)) * u1*u3)
    
    du1 = -(Gamma_100_term + Gamma_10i_term + Gamma_1ii_term)
    
    # du2/dlambda (y acceleration) - symmetric
    Gamma_200_term = (grad_phi_y / c2) / a2 * u0*u0
    Gamma_20i_term = 2 * (adot_a - phi_dot/c2) * u0 * u2
    Gamma_2ii_term = (-(grad_phi_y/c2) * u2*u2 +
                      (grad_phi_y/c2) * (u1*u1 + u3*u3) +
                      (-(grad_phi_x/c2)) * u2*u1 +
                      (-(grad_phi_z/c2)) * u2*u3)
    
    du2 = -(Gamma_200_term + Gamma_20i_term + Gamma_2ii_term)
    
    # du3/dlambda (z acceleration) - symmetric
    Gamma_300_term = (grad_phi_z / c2) / a2 * u0*u0
    Gamma_30i_term = 2 * (adot_a - phi_dot/c2) * u0 * u3
    Gamma_3ii_term = (-(grad_phi_z/c2) * u3*u3 +
                      (grad_phi_z/c2) * (u1*u1 + u2*u2) +
                      (-(grad_phi_x/c2)) * u3*u1 +
                      (-(grad_phi_y/c2)) * u3*u2)
    
    du3 = -(Gamma_300_term + Gamma_30i_term + Gamma_3ii_term)
    
    return du0, du1, du2, du3
    Gamma_30i_term = 2 * (adot_a - phi_dot/c2) * u0 * u3
    Gamma_3ii_term = (-grad_phi_z/c2 * u3*u3 +
                      grad_phi_z/c2 * (u1*u1 + u2*u2) +
                      (-grad_phi_x/c2) * u3*u1 +
                      (-grad_phi_y/c2) * u3*u2)
    
    du3 = -(Gamma_300_term + Gamma_30i_term + Gamma_3ii_term)
    
    return du0, du1, du2, du3


class PerturbedFLRWMetricFast(Metric):
    """
    Optimized FLRW metric with perturbations.
    Uses Numba-compiled geodesic calculations and caching.
    """
    def __init__(self, a_of_eta, grid, interpolator):
        self.a_of_eta = a_of_eta
        self.grid = grid
        self.interp = interpolator
        
        # Cache for a(eta) and adot to avoid repeated interpolation calls
        self._eta_cache = None
        self._a_cache = None
        self._adot_cache = None
    
    def metric_tensor(self, x):
        """Metric tensor - kept for compatibility, rarely used in integration."""
        eta, pos = x[0], x[1:]
        a, _ = self._get_scale_factor_and_derivative(eta)
        
        # Get potential in SI units, then normalize by c²
        phi_SI, _, _ = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)
        phi = phi_SI / (c**2)  # Dimensionless
        psi = phi

        g = np.zeros((4,4))
        g[0,0] = -a**2 * (1 + 2*psi) * (c**2)
        g[1,1] = a**2 * (1 - 2*phi)
        g[2,2] = a**2 * (1 - 2*phi)
        g[3,3] = a**2 * (1 - 2*phi)
        return g
    
    def christoffel(self, x):
        """
        Christoffel symbols - kept for compatibility.
        Not used in optimized geodesic_equations.
        """
        eta, pos = x[0], x[1:]
        a, adot = self._get_scale_factor_and_derivative(eta)
        phi, grad_phi, phi_dot = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)
        psi, grad_psi, psi_dot = (phi/c**2, grad_phi/c**2, phi_dot/c**2)

        Γ = np.zeros((4,4,4))
        Γ[0,0,0] = 1/c**2 * psi_dot
        Γ[1,0,0] = grad_psi[0] / a**2
        Γ[2,0,0] = grad_psi[1] / a**2
        Γ[3,0,0] = grad_psi[2] / a**2
        Γ[1,1,1] = - 1/c**2 * grad_phi[0]
        Γ[2,2,2] = - 1/c**2 * grad_phi[1]
        Γ[3,3,3] = - 1/c**2 * grad_phi[2]
        Γ[0,1,1] = a*adot/c**2 + 2*a*adot/c**4 * (phi + psi) - a**2/c**4 * phi_dot
        Γ[0,2,2] = a*adot/c**2 + 2*a*adot/c**4 * (phi + psi) - a**2/c**4 * phi_dot
        Γ[0,3,3] = a*adot/c**2 + 2*a*adot/c**4 * (phi + psi) - a**2/c**4 * phi_dot
        Γ[0,0,1] = Γ[0,1,0] = grad_psi[0] / c**2
        Γ[0,0,2] = Γ[0,2,0] = grad_psi[1] / c**2
        Γ[0,0,3] = Γ[0,3,0] = grad_psi[2] / c**2
        Γ[1,1,0] = Γ[1,0,1] = adot/a - phi_dot / c**2
        Γ[2,2,0] = Γ[2,0,2] = adot/a - phi_dot / c**2
        Γ[3,3,0] = Γ[3,0,3] = adot/a - phi_dot / c**2
        Γ[1,2,2] = Γ[1,3,3] = grad_phi[0] / c**2
        Γ[2,1,1] = Γ[2,3,3] = grad_phi[1] / c**2
        Γ[3,1,1] = Γ[3,2,2] = grad_phi[2] / c**2
        Γ[1,1,2] = Γ[1,2,1] = - grad_phi[1] / c**2
        Γ[1,1,3] = Γ[1,3,1] = - grad_phi[2] / c**2
        Γ[2,2,1] = Γ[2,1,2] = - grad_phi[0] / c**2
        Γ[2,2,3] = Γ[2,3,2] = - grad_phi[2] / c**2
        Γ[3,3,1] = Γ[3,1,3] = - grad_phi[0] / c**2
        Γ[3,3,2] = Γ[3,2,3] = - grad_phi[1] / c**2
        return Γ
        
    def _get_scale_factor_and_derivative(self, eta):
        """Get a(eta) and adot with caching."""
        if self._eta_cache == eta:
            return self._a_cache, self._adot_cache
        
        a = self.a_of_eta(eta)
        # Compute derivative with finite difference
        dt = 1e-5
        adot = (self.a_of_eta(eta + dt) - self.a_of_eta(eta - dt)) / (2 * dt)
        
        # Update cache
        self._eta_cache = eta
        self._a_cache = a
        self._adot_cache = adot
        
        return a, adot

    def geodesic_equations(self, state):
        """
        Optimized geodesic equations using Numba-compiled acceleration calculation.
        """
        x, u = state[:4], state[4:]
        eta, pos = x[0], x[1:]
        
        # Get scale factor (with caching)
        a, adot = self._get_scale_factor_and_derivative(eta)
        
        # Get potential and gradient (from fast interpolator) - in SI units
        phi_SI, grad_phi_SI, phi_dot_SI = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)
        
        # Normalize potential and time derivative by c² (to make dimensionless)
        # But gradient has units m/s^2 (acceleration), keep in SI!
        phi_normalized = phi_SI / (c**2)
        grad_phi_SI_vec = grad_phi_SI  # Keep in m/s^2
        phi_dot_normalized = phi_dot_SI / (c**2)
        
        # Compute accelerations with Numba
        du0, du1, du2, du3 = compute_geodesic_acceleration(
            u[0], u[1], u[2], u[3],
            a, adot,
            phi_normalized,  # dimensionless
            grad_phi_SI_vec[0], grad_phi_SI_vec[1], grad_phi_SI_vec[2],  # m/s^2
            phi_dot_normalized,  # dimensionless
            c
        )
        
        # Return velocities and accelerations
        return np.concatenate([u, np.array([du0, du1, du2, du3])])
    
    def metric_physical_quantities(self, state):
        """Get physical quantities for recording."""
        eta, pos = state[0], state[1:4]
        a, _ = self._get_scale_factor_and_derivative(eta)
        phi, grad_phi, phi_dot = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)
        
        quantities = np.array([a, phi, *grad_phi, phi_dot])
        return quantities
