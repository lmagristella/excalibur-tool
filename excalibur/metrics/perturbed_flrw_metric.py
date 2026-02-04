# metrics/perturbed_flrw_metric.py
import numpy as np
from .base_metric import Metric
from excalibur.core.constants import *



class PerturbedFLRWMetric(Metric):
    """
    FLRW avec perturbations scalaires au premier ordre :
    ds² = a²(η)[-(1+2Ψ)dη² + (1-2Φ)δ_ij dx^i dx^j]
    """
    def __init__(self, cosmology, grid, interpolator, analytical_geodesics=False, free_time_geodesic=False):
        self.cosmology = cosmology
        self.a_of_eta = cosmology.a_of_eta
        self.adot_of_eta = cosmology.adot_of_eta  # OPTIMIZATION: Direct access to fast adot
        self.interp = interpolator
        self.analytical_geodesics = analytical_geodesics
        self.free_time_geodesic = free_time_geodesic

        # Cache for expensive interpolation calls.
        #
        # IMPORTANT: cache-hit checks must be cheap; using np.allclose/isclose in a
        # tight RK loop is often more expensive than the interpolation itself.
        # We therefore cache by a discrete key:
        #   (eta_bin, i, j, k)
        # where (i,j,k) are the grid-cell indices containing pos.
        #
        # This does NOT change the physics: we only reuse cached interpolation
        # results when the photon position stays in the same cell (common in RK4
        # sub-stages) and eta hasn't changed beyond a tiny bin.
        self._cache_key = None
        self._cache_phi_data = None
        self._eta_bin_width = 0.0  # 0 disables eta-binning; keep exact eta.

    def metric_tensor(self, x):
        if len(x) == 4:
            eta, pos = x[0], x[1:4]
        else:
            eta = 1.0
            pos = x
        a = self.a_of_eta(eta)
        phi_dim, _, _ = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)
        phi = phi_dim / (c**2)  # Dimensionless
        psi = phi  # Supposons Ψ = Φ, aucun stress anisotrope

        g = np.zeros((4,4))
        g[0,0] = -a**2 * (1 + 2*psi) * c**2
        g[1,1] = a**2 * (1 - 2*phi)
        g[2,2] = a**2 * (1 - 2*phi)
        g[3,3] = a**2 * (1 - 2*phi)

        return g

    def christoffel(self, x):
        eta, pos = x[0], x[1:]
        a = self.a_of_eta(eta)
        adot = self.adot_of_eta(eta)  

        # Cheap cache key: keep only the grid cell indices and a quantized eta.
        # This avoids millions of np.allclose/isclose calls in batch runs.
        if self._eta_bin_width and self._eta_bin_width > 0:
            eta_bin = float(np.round(eta / self._eta_bin_width) * self._eta_bin_width)
        else:
            eta_bin = float(eta)

        try:
            i, j, k = self.interp.grid.indices_from_position(pos)
            cache_key = (eta_bin, int(i), int(j), int(k))
        except Exception:
            # If indices computation fails for any reason, fall back to no-cache.
            cache_key = None

        if cache_key is not None and cache_key == self._cache_key and self._cache_phi_data is not None:
            phi, grad_phi, phi_dot = self._cache_phi_data
        else:
            phi, grad_phi, phi_dot = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)
            self._cache_key = cache_key
            self._cache_phi_data = (phi, grad_phi, phi_dot)
            
        psi, grad_psi, psi_dot = (phi, grad_phi, phi_dot) # Supposons Ψ = Φ 

        # PERFORMANCE OPTIMIZATION: Pre-calculate common terms
        c_inv = 1.0 / c
        c_inv2 = c_inv * c_inv
        c_inv4 = c_inv2 * c_inv2
        a_inv = 1.0 / a
        a_inv2 = a_inv * a_inv
        adot_over_a = adot * a_inv
        phi_plus_psi = phi + psi
        a_adot_c_inv2 = a * adot * c_inv2
        a2_phi_dot_c_inv4 = a * a * phi_dot * c_inv4

        Γ = np.zeros((4,4,4))

        # OPTIMIZED: Use pre-calculated terms
        Γ[0,0,0] = c_inv2 * psi_dot
        Γ[1,0,0] = grad_psi[0] * a_inv2
        Γ[2,0,0] = grad_psi[1] * a_inv2
        Γ[3,0,0] = grad_psi[2] * a_inv2
        Γ[1,1,1] = -c_inv2 * grad_phi[0]
        Γ[2,2,2] = -c_inv2 * grad_phi[1]
        Γ[3,3,3] = -c_inv2 * grad_phi[2]
        
        # Common term for diagonal metric components
        diag_term = a_adot_c_inv2 + 2 * a_adot_c_inv2 * c_inv2 * phi_plus_psi - a2_phi_dot_c_inv4
        Γ[0,1,1] = diag_term
        Γ[0,2,2] = diag_term  
        Γ[0,3,3] = diag_term
        
        Γ[0,0,1] = Γ[0,1,0] = grad_psi[0] * c_inv2
        Γ[0,0,2] = Γ[0,2,0] = grad_psi[1] * c_inv2
        Γ[0,0,3] = Γ[0,3,0] = grad_psi[2] * c_inv2
        
        time_mix_term = adot_over_a - phi_dot * c_inv2
        Γ[1,1,0] = Γ[1,0,1] = time_mix_term
        Γ[2,2,0] = Γ[2,0,2] = time_mix_term
        Γ[3,3,0] = Γ[3,0,3] = time_mix_term
        
        Γ[1,2,2] = Γ[1,3,3] = grad_phi[0] * c_inv2
        Γ[2,1,1] = Γ[2,3,3] = grad_phi[1] * c_inv2
        Γ[3,1,1] = Γ[3,2,2] = grad_phi[2] * c_inv2
        Γ[1,1,2] = Γ[1,2,1] = -grad_phi[1] * c_inv2
        Γ[1,1,3] = Γ[1,3,1] = -grad_phi[2] * c_inv2
        Γ[2,2,1] = Γ[2,1,2] = -grad_phi[0] * c_inv2
        Γ[2,2,3] = Γ[2,3,2] = -grad_phi[2] * c_inv2
        Γ[3,3,1] = Γ[3,1,3] = -grad_phi[0] * c_inv2
        Γ[3,3,2] = Γ[3,2,3] = -grad_phi[1] * c_inv2

        return Γ

    def geodesic_equations(self, state):
        if self.analytical_geodesics:
            return self.geodesic_equations_analytical(state)
        else: 
            return self.geodesic_equations_tensor(state)

    def geodesic_equations_tensor(self, state):
        x, u = state[:4], state[4:]
        Γ = self.christoffel(x)
        du = -np.einsum('mij,i,j->m', Γ, u, u)
        
        return np.concatenate([u, du])
    
    def geodesic_equations_analytical(self, state):
        eta,x,y,z,detadl,dxdl,dydl,dzdl = state

        a, phi, dphidx, dphidy, dphidz, dphideta = self.metric_physical_quantities(state)
        a_prime = self.adot_of_eta(eta)

        #r_squared = x**2 + y**2 + z**2
        v_squared = dxdl**2 + dydl**2 + dzdl**2
        
        #k = 0
        
        dphidlambda = dphidx*dxdl + dphidy*dydl + dphidz*dzdl

        psi = phi
        dpsideta, dpsidx, dpsidy, dpsidz = (dphideta, dphidx, dphidy, dphidz)
        
        
        dydlambda = np.zeros_like(state)
        
        if self.free_time_geodesic:
            dydlambda[4] = -(a_prime/a + dphideta/c**2) * detadl**2 - 2/c**2 * detadl * dphidlambda - (a_prime/a/c**2 - 2*a_prime/a/c**4 * (phi + psi) - dpsideta/c**4) * v_squared
        else:
            detadl = - np.sqrt((1-2*psi)/(1+2*phi)) * np.sqrt(v_squared) / c
            dydlambda[4] = 0.0 

        dydlambda[0] = detadl
        dydlambda[1] = dxdl
        dydlambda[2] = dydl
        dydlambda[3] = dzdl
        #dydlambda[4] = -(a_prime/a + dphideta/c**2) * detadl**2 - 2/c**2 * detadl * dphidlambda - (a_prime/a/c**2 - 2*a_prime/a/c**4 * (phi + psi) - dpsideta/c**4) * v_squared
        dydlambda[5] = - dphidx * detadl ** 2 - 2 * (a_prime/a - dpsideta/c**2) * detadl * dxdl + 2/c**2 * dxdl * (dpsidy * dydl + dpsidz * dzdl) + dpsidx/c**2 * (2*dxdl**2 - v_squared)
        dydlambda[6] = - dphidy * detadl ** 2 - 2 * (a_prime/a - dpsideta/c**2) * detadl * dydl + 2/c**2 * dydl * (dpsidx * dxdl + dpsidz * dzdl) + dpsidy/c**2 * (2*dydl**2 - v_squared)
        dydlambda[7] = - dphidz * detadl ** 2 - 2 * (a_prime/a - dpsideta/c**2) * detadl * dzdl + 2/c**2 * dzdl * (dpsidy * dydl + dpsidx * dxdl) + dpsidz/c**2 * (2*dzdl**2 - v_squared)
        return dydlambda
    
    def metric_physical_quantities(self, state):
        eta, pos = state[0], state[1:4]
        a = self.a_of_eta(eta)
        phi, grad_phi, phi_dot = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)

        quantities = np.array([a, phi, *grad_phi, phi_dot])
        return quantities
        
