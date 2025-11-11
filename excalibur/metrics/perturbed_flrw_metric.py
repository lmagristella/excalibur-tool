# metrics/perturbed_flrw_metric.py
import numpy as np
from .base_metric import Metric
from excalibur.core.constants import *



class PerturbedFLRWMetric(Metric):
    """
    FLRW avec perturbations scalaires au premier ordre :
    ds² = a²(η)[-(1+2Ψ)dη² + (1-2Φ)δ_ij dx^i dx^j]
    """
    def __init__(self, a_of_eta, grid, interpolator):
        self.a_of_eta = a_of_eta
        self.grid = grid
        self.interp = interpolator

    def metric_tensor(self, x):
        eta, pos = x[0], x[1:]
        a = self.a_of_eta(eta)
        phi, _, _ = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)
        psi = phi/c**2  # Supposons Ψ = Φ 

        g = np.zeros((4,4))
        g[0,0] = -a**2 * (1 + 2*psi) / c**2
        g[1,1] = a**2 * (1 - 2*phi)
        g[2,2] = a**2 * (1 - 2*phi)
        g[3,3] = a**2 * (1 - 2*phi)

        return g

    def christoffel(self, x):
        eta, pos = x[0], x[1:]
        a = self.a_of_eta(eta)
        adot = (self.a_of_eta(eta + 1e-5) - self.a_of_eta(eta - 1e-5)) / 2e-5
        phi, grad_phi, phi_dot = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)
        psi, grad_psi, psi_dot = (phi/c**2, grad_phi/c**2, phi_dot/c**2) # Supposons Ψ = Φ 

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

    def geodesic_equations(self, state):
        x, u = state[:4], state[4:]
        Γ = self.christoffel(x)
        du = np.zeros(4)
        for μ in range(4): 
            du[μ] = -np.einsum('ij,i,j->', Γ[μ,:,:], u, u)
        return np.concatenate([u, du])
    
    def metric_physical_quantities(self, state):
        eta, pos = state[0], state[1:4]
        a = self.a_of_eta(eta)
        phi, grad_phi, phi_dot = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)

        quantities = np.array([a, phi, *grad_phi, phi_dot])
        return quantities
        
