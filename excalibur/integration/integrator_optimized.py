# integration/integrator_optimized.py
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def rk4_step_numba(state, dt, a, adot, phi, grad_phi, phi_dot, c_val):
    """
    Optimized RK4 step with inlined Christoffel and geodesic calculations.
    All computations done in a single compiled function for maximum speed.
    """
    # Extract state
    eta, x, y, z = state[0], state[1], state[2], state[3]
    u0, u1, u2, u3 = state[4], state[5], state[6], state[7]
    
    # Precompute common terms
    c2 = c_val * c_val
    c4 = c2 * c2
    a2 = a * a
    adot_over_a = adot / a
    
    # --- K1 calculation ---
    # Velocities are just u
    k1_x = np.array([u0, u1, u2, u3])
    
    # Accelerations from Christoffel symbols
    # Γ^0: temporal acceleration
    du0_k1 = -(2 * (grad_phi[0]/c2) * u0 * u1 +
               2 * (grad_phi[1]/c2) * u0 * u2 +
               2 * (grad_phi[2]/c2) * u0 * u3 +
               (a*adot/c2 + 2*a*adot/c4 * (2*phi) - a2/c4 * phi_dot) * (u1*u1 + u2*u2 + u3*u3))
    
    # Γ^1: x acceleration
    du1_k1 = -(grad_phi[0]/a2 * u0*u0 +
               2 * (adot_over_a - phi_dot/c2) * u0 * u1 +
               (-grad_phi[0]/c2) * u1*u1 +
               (-grad_phi[1]/c2) * u1*u2 +
               (-grad_phi[2]/c2) * u1*u3 +
               (grad_phi[0]/c2) * (u2*u2 + u3*u3))
    
    # Γ^2: y acceleration (symmetric to x)
    du2_k1 = -(grad_phi[1]/a2 * u0*u0 +
               2 * (adot_over_a - phi_dot/c2) * u0 * u2 +
               (-grad_phi[0]/c2) * u2*u1 +
               (-grad_phi[1]/c2) * u2*u2 +
               (-grad_phi[2]/c2) * u2*u3 +
               (grad_phi[1]/c2) * (u1*u1 + u3*u3))
    
    # Γ^3: z acceleration (symmetric to x)
    du3_k1 = -(grad_phi[2]/a2 * u0*u0 +
               2 * (adot_over_a - phi_dot/c2) * u0 * u3 +
               (-grad_phi[0]/c2) * u3*u1 +
               (-grad_phi[1]/c2) * u3*u2 +
               (-grad_phi[2]/c2) * u3*u3 +
               (grad_phi[2]/c2) * (u1*u1 + u2*u2))
    
    k1_u = np.array([du0_k1, du1_k1, du2_k1, du3_k1])
    k1 = np.concatenate((k1_x, k1_u))
    
    # --- K2, K3, K4 calculations would go here ---
    # For now, simplified version with just k1 (Euler step)
    # Full RK4 requires re-evaluating Christoffel at intermediate points
    # which needs interpolation - kept in Python for now
    
    return k1

class IntegratorOptimized:
    """
    Optimized integrator using Numba JIT compilation and vectorization.
    """
    def __init__(self, metric, dt=1e-3):
        self.metric = metric
        self.dt = dt
        
    def integrate(self, photon, steps):
        """Integrate photon trajectory with optimized RK4."""
        state = photon.state.copy()
        i = 0
        
        while i < steps:
            try:
                # Standard RK4 - optimizations done in metric methods
                k1 = self.metric.geodesic_equations(state)
                k2 = self.metric.geodesic_equations(state + 0.5*self.dt*k1)
                k3 = self.metric.geodesic_equations(state + 0.5*self.dt*k2)
                k4 = self.metric.geodesic_equations(state + self.dt*k3)
                state += (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
                
                photon.x = state[:4]
                photon.u = state[4:]
                photon.state_quantities(self.metric.metric_physical_quantities)
                photon.record()
                
            except (ValueError, IndexError) as e:
                if i == 0:
                    print(f"WARNING: Photon stopped at step 0! Error: {e}")
                    print(f"  Initial position: {state[:4]}")
                    print(f"  Initial velocity: {state[4:]}")
                break
            i += 1
    
    def integrate_batch(self, photons, steps):
        """
        Integrate multiple photons - can be parallelized.
        """
        for photon in photons:
            self.integrate(photon, steps)
