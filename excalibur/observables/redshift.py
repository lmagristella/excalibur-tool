#!/usr/bin/env python3
"""
Redshift calculation module for perturbed FLRW cosmology.

This module computes the various contributions to the observed redshift
in a perturbed FLRW spacetime, following the approach of cosmological
perturbation theory.

Redshift components in perturbed FLRW:
---------------------------------------

The total observed redshift 1+z can be decomposed as:

    1 + z = (1 + z_H) × (1 + z_D) × (1 + z_ISW) × (1 + z_SW) × (1 + z_v)

where:

1. **Homogeneous (expansion) redshift**: z_H
   - Due to cosmic expansion: 1 + z_H = a_obs / a_em
   - Dominant term for cosmological distances
   - Computed from scale factor evolution

2. **Doppler redshift**: z_D
   - Due to peculiar velocities of source and observer
   - 1 + z_D ≈ (1 + v_obs·n) / (1 + v_em·n)
   - Typically small (v << c)

3. **Integrated Sachs-Wolfe (ISW)**: z_ISW
   - Time-varying gravitational potential along path
   - Δz_ISW = -∫ (∂Φ/∂η) dη / c²
   - Important for evolving structures

4. **Sachs-Wolfe (SW)**: z_SW
   - Gravitational potential at emission/observation
   - Δz_SW = (Φ_obs - Φ_em) / c²
   - Ordinary Sachs-Wolfe effect

5. **Velocity terms**: z_v
   - Perturbations to velocity field
   - Includes terms like ∇·v, shear, etc.

Mathematical formulation:
------------------------

In conformal time η with scale factor a(η), the redshift is:

    1 + z = (a_obs/a_em) × exp[∫_path (terms) dη]

The integral terms include:
- Potential derivatives: -∂Φ/∂η / c²  (ISW)
- Velocity gradients: -∂v^i/∂x^i  (volume expansion)
- Metric perturbations: various h_μν contributions

Implementation notes:
--------------------
- Uses backward ray tracing: photon trajectories from observer → source
- Integrates perturbations along actual deflected path
- Requires: Φ(x, η), v(x, η), a(η) along trajectory
- Output: individual components + total redshift

References:
-----------
- Sachs & Wolfe (1967): Gravitational effects on light
- Bertschinger (1995): Cosmological perturbation theory
- Dodelson (2003): Modern Cosmology, Chapter 8
"""

import numpy as np
from scipy.integrate import simpson
from excalibur.core.constants import c


class RedshiftCalculator:
    """
    Calculator for redshift components in perturbed FLRW cosmology.
    
    Computes all contributions to the observed redshift from a photon
    trajectory obtained via backward ray tracing.
    
    **NEW: Automatically extracts pre-computed quantities from trajectory if available.**
    
    Trajectory structure:
    --------------------
    If quantities are saved (recommended), trajectory has columns:
    [0:4]   η, x, y, z (spacetime position)
    [4:8]   u0, u1, u2, u3 (4-velocity)
    [8]     a (scale factor) - extracted automatically
    [9]     phi/c² (potential normalized by c²) - extracted
    [10:13] grad_phi (potential gradient in m/s²) - extracted
    [13]    phi_dot/c² (time derivative normalized by c²) - extracted
    
    Attributes:
    -----------
    trajectory : ndarray
        Photon trajectory states (n_steps, 8+) with [η, x, y, z, u0, u1, u2, u3, ...]
    has_quantities : bool
        Whether trajectory contains pre-computed quantities (a, phi, grad_phi, phi_dot)
    a_of_eta : callable or None
        Scale factor as function of conformal time: a(η) (only if no quantities)
    potential_func : callable or None
        Gravitational potential: Φ(x, y, z, η) (only if no quantities)
    velocity_func : callable or None
        Peculiar velocity field: v(x, y, z, η) → [vx, vy, vz]
    """
    
    def __init__(self, trajectory, a_of_eta=None, potential_func=None, velocity_func=None):
        """
        Initialize redshift calculator.
        
        **AUTO-DETECTION**: If trajectory has 14+ columns, assumes quantities are present
        and extracts them directly. Otherwise, requires callable functions.
        
        Parameters:
        -----------
        trajectory : ndarray (n_steps, 8+)
            Photon trajectory with [η, x, y, z, u0, u1, u2, u3, ...]
            If shape[1] >= 14, assumes [8:14] contains [a, phi, gx, gy, gz, phi_dot]
        a_of_eta : callable, optional
            Function a(η) returning scale factor (required if no quantities)
        potential_func : callable, optional
            Function Φ(x, y, z, η) or Φ(x, η) returning gravitational potential (required if no quantities)
        velocity_func : callable, optional
            Function v(x, y, z, η) returning peculiar velocity [vx, vy, vz]
        """
        self.trajectory = trajectory
        self.velocity_func = velocity_func
        
        # Check if quantities are pre-computed in trajectory
        self.has_quantities = (trajectory.shape[1] >= 14)
        
        if self.has_quantities:
            # Extract pre-computed quantities from trajectory
            self.a_array = trajectory[:, 8]           # Scale factor
            self.phi_array = trajectory[:, 9]         # Potential (normalized by c²)
            self.grad_phi_array = trajectory[:, 10:13]  # Gradient (m/s²)
            self.phi_dot_array = trajectory[:, 13]    # Time derivative (normalized by c²)
            
            # Set callables to None (not needed)
            self.a_of_eta = None
            self.potential_func = None
        else:
            # Use provided callables
            if a_of_eta is None or potential_func is None:
                raise ValueError(
                    "trajectory has no quantities (shape[1] < 14), so a_of_eta and "
                    "potential_func must be provided as callables"
                )
            self.a_of_eta = a_of_eta
            self.potential_func = potential_func
        
        # Extract key points
        self.eta_obs = trajectory[0, 0]      # Observation time (conformal)
        self.eta_em = trajectory[-1, 0]      # Emission time
        self.x_obs = trajectory[0, 1:4]      # Observer position
        self.x_em = trajectory[-1, 1:4]      # Source position
        
        # Scale factors at endpoints
        if self.has_quantities:
            self.a_obs = self.a_array[0]
            self.a_em = self.a_array[-1]
        else:
            self.a_obs = a_of_eta(self.eta_obs)
            self.a_em = a_of_eta(self.eta_em)
    
    def compute_homogeneous_redshift(self):
        """
        Compute homogeneous (expansion) redshift.
        
        This is the dominant term for cosmological distances:
            1 + z_H = a_obs / a_em
        
        Returns:
        --------
        z_H : float
            Homogeneous redshift
        """
        z_H = (self.a_obs / self.a_em) - 1.0
        return z_H
    
    def compute_potential_at_endpoints(self):
        """
        Compute gravitational potential at observation and emission points.
        
        Returns:
        --------
        Phi_obs, Phi_em : float, float
            Potential at observer and source (in SI units: m²/s²)
        """
        if self.has_quantities:
            # Use pre-computed values (already normalized by c², so multiply back)
            Phi_obs = self.phi_array[0] 
            Phi_em = self.phi_array[-1] 
        else:
            # Evaluate potential at endpoints
            Phi_obs = self.potential_func(*self.x_obs, self.eta_obs)
            Phi_em = self.potential_func(*self.x_em, self.eta_em)
        
        return Phi_obs, Phi_em
    
    def compute_sachs_wolfe_redshift(self):
        """
        Compute Sachs-Wolfe redshift contribution.
        
        Gravitational redshift from potential difference:
            Δz_SW = (Φ_obs - Φ_em) / c²
        
        Positive Φ_obs (observer in potential well) → blue shift
        Positive Φ_em (source in potential well) → red shift
        
        Returns:
        --------
        z_SW : float
            Sachs-Wolfe redshift
        Phi_obs, Phi_em : float, float
            Potentials at endpoints
        """
        Phi_obs, Phi_em = self.compute_potential_at_endpoints()
        
        # Sachs-Wolfe effect: Δz = (Φ_obs - Φ_em) / c²
        z_SW = (Phi_obs - Phi_em) / c**2
        
        return z_SW, Phi_obs, Phi_em
    
    def compute_integrated_sachs_wolfe_redshift(self):
        """
        Compute Integrated Sachs-Wolfe (ISW) redshift.
        
        This term arises from time-varying gravitational potentials
        along the photon path:
        
            Δz_ISW = -∫[path] (∂Φ/∂η) dη / c²
        
        For a decaying potential (∂Φ/∂η < 0):
            - Photon gains energy entering potential well (blue shift)
            - Potential decays while photon inside
            - Photon exits with less depth → net blue shift
        
        For growing potential: opposite (red shift)
        
        Returns:
        --------
        z_ISW : float
            Integrated Sachs-Wolfe redshift
        dPhi_deta_array : ndarray
            Time derivative of potential along path (for diagnostics)
        """
        n_steps = len(self.trajectory)
        
        # Extract conformal times along trajectory
        eta_array = self.trajectory[:, 0]
        
        if self.has_quantities:
            # Use pre-computed phi_dot (already normalized by c², so multiply by c²)
            dPhi_deta_array = self.phi_dot_array * (c**2)
        else:
            # Compute potential along trajectory
            x_array = self.trajectory[:, 1]
            y_array = self.trajectory[:, 2]
            z_array = self.trajectory[:, 3]
            
            Phi_array = np.array([
                self.potential_func(x, y, z, eta)
                for x, y, z, eta in zip(x_array, y_array, z_array, eta_array)
            ])
            
            # Compute ∂Φ/∂η using finite differences
            dPhi_deta_array = np.gradient(Phi_array, eta_array)
        
        # Integrate: -∫ (∂Φ/∂η) dη / c²
        # Use trapezoidal rule for integration
        z_ISW = -simpson(dPhi_deta_array, x=eta_array) / c**2
        
        return z_ISW, dPhi_deta_array
    
    def compute_doppler_redshift(self):
        """
        Compute Doppler redshift from peculiar velocities.
        
        Due to peculiar motion of source and observer relative to
        the Hubble flow:
        
            1 + z_D = (1 + v_obs · n̂) / (1 + v_em · n̂)
        
        where n̂ is the direction of propagation (spatial).
        
        For small velocities (v << c):
            z_D ≈ (v_obs - v_em) · n̂ / c
        
        Returns:
        --------
        z_D : float
            Doppler redshift
        v_obs, v_em : ndarray, ndarray
            Peculiar velocities at observer and source
        """
        if self.velocity_func is None:
            # No velocity field provided
            return 0.0, np.zeros(3), np.zeros(3)
        
        # Evaluate peculiar velocities at endpoints
        v_obs = self.velocity_func(*self.x_obs, self.eta_obs)
        v_em = self.velocity_func(*self.x_em, self.eta_em)
        
        # Direction of photon propagation (unit vector)
        # From observer to source (backward tracing goes emission → observer)
        displacement = self.x_em - self.x_obs
        n_hat = displacement / np.linalg.norm(displacement)
        
        # Doppler formula (first-order in v/c)
        # Positive v_obs·n → observer moving away from source → red shift
        # Positive v_em·n → source moving away from observer → red shift
        z_D = (np.dot(v_obs, n_hat) - np.dot(v_em, n_hat)) / c
        
        return z_D, v_obs, v_em
    
    def compute_velocity_gradient_redshift(self):
        """
        Compute redshift from velocity field gradients along path.
        
        Volume expansion and shear in the velocity field contribute:
            Δz_v ≈ -∫[path] (∇·v) dη
        
        This term is usually subdominant but can be important for
        rapidly evolving structures.
        
        Returns:
        --------
        z_v : float
            Velocity gradient contribution
        """
        if self.velocity_func is None:
            return 0.0
        
        # TODO: Implement if velocity gradient data available
        # Requires computing ∇·v along trajectory
        # For now, return zero (subdominant term)
        
        return 0.0
    
    def compute_all_components(self, verbose=False):
        """
        Compute all redshift components.
        
        Returns a dictionary with individual contributions and total redshift.
        
        Parameters:
        -----------
        verbose : bool
            If True, print detailed breakdown
        
        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'z_H': homogeneous (expansion) redshift
            - 'z_SW': Sachs-Wolfe redshift
            - 'z_ISW': Integrated Sachs-Wolfe redshift
            - 'z_D': Doppler redshift
            - 'z_v': velocity gradient redshift
            - 'z_total': total redshift (approximate sum for small perturbations)
            - 'z_total_exact': exact product formula
            - Additional diagnostic data
        """
        results = {}
        
        # 1. Homogeneous redshift (dominant)
        z_H = self.compute_homogeneous_redshift()
        results['z_H'] = z_H
        
        # 2. Sachs-Wolfe
        z_SW, Phi_obs, Phi_em = self.compute_sachs_wolfe_redshift()
        results['z_SW'] = z_SW
        results['Phi_obs'] = Phi_obs
        results['Phi_em'] = Phi_em
        
        # 3. Integrated Sachs-Wolfe
        z_ISW, dPhi_deta = self.compute_integrated_sachs_wolfe_redshift()
        results['z_ISW'] = z_ISW
        results['dPhi_deta_array'] = dPhi_deta
        
        # 4. Doppler
        z_D, v_obs, v_em = self.compute_doppler_redshift()
        results['z_D'] = z_D
        results['v_obs'] = v_obs
        results['v_em'] = v_em
        
        # 5. Velocity gradients (subdominant, often neglected)
        z_v = self.compute_velocity_gradient_redshift()
        results['z_v'] = z_v
        
        # Total redshift
        # For small perturbations: z_total ≈ z_H + z_SW + z_ISW + z_D + z_v
        z_total_linear = z_H + z_SW + z_ISW + z_D + z_v
        results['z_total'] = z_total_linear
        
        # Exact multiplicative formula: 1+z = (1+z_H)(1+z_SW)(1+z_ISW)(1+z_D)(1+z_v)
        z_total_exact = ((1 + z_H) * (1 + z_SW) * (1 + z_ISW) * 
                        (1 + z_D) * (1 + z_v)) - 1.0
        results['z_total_exact'] = z_total_exact
        
        # Scale factors
        results['a_obs'] = self.a_obs
        results['a_em'] = self.a_em
        
        if verbose:
            print("\n" + "="*70)
            print("REDSHIFT DECOMPOSITION")
            print("="*70)
            print(f"Scale factors:")
            print(f"  a(η_obs) = {self.a_obs:.6f}")
            print(f"  a(η_em)  = {self.a_em:.6f}")
            print(f"\nRedshift components:")
            print(f"  z_H   (expansion)    = {z_H:.6e}  ({z_H/(z_total_linear+1e-20)*100:6.2f}%)")
            print(f"  z_SW  (Sachs-Wolfe)  = {z_SW:.6e}  ({z_SW/(z_total_linear+1e-20)*100:6.2f}%)")
            print(f"  z_ISW (Integrated)   = {z_ISW:.6e}  ({z_ISW/(z_total_linear+1e-20)*100:6.2f}%)")
            print(f"  z_D   (Doppler)      = {z_D:.6e}  ({z_D/(z_total_linear+1e-20)*100:6.2f}%)")
            print(f"  z_v   (velocity∇)    = {z_v:.6e}  ({z_v/(z_total_linear+1e-20)*100:6.2f}%)")
            print(f"\nTotal redshift:")
            print(f"  z (linear sum)       = {z_total_linear:.6e}")
            print(f"  z (exact product)    = {z_total_exact:.6e}")
            print(f"  Difference           = {abs(z_total_exact - z_total_linear):.6e}")
            print("="*70)
        
        return results


def compute_redshift_components(trajectory, a_of_eta, potential_func, 
                                velocity_func=None, verbose=False):
    """
    Convenience function to compute all redshift components.
    
    Parameters:
    -----------
    trajectory : ndarray
        Photon trajectory (n_steps, 8+)
    a_of_eta : callable
        Scale factor function a(η)
    potential_func : callable
        Gravitational potential Φ(x, y, z, η)
    velocity_func : callable, optional
        Peculiar velocity v(x, y, z, η)
    verbose : bool
        Print detailed breakdown
    
    Returns:
    --------
    results : dict
        Dictionary with all redshift components
    """
    calc = RedshiftCalculator(trajectory, a_of_eta, potential_func, velocity_func)
    return calc.compute_all_components(verbose=verbose)


def compute_total_redshift(trajectory, a_of_eta, potential_func, 
                           velocity_func=None):
    """
    Compute total observed redshift (simplified interface).
    
    Returns only the total redshift value.
    
    Parameters:
    -----------
    trajectory : ndarray
        Photon trajectory
    a_of_eta : callable
        Scale factor a(η)
    potential_func : callable
        Potential Φ(x, y, z, η)
    velocity_func : callable, optional
        Velocity v(x, y, z, η)
    
    Returns:
    --------
    z_total : float
        Total observed redshift
    """
    results = compute_redshift_components(
        trajectory, a_of_eta, potential_func, velocity_func, verbose=False
    )
    return results['z_total']
