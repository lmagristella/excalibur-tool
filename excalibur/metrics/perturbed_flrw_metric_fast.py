# metrics/perturbed_flrw_metric_fast.py
import numpy as np
from .base_metric import Metric
from excalibur.core.constants import *
from numba import njit

from excalibur.observables.riemann_perturbed_flrw import riemann_blocks_kernel
from excalibur.observables.sachs_basis import screen_projected_sachs_transport_rhs
from excalibur.observables.optical_tidal_matrix import (
    optical_tidal_matrix_for_local_screen_convention,
    optical_tidal_matrix_optimized,
    jacobi_rhs,
)

@njit(cache=True, fastmath=True)
def compute_tensorial_acceleration(u0, u1, u2, u3, a, adot, phi, grad_phi_x, grad_phi_y, grad_phi_z, phi_dot, c_val):
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
    # Christoffel pieces (matching compute_analytical_acceleration):
    #   Gamma^0_00  = adot_a + phi_dot/c^2     (FLRW background + perturbation)
    #   Gamma^0_ij  = delta_ij * [adot_a/c^2 - 2 adot_a (phi+psi)/c^4 - phi_dot/c^4]
    #   Gamma^0_0i  = grad_psi_i / c^2
    # The previous implementation lacked the FLRW background term in
    # Gamma^0_00 and used a*adot in place of adot/a, which violated the
    # null condition by ~46% over a Gpc-scale path.
    Gamma_000_term = (adot_a + phi_dot / c2) * u0 * u0
    Gamma_0ij_term = (adot_a / c2 - 2.0 * adot_a / c4 * (phi + psi) - phi_dot / c4) * u_spatial_sq
    Gamma_00i_term = 2 * ((grad_phi_x/c2) * u0 * u1 +
                          (grad_phi_y/c2) * u0 * u2 +
                          (grad_phi_z/c2) * u0 * u3)

    du0 = -(Gamma_000_term + Gamma_0ij_term + Gamma_00i_term)
    
    # du1/dlambda (x acceleration)
    # Gamma^1_00 = grad_psi_x / a^2, grad in m/s^2, normalize by c^2
    Gamma_100_term = (grad_phi_x ) / a2 * u0*u0
    Gamma_10i_term = 2 * (adot_a - phi_dot/c2) * u0 * u1
    # grad_phi in m/s^2, normalize by c^2 for dimensionless
    Gamma_1ii_term = (-(grad_phi_x/c2) * u1*u1 +
                      (grad_phi_x/c2) * (u2*u2 + u3*u3) +
                      (-(grad_phi_y/c2)) * u1*u2 +
                      (-(grad_phi_z/c2)) * u1*u3)
    
    du1 = -(Gamma_100_term + Gamma_10i_term + Gamma_1ii_term)
    
    # du2/dlambda (y acceleration) - symmetric
    Gamma_200_term = (grad_phi_y) / a2 * u0*u0
    Gamma_20i_term = 2 * (adot_a - phi_dot/c2) * u0 * u2
    Gamma_2ii_term = (-(grad_phi_y/c2) * u2*u2 +
                      (grad_phi_y/c2) * (u1*u1 + u3*u3) +
                      (-(grad_phi_x/c2)) * u2*u1 +
                      (-(grad_phi_z/c2)) * u2*u3)
    
    du2 = -(Gamma_200_term + Gamma_20i_term + Gamma_2ii_term)
    
    # du3/dlambda (z acceleration) - symmetric
    Gamma_300_term = (grad_phi_z) / a2 * u0*u0
    Gamma_30i_term = 2 * (adot_a - phi_dot/c2) * u0 * u3
    Gamma_3ii_term = (-(grad_phi_z/c2) * u3*u3 +
                      (grad_phi_z/c2) * (u1*u1 + u2*u2) +
                      (-(grad_phi_x/c2)) * u3*u1 +
                      (-(grad_phi_y/c2)) * u3*u2)
    
    du3 = -(Gamma_300_term + Gamma_30i_term + Gamma_3ii_term)
    
    return du0, du1, du2, du3

@njit(cache=True, fastmath=True)
def compute_analytical_acceleration(u0, u1, u2, u3, a, adot, phi, dphidx, dphidy, dphidz, dphideta, c_val):
    detadl,dxdl,dydl,dzdl = u0, u1, u2, u3

    a_prime = adot
    c = c_val
    #r_squared = x**2 + y**2 + z**2
    v_squared = dxdl**2 + dydl**2 + dzdl**2
    
    #k = 0
    
    dphidlambda = dphidx*dxdl + dphidy*dydl + dphidz*dzdl

    psi = phi
    dpsideta, dpsidx, dpsidy, dpsidz = (dphideta, dphidx, dphidy, dphidz)
    
    
    du0 = -(a_prime/a + dphideta/c**2) * detadl**2 - 2/c**2 * detadl * dphidlambda - (a_prime/a/c**2 - 2*a_prime/a/c**4 * (phi + psi) - dpsideta/c**4) * v_squared
    du1 = - dphidx * detadl ** 2 - 2 * (a_prime/a - dpsideta/c**2) * detadl * dxdl + 2/c**2 * dxdl * (dpsidy * dydl + dpsidz * dzdl) + dpsidx/c**2 * (2*dxdl**2 - v_squared)
    du2 = - dphidy * detadl ** 2 - 2 * (a_prime/a - dpsideta/c**2) * detadl * dydl + 2/c**2 * dydl * (dpsidx * dxdl + dpsidz * dzdl) + dpsidy/c**2 * (2*dydl**2 - v_squared)
    du3 = - dphidz * detadl ** 2 - 2 * (a_prime/a - dpsideta/c**2) * detadl * dzdl + 2/c**2 * dzdl * (dpsidy * dydl + dpsidx * dxdl) + dpsidz/c**2 * (2*dzdl**2 - v_squared)
    return du0, du1, du2, du3


class PerturbedFLRWMetricFast(Metric):
    """
    Optimized FLRW metric with perturbations.
    Uses Numba-compiled geodesic calculations and caching.

    Parameters
    ----------
    enable_lensing : bool, optional (default False)
        When True, ``geodesic_equations`` automatically dispatches to
        ``geodesic_equations_extended`` for 24-component state vectors,
        integrating the Sachs basis, optical tidal matrix and Jacobi map
        alongside the standard geodesic.  When False, only the 8-component
        geodesic is integrated regardless of the state size.

    slow_roll : bool, optional (default False)
        When True, all *temporal* potential derivatives are forced to zero:
        Phi' = 0, Phi'' = 0, d_i Phi' = 0  (and similarly Psi' since Psi = Phi).
        This avoids the finite-difference calls in conformal time that
        are needed to evaluate those derivatives, yielding a significant
        speedup when the potentials are quasi-static.

    sachs_screen_convention : str, optional (default "metric")
        Screen convention used by the online 24-state solver when transporting
        the Sachs basis. Use ``"conformal_metric"`` to evolve the conformal
        Sachs screen instead of the historical physical one.
    """
    def __init__(
        self,
        a_of_eta,
        grid,
        interpolator,
        analytical_geodesics=True,
        adot_of_eta=None,
        cosmology=None,
        enable_lensing=False,
        slow_roll=False,
        sachs_screen_convention="metric",
    ):
        self.analytical_geodesics = analytical_geodesics
        self.a_of_eta = a_of_eta
        self.adot_of_eta = adot_of_eta
        self.cosmology = cosmology          # needed for conformal_hubble / conformal_hubble_prime
        self.grid = grid
        self.interp = interpolator
        self.enable_lensing = enable_lensing
        self.slow_roll = slow_roll
        self.sachs_screen_convention = sachs_screen_convention
        
        # Cache for a(eta) and adot to avoid repeated interpolation calls
        self._eta_cache = None
        self._a_cache = None
        self._adot_cache = None
    
    def metric_tensor(self, x):
        """Metric tensor - kept for compatibility, rarely used in integration."""
        eta, pos = x[0], x[1:]
        a, _ = self._get_scale_factor_and_derivative(eta)
        
        # Get potential in SI units, then normalize by c^2
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
        Christoffel symbols  --  used for Sachs basis parallel transport.

        Convention: phi, grad_phi, phi_dot are all in **SI units**
        (m^2/s^2, m/s^2, m^2/s^3 respectively).  No pre-normalisation by c^2.
        This matches ``PerturbedFLRWMetric.christoffel()`` exactly.
        """
        eta, pos = x[0], x[1:]
        a, adot = self._get_scale_factor_and_derivative(eta)
        phi, grad_phi, phi_dot = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)

        # Psi = Phi (no anisotropic stress)
        psi = phi
        grad_psi = grad_phi          # tuple of 3 floats, SI (m/s^2)
        psi_dot = phi_dot

        c_inv  = 1.0 / c
        c_inv2 = c_inv * c_inv
        c_inv4 = c_inv2 * c_inv2
        a_inv  = 1.0 / a
        a_inv2 = a_inv * a_inv
        adot_over_a = adot * a_inv
        phi_plus_psi = phi + psi
        a_adot_c_inv2    = a * adot * c_inv2
        a2_phi_dot_c_inv4 = a * a * phi_dot * c_inv4

        G3 = np.zeros((4, 4, 4))

        G3[0,0,0] = c_inv2 * psi_dot
        G3[1,0,0] = grad_psi[0] * a_inv2
        G3[2,0,0] = grad_psi[1] * a_inv2
        G3[3,0,0] = grad_psi[2] * a_inv2
        G3[1,1,1] = -c_inv2 * grad_phi[0]
        G3[2,2,2] = -c_inv2 * grad_phi[1]
        G3[3,3,3] = -c_inv2 * grad_phi[2]

        diag_term = a_adot_c_inv2 + 2 * a_adot_c_inv2 * c_inv2 * phi_plus_psi - a2_phi_dot_c_inv4
        G3[0,1,1] = diag_term
        G3[0,2,2] = diag_term
        G3[0,3,3] = diag_term

        G3[0,0,1] = G3[0,1,0] = grad_psi[0] * c_inv2
        G3[0,0,2] = G3[0,2,0] = grad_psi[1] * c_inv2
        G3[0,0,3] = G3[0,3,0] = grad_psi[2] * c_inv2

        time_mix_term = adot_over_a - phi_dot * c_inv2
        G3[1,1,0] = G3[1,0,1] = time_mix_term
        G3[2,2,0] = G3[2,0,2] = time_mix_term
        G3[3,3,0] = G3[3,0,3] = time_mix_term

        G3[1,2,2] = G3[1,3,3] = grad_phi[0] * c_inv2
        G3[2,1,1] = G3[2,3,3] = grad_phi[1] * c_inv2
        G3[3,1,1] = G3[3,2,2] = grad_phi[2] * c_inv2
        G3[1,1,2] = G3[1,2,1] = -grad_phi[1] * c_inv2
        G3[1,1,3] = G3[1,3,1] = -grad_phi[2] * c_inv2
        G3[2,2,1] = G3[2,1,2] = -grad_phi[0] * c_inv2
        G3[2,2,3] = G3[2,3,2] = -grad_phi[2] * c_inv2
        G3[3,3,1] = G3[3,1,3] = -grad_phi[0] * c_inv2
        G3[3,3,2] = G3[3,2,3] = -grad_phi[1] * c_inv2
        return G3
        
    def _get_scale_factor_and_derivative(self, eta):
        """Get a(eta) and adot with caching. Assumes self.a_of_eta can return derivative
        if it provides a method 'a_and_adot'. Otherwise, this builds a cached adot-interpolator externally.
        """
        if self._eta_cache == eta:
            return self._a_cache, self._adot_cache

        # Prefer a fast combined method if available
        if hasattr(self.a_of_eta, "a_and_adot"):
            a, adot = self.a_of_eta.a_and_adot(eta)
        elif callable(self.adot_of_eta):
            # Best option: accept an analytical (or otherwise accurate) adot_of_eta.
            a = self.a_of_eta(eta)
            adot = self.adot_of_eta(eta)
        else:
            # Fallback: robust finite-difference derivative.
            # IMPORTANT: eta is ~1e18 in typical runs; a tiny absolute dt (e.g. 1e-5)
            # collapses to (eta+/-dt)==eta in float64, yielding adot=0.
            a = self.a_of_eta(eta)
            dt = max(1e-6 * abs(eta), 1e-6)
            adot = (self.a_of_eta(eta + dt) - self.a_of_eta(eta - dt)) / (2 * dt)

        # Guard against invalid scale factor (prevents division by zero downstream).
        try:
            a = float(a)
        except Exception:
            a = np.nan
        if not np.isfinite(a) or abs(a) < 1e-30:
            a = 1e-30
        if not np.isfinite(adot):
            adot = 0.0

        self._eta_cache = eta
        self._a_cache = a
        self._adot_cache = adot
        return a, adot

    def geodesic_equations(self, state):
        """
        Optimized geodesic equations using Numba-compiled acceleration calculation.

        If ``enable_lensing`` is True and *state* has 24 components, this
        transparently forwards to ``geodesic_equations_extended`` so that the
        integrator does not need to know about the lensing machinery.

        Returns a numpy array of shape ``(N,)`` where *N* matches the input
        state size (8 or 24).
        """
        # --- Lensing dispatch -------------------------------------------
        if self.enable_lensing and state.shape[0] == 24:
            return self.geodesic_equations_extended(state)

        x, u = state[:4], state[4:8]
        eta, pos = x[0], x[1:]
        
        # Get scale factor (with caching)
        a, adot = self._get_scale_factor_and_derivative(eta)
        
        # Get potential and gradient (fast interpolator) - in SI units
        phi_SI, grad_phi_tuple, phi_dot_SI = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)
        gx, gy, gz = grad_phi_tuple  # tuple of floats
        
        # Normalize
        phi_normalized = phi_SI / (c**2)

        # --- slow_roll: zero out time derivative if requested -----------
        if self.slow_roll:
            phi_dot_normalized = 0.0
        else:
            phi_dot_normalized = phi_dot_SI / (c**2)
        
        # Compute accelerations with Numba
        if self.analytical_geodesics:
            du0, du1, du2, du3 = compute_analytical_acceleration(
                u[0], u[1], u[2], u[3],
                a, adot,
                phi_normalized,
                gx, gy, gz,
                phi_dot_normalized,
                c
            )
        elif not self.analytical_geodesics:
            du0, du1, du2, du3 = compute_tensorial_acceleration(
                u[0], u[1], u[2], u[3],
                a, adot,
                phi_normalized,
                gx, gy, gz,
                phi_dot_normalized,
                c
            )
        
        # Build output array in-place to avoid concatenation
        out = np.empty(8, dtype=float)
        out[0:4] = u
        out[4] = du0
        out[5] = du1
        out[6] = du2
        out[7] = du3
        return out
    
    def metric_physical_quantities(self, state):
        """Get physical quantities for recording."""
        eta, pos = state[0], state[1:4]
        a, _ = self._get_scale_factor_and_derivative(eta)
        phi, grad_phi, phi_dot = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)
        
        quantities = np.array([a, phi, *grad_phi, phi_dot])
        return quantities

    # ------------------------------------------------------------------
    #  Extended geodesic equations for weak-lensing observables
    # ------------------------------------------------------------------

    def geodesic_equations_extended(self, state):
        r"""
        24-component ODE for geodesic + Sachs transport + Jacobi map.

        State layout (24 components)::

            state[ 0: 4]  =  x^mu       (position: eta, x, y, z)
            state[ 4: 8]  =  k^mu       (4-velocity / wave-vector)
            state[ 8:12]  =  e_1^mu     (first Sachs screen vector)
            state[12:16]  =  e_2^mu     (second Sachs screen vector)
            state[16:20]  =  D_{AB}    (Jacobi map, flat row-major: D11,D12,D21,D22)
            state[20:24]  =  dP_{AB}    (dD/dlambda,    flat row-major: P11,P12,P21,P22)

        Returns
        -------
        dstate : ndarray (24,)
            Time derivative of the full state.

        Notes
        -----
        The first 8 components are computed by calling the existing
        ``geodesic_equations(state[:8])``.  The remaining 16 are:

        - **Sachs transport** (8 components):
          de_A^mu/dlambda = -Gamma^mu_{nu sigma} k^sigma e_A^nu

        - **Jacobi map** (8 components):
          dD/dlambda = P,  dP/dlambda = R*D
          where R_{AB} is the optical tidal matrix.

        Requires ``self.cosmology`` to be set (provides ``conformal_hubble``
        and ``conformal_hubble_prime``).
        """
        if self.cosmology is None:
            raise RuntimeError(
                "geodesic_equations_extended requires self.cosmology to be set. "
                "Pass cosmology=... to PerturbedFLRWMetricFast.__init__()."
            )

        # Unpack state
        x_mu = state[0:4]
        k_mu = state[4:8]
        e1_mu = state[8:12]
        e2_mu = state[12:16]
        D_flat = state[16:20]
        P_flat = state[20:24]

        eta = x_mu[0]
        pos = x_mu[1:4]

        # === 1. Standard geodesic equations (first 8 components) ===
        geo_rhs = self.geodesic_equations(state[:8])
        # geo_rhs = [k^mu(4), dk^mu(4)]

        # === 2. Get all field data we need ===
        a, adot = self._get_scale_factor_and_derivative(eta)
        H_conf = self.cosmology.conformal_hubble(eta)
        H_prime = self.cosmology.conformal_hubble_prime(eta)

        # Full interpolation: value, spatial gradient, spatial Hessian, time derivative
        phi_SI, grad3_tuple, hess3_tuple, phi_dot_SI = \
            self.interp.value_gradient_hessian_and_time_derivative(pos, "Phi", eta)

        gx, gy, gz = grad3_tuple
        hxx, hyy, hzz, hxy, hxz, hyz = hess3_tuple

        # --- Temporal derivatives: skip FD when slow_roll is on ----------
        if self.slow_roll:
            grad_phi_dot = np.zeros(3)
            phi_ddot = 0.0
            phi_dot_SI = 0.0        # override for Riemann blocks as well
        else:
            # d_i Phi' = d^2Phi/(dx^i deta)  and  Phi'' = d^2Phi/deta^2
            # Use finite difference in conformal time
            dt_fd = max(1e-6 * abs(eta), 1.0)  # absolute delta in seconds

            _, grad3_p, _, phi_dot_p = \
                self.interp.value_gradient_hessian_and_time_derivative(pos, "Phi", eta + dt_fd)
            _, grad3_m, _, phi_dot_m = \
                self.interp.value_gradient_hessian_and_time_derivative(pos, "Phi", eta - dt_fd)

            gxp, gyp, gzp = grad3_p
            gxm, gym, gzm = grad3_m

            # Mixed derivatives  d_i Phi'  (spatial gradient of time derivative)
            grad_phi_dot = np.array([
                (gxp - gxm) / (2.0 * dt_fd),
                (gyp - gym) / (2.0 * dt_fd),
                (gzp - gzm) / (2.0 * dt_fd),
            ])

            # Second time derivative  Phi''
            phi_ddot = (phi_dot_p - phi_dot_m) / (2.0 * dt_fd)

        # Spatial gradient and Hessian as arrays
        grad_phi = np.array([gx, gy, gz])
        hess_phi = np.array([
            [hxx, hxy, hxz],
            [hxy, hyy, hyz],
            [hxz, hyz, hzz],
        ])

        # === 3. Christoffel symbols for Sachs transport ===
        gamma_christoffel = self.christoffel(x_mu)

        # Metric tensor at current position (diagonal FLRW)
        phi_norm = phi_SI / (c * c)
        psi_norm = phi_norm
        g_mu_nu = np.zeros((4, 4))
        g_mu_nu[0, 0] = -a * a * (1.0 + 2.0 * psi_norm) * c * c
        g_mu_nu[1, 1] = a * a * (1.0 - 2.0 * phi_norm)
        g_mu_nu[2, 2] = a * a * (1.0 - 2.0 * phi_norm)
        g_mu_nu[3, 3] = a * a * (1.0 - 2.0 * phi_norm)

        if self.sachs_screen_convention == "conformal_metric":
            transport_g_mu_nu = g_mu_nu / (a * a)
        else:
            transport_g_mu_nu = g_mu_nu

        # === 4. Sachs transport RHS ===
        de1 = screen_projected_sachs_transport_rhs(e1_mu, gamma_christoffel, k_mu, transport_g_mu_nu)
        de2 = screen_projected_sachs_transport_rhs(e2_mu, gamma_christoffel, k_mu, transport_g_mu_nu)

        # === 5. Riemann blocks (all-down)  -> optical tidal matrix  -> Jacobi RHS ===
        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H_conf, H_prime,
            phi_SI, phi_dot_SI, phi_ddot,
            grad_phi, grad_phi_dot, hess_phi,
            c,
        )

        # Optical tidal matrix R_{AB}
        if self.sachs_screen_convention == "metric":
            R_AB = optical_tidal_matrix_optimized(
                Rd_k00l, Rd_0lki, Rd_kijl,
                k_mu, e1_mu, e2_mu, g_mu_nu,
            )
        else:
            # Non-default conventions are controlled by the local projection
            # choice rather than by the historical transported physical basis.
            R_AB, _, _ = optical_tidal_matrix_for_local_screen_convention(
                Rd_k00l, Rd_0lki, Rd_kijl,
                k_mu, g_mu_nu, a,
                convention=self.sachs_screen_convention,
            )

        # Jacobi map RHS:  dD/dlambda = P,  dP/dlambda = R*D
        DP_state = np.empty(8)
        DP_state[0:4] = D_flat
        DP_state[4:8] = P_flat
        jac_rhs = jacobi_rhs(DP_state, R_AB)

        # === 6. Assemble full 24-component RHS ===
        dstate = np.empty(24)
        dstate[0:8] = geo_rhs           # dx/dlambda, dk/dlambda
        dstate[8:12] = de1               # de1/dlambda
        dstate[12:16] = de2              # de2/dlambda
        dstate[16:20] = jac_rhs[0:4]    # dD/dlambda = P
        dstate[20:24] = jac_rhs[4:8]    # dP/dlambda = R*D
        return dstate

    def metric_physical_quantities(self, state):
        """Get physical quantities for recording."""
        eta, pos = state[0], state[1:4]
        a, _ = self._get_scale_factor_and_derivative(eta)
        phi, grad_phi, phi_dot = self.interp.value_gradient_and_time_derivative(pos, "Phi", eta)

        quantities = np.array([a, phi, *grad_phi, phi_dot])
        return quantities
