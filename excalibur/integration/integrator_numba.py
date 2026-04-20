r"""
Full-njit integration backend for perturbed FLRW lensing.

Scope
-----
Targets the user's specific configuration:
  - tricubic + clamp interpolation
  - AMR (patch-based, finest-first dispatch)
  - slow_roll=True  (Phi' = Phi'' = d_i Phi' = 0)
  - analytical_geodesics=False  (uses compute_tensorial_acceleration)
  - 8-component state (geodesic only) or 24-component (with Sachs + Jacobi)
  - fixed-step RK4

The entire per-step hot-path runs under @njit: AMR patch lookup, tricubic
interpolation, Christoffel assembly, Riemann blocks, optical tidal matrix,
Jacobi RHS. No Python object calls inside the RK4 stages.

Layout
------
AMR patches are flattened at backend construction into:
  patch_origins   (P, 3)  float64
  patch_uppers    (P, 3)
  patch_spacings  (P, 3)
  patch_shapes    (P, 3)  int64
  patch_fields    numba.typed.List of (nx,ny,nz) float64 C-contiguous arrays

Ordering: finest-first, with the root grid as the last entry (always matches
queries inside the domain, acting as fallback).

Eta -> (a, adot, H_conf, H_prime) are sampled on a dense uniform grid; lookup
is a single integer index + linear interpolation.

Usage
-----
    backend = NumbaAMRBackend(
        amr_grid, cosmology,
        c_val=c, slow_roll=True, lensing=True,
    )
    traj, final_state, lam = backend.integrate_rk4(
        state0_24, dt=dt_fine, n_steps=N,
    )
"""
from __future__ import annotations

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from excalibur.grid.interpolator_4d_fast import _fused_3d_tricubic_clamp
from excalibur.metrics.perturbed_flrw_metric_fast import compute_tensorial_acceleration
from excalibur.observables.riemann_perturbed_flrw import riemann_blocks_kernel
from excalibur.observables.sachs_basis import sachs_transport_rhs
from excalibur.observables.optical_tidal_matrix import (
    optical_tidal_matrix_optimized,
    jacobi_rhs,
)


# ======================================================================
#  Eta-table interpolation
# ======================================================================

@njit(cache=True, fastmath=True)
def _interp_eta_table(eta, eta_min, inv_deta, table):
    """Linear interpolation on a uniform eta grid. Clamps at boundaries."""
    n = table.shape[0]
    pos = (eta - eta_min) * inv_deta
    if pos <= 0.0:
        return table[0]
    if pos >= n - 1:
        return table[n - 1]
    i = int(pos)
    t = pos - i
    return table[i] * (1.0 - t) + table[i + 1] * t


@njit(cache=True, fastmath=True)
def _a_adot_H_Hp(eta, eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab):
    a = _interp_eta_table(eta, eta_min, inv_deta, a_tab)
    adot = _interp_eta_table(eta, eta_min, inv_deta, adot_tab)
    H = _interp_eta_table(eta, eta_min, inv_deta, H_tab)
    Hp = _interp_eta_table(eta, eta_min, inv_deta, Hp_tab)
    return a, adot, H, Hp


# ======================================================================
#  AMR patch lookup + tricubic interpolation
# ======================================================================

@njit(cache=True, fastmath=True)
def _find_patch(x, y, z, origins, uppers, P):
    """Return index of finest patch containing (x,y,z), else P-1 (root fallback)."""
    for p in range(P):
        if (x >= origins[p, 0] and x < uppers[p, 0]
                and y >= origins[p, 1] and y < uppers[p, 1]
                and z >= origins[p, 2] and z < uppers[p, 2]):
            return p
    return P - 1


@njit(cache=True, fastmath=True)
def _amr_tricubic_clamp(x, y, z, origins, uppers, spacings, shapes, fields, P):
    """
    Finest-patch tricubic (clamp) interpolation.
    Returns (val, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz).
    """
    p = _find_patch(x, y, z, origins, uppers, P)
    return _fused_3d_tricubic_clamp(
        x, y, z, fields[p],
        origins[p, 0], origins[p, 1], origins[p, 2],
        spacings[p, 0], spacings[p, 1], spacings[p, 2],
        shapes[p, 0], shapes[p, 1], shapes[p, 2],
    )


# ======================================================================
#  Christoffel symbols for perturbed FLRW  (SI-unit convention)
# ======================================================================

@njit(cache=True, fastmath=True)
def _christoffel_flrw(a, adot, phi_SI, gx, gy, gz, phi_dot_SI, c_val):
    """
    Build Gamma^mu_{nu sigma} for perturbed FLRW (Psi = Phi, Newtonian gauge).
    Matches PerturbedFLRWMetricFast.christoffel exactly.
    """
    c_inv = 1.0 / c_val
    c_inv2 = c_inv * c_inv
    c_inv4 = c_inv2 * c_inv2
    a_inv = 1.0 / a
    a_inv2 = a_inv * a_inv
    adot_over_a = adot * a_inv
    phi_plus_psi = 2.0 * phi_SI
    a_adot_c_inv2 = a * adot * c_inv2
    a2_phi_dot_c_inv4 = a * a * phi_dot_SI * c_inv4

    G = np.zeros((4, 4, 4))

    G[0, 0, 0] = c_inv2 * phi_dot_SI
    G[1, 0, 0] = gx * a_inv2
    G[2, 0, 0] = gy * a_inv2
    G[3, 0, 0] = gz * a_inv2
    G[1, 1, 1] = -c_inv2 * gx
    G[2, 2, 2] = -c_inv2 * gy
    G[3, 3, 3] = -c_inv2 * gz

    diag = a_adot_c_inv2 + 2.0 * a_adot_c_inv2 * c_inv2 * phi_plus_psi - a2_phi_dot_c_inv4
    G[0, 1, 1] = diag
    G[0, 2, 2] = diag
    G[0, 3, 3] = diag

    gxc = gx * c_inv2
    gyc = gy * c_inv2
    gzc = gz * c_inv2
    G[0, 0, 1] = gxc; G[0, 1, 0] = gxc
    G[0, 0, 2] = gyc; G[0, 2, 0] = gyc
    G[0, 0, 3] = gzc; G[0, 3, 0] = gzc

    tmix = adot_over_a - phi_dot_SI * c_inv2
    G[1, 1, 0] = tmix; G[1, 0, 1] = tmix
    G[2, 2, 0] = tmix; G[2, 0, 2] = tmix
    G[3, 3, 0] = tmix; G[3, 0, 3] = tmix

    G[1, 2, 2] = gxc; G[1, 3, 3] = gxc
    G[2, 1, 1] = gyc; G[2, 3, 3] = gyc
    G[3, 1, 1] = gzc; G[3, 2, 2] = gzc
    G[1, 1, 2] = -gyc; G[1, 2, 1] = -gyc
    G[1, 1, 3] = -gzc; G[1, 3, 1] = -gzc
    G[2, 2, 1] = -gxc; G[2, 1, 2] = -gxc
    G[2, 2, 3] = -gzc; G[2, 3, 2] = -gzc
    G[3, 3, 1] = -gxc; G[3, 1, 3] = -gxc
    G[3, 3, 2] = -gyc; G[3, 2, 3] = -gyc
    return G


# ======================================================================
#  8-component geodesic RHS (no lensing)
# ======================================================================

@njit(cache=True, fastmath=True)
def _geodesic_rhs_8(
    state,
    origins, uppers, spacings, shapes, fields, P,
    eta_min, inv_deta, a_tab, adot_tab,
    c_val, slow_roll,
):
    eta = state[0]
    x = state[1]; y = state[2]; z = state[3]
    u0 = state[4]; u1 = state[5]; u2 = state[6]; u3 = state[7]

    a = _interp_eta_table(eta, eta_min, inv_deta, a_tab)
    adot = _interp_eta_table(eta, eta_min, inv_deta, adot_tab)

    val, gx, gy, gz, _hxx, _hyy, _hzz, _hxy, _hxz, _hyz = _amr_tricubic_clamp(
        x, y, z, origins, uppers, spacings, shapes, fields, P,
    )

    c2 = c_val * c_val
    phi_n = val / c2
    phi_dot_n = 0.0  # slow_roll assumed; non-slow-roll falls back to Python path

    du0, du1, du2, du3 = compute_tensorial_acceleration(
        u0, u1, u2, u3, a, adot, phi_n, gx, gy, gz, phi_dot_n, c_val,
    )

    out = np.empty(8)
    out[0] = u0; out[1] = u1; out[2] = u2; out[3] = u3
    out[4] = du0; out[5] = du1; out[6] = du2; out[7] = du3
    return out


# ======================================================================
#  24-component geodesic + Sachs + Jacobi RHS (lensing)
# ======================================================================

@njit(cache=True, fastmath=True)
def _geodesic_rhs_24(
    state,
    origins, uppers, spacings, shapes, fields, P,
    eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
    c_val, slow_roll,
):
    eta = state[0]
    x = state[1]; y = state[2]; z = state[3]
    u0 = state[4]; u1 = state[5]; u2 = state[6]; u3 = state[7]

    a, adot, H_conf, H_prime = _a_adot_H_Hp(
        eta, eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
    )

    val, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz = _amr_tricubic_clamp(
        x, y, z, origins, uppers, spacings, shapes, fields, P,
    )

    c2 = c_val * c_val
    phi_n = val / c2
    phi_dot_n = 0.0          # slow_roll
    phi_dot_SI = 0.0
    phi_ddot = 0.0

    # --- Geodesic acceleration (8 comp) ---
    du0, du1, du2, du3 = compute_tensorial_acceleration(
        u0, u1, u2, u3, a, adot, phi_n, gx, gy, gz, phi_dot_n, c_val,
    )

    # --- Christoffel for Sachs transport ---
    k_mu = np.empty(4)
    k_mu[0] = u0; k_mu[1] = u1; k_mu[2] = u2; k_mu[3] = u3
    G = _christoffel_flrw(a, adot, val, gx, gy, gz, phi_dot_SI, c_val)

    e1 = state[8:12]
    e2 = state[12:16]
    de1 = sachs_transport_rhs(e1, G, k_mu)
    de2 = sachs_transport_rhs(e2, G, k_mu)

    # --- Riemann blocks ---
    grad_phi = np.empty(3); grad_phi[0] = gx; grad_phi[1] = gy; grad_phi[2] = gz
    grad_phi_dot = np.zeros(3)   # slow_roll
    hess_phi = np.empty((3, 3))
    hess_phi[0, 0] = hxx; hess_phi[1, 1] = hyy; hess_phi[2, 2] = hzz
    hess_phi[0, 1] = hxy; hess_phi[1, 0] = hxy
    hess_phi[0, 2] = hxz; hess_phi[2, 0] = hxz
    hess_phi[1, 2] = hyz; hess_phi[2, 1] = hyz

    Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
        a, H_conf, H_prime,
        val, phi_dot_SI, phi_ddot,
        grad_phi, grad_phi_dot, hess_phi,
        c_val,
    )

    # --- Metric tensor at current position (diagonal FLRW) ---
    g_mu_nu = np.zeros((4, 4))
    a2 = a * a
    g_mu_nu[0, 0] = -a2 * (1.0 + 2.0 * phi_n) * c2
    g_mu_nu[1, 1] = a2 * (1.0 - 2.0 * phi_n)
    g_mu_nu[2, 2] = a2 * (1.0 - 2.0 * phi_n)
    g_mu_nu[3, 3] = a2 * (1.0 - 2.0 * phi_n)

    R_AB = optical_tidal_matrix_optimized(
        Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g_mu_nu,
    )

    # --- Jacobi map RHS ---
    DP = np.empty(8)
    DP[0:4] = state[16:20]
    DP[4:8] = state[20:24]
    jrhs = jacobi_rhs(DP, R_AB)

    # --- Assemble 24-comp RHS ---
    out = np.empty(24)
    out[0] = u0; out[1] = u1; out[2] = u2; out[3] = u3
    out[4] = du0; out[5] = du1; out[6] = du2; out[7] = du3
    for i in range(4):
        out[8 + i] = de1[i]
        out[12 + i] = de2[i]
    for i in range(4):
        out[16 + i] = jrhs[i]
        out[20 + i] = jrhs[4 + i]
    return out


# ======================================================================
#  RK4 integration loops
# ======================================================================

@njit(cache=True, fastmath=True)
def _rk4_step_8(state, dt,
                origins, uppers, spacings, shapes, fields, P,
                eta_min, inv_deta, a_tab, adot_tab,
                c_val, slow_roll):
    k1 = _geodesic_rhs_8(state, origins, uppers, spacings, shapes, fields, P,
                         eta_min, inv_deta, a_tab, adot_tab, c_val, slow_roll)
    s2 = state + 0.5 * dt * k1
    k2 = _geodesic_rhs_8(s2, origins, uppers, spacings, shapes, fields, P,
                         eta_min, inv_deta, a_tab, adot_tab, c_val, slow_roll)
    s3 = state + 0.5 * dt * k2
    k3 = _geodesic_rhs_8(s3, origins, uppers, spacings, shapes, fields, P,
                         eta_min, inv_deta, a_tab, adot_tab, c_val, slow_roll)
    s4 = state + dt * k3
    k4 = _geodesic_rhs_8(s4, origins, uppers, spacings, shapes, fields, P,
                         eta_min, inv_deta, a_tab, adot_tab, c_val, slow_roll)
    out = np.empty(8)
    for i in range(8):
        out[i] = state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
    return out


@njit(cache=True, fastmath=True)
def _rk4_step_24(state, dt,
                 origins, uppers, spacings, shapes, fields, P,
                 eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
                 c_val, slow_roll):
    k1 = _geodesic_rhs_24(state, origins, uppers, spacings, shapes, fields, P,
                          eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
                          c_val, slow_roll)
    s2 = state + 0.5 * dt * k1
    k2 = _geodesic_rhs_24(s2, origins, uppers, spacings, shapes, fields, P,
                          eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
                          c_val, slow_roll)
    s3 = state + 0.5 * dt * k2
    k3 = _geodesic_rhs_24(s3, origins, uppers, spacings, shapes, fields, P,
                          eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
                          c_val, slow_roll)
    s4 = state + dt * k3
    k4 = _geodesic_rhs_24(s4, origins, uppers, spacings, shapes, fields, P,
                          eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
                          c_val, slow_roll)
    out = np.empty(24)
    for i in range(24):
        out[i] = state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
    return out


@njit(cache=True, fastmath=True)
def _integrate_loop_8(
    state0, dt, n_steps, lambda_stop, record_every,
    origins, uppers, spacings, shapes, fields, P,
    eta_min, inv_deta, a_tab, adot_tab,
    c_val, slow_roll,
):
    """
    Run fixed-step RK4 on the 8-component state.

    Stops at whichever comes first: n_steps reached OR |lambda| >= |lambda_stop|.
    Returns (traj, final_state, n_recorded, lambda_final, n_steps_actual).
    traj[i] contains state recorded at step i * record_every (i=0 is initial).
    """
    state = state0.copy()
    lam = 0.0

    # Allocate trajectory buffer
    if record_every <= 0:
        n_rec = 1   # only final
    else:
        n_rec = 1 + n_steps // record_every + 2
    traj = np.empty((n_rec, 8))
    traj[0] = state
    rec_idx = 1

    use_lam_stop = lambda_stop > 0.0
    step = 0
    while step < n_steps:
        if use_lam_stop and abs(lam) >= lambda_stop:
            break
        # Clamp final dt to not overshoot lambda_stop
        dt_eff = dt
        if use_lam_stop:
            remaining = lambda_stop - abs(lam)
            if remaining < abs(dt_eff):
                dt_eff = remaining if dt > 0 else -remaining
        state = _rk4_step_8(state, dt_eff,
                            origins, uppers, spacings, shapes, fields, P,
                            eta_min, inv_deta, a_tab, adot_tab,
                            c_val, slow_roll)
        lam += dt_eff
        step += 1
        if record_every > 0 and (step % record_every) == 0 and rec_idx < n_rec:
            traj[rec_idx] = state
            rec_idx += 1

    # Always record final
    if rec_idx < n_rec:
        traj[rec_idx] = state
        rec_idx += 1

    return traj[:rec_idx], state, rec_idx, lam, step


@njit(cache=True, fastmath=True)
def _integrate_loop_24(
    state0, dt, n_steps, lambda_stop, record_every,
    origins, uppers, spacings, shapes, fields, P,
    eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
    c_val, slow_roll,
):
    state = state0.copy()
    lam = 0.0

    if record_every <= 0:
        n_rec = 1
    else:
        n_rec = 1 + n_steps // record_every + 2
    traj = np.empty((n_rec, 24))
    traj[0] = state
    rec_idx = 1

    use_lam_stop = lambda_stop > 0.0
    step = 0
    while step < n_steps:
        if use_lam_stop and abs(lam) >= lambda_stop:
            break
        dt_eff = dt
        if use_lam_stop:
            remaining = lambda_stop - abs(lam)
            if remaining < abs(dt_eff):
                dt_eff = remaining if dt > 0 else -remaining
        state = _rk4_step_24(state, dt_eff,
                             origins, uppers, spacings, shapes, fields, P,
                             eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
                             c_val, slow_roll)
        lam += dt_eff
        step += 1
        if record_every > 0 and (step % record_every) == 0 and rec_idx < n_rec:
            traj[rec_idx] = state
            rec_idx += 1

    if rec_idx < n_rec:
        traj[rec_idx] = state
        rec_idx += 1

    return traj[:rec_idx], state, rec_idx, lam, step


# ======================================================================
#  Python wrapper: flattens AMR + builds eta tables
# ======================================================================

class NumbaAMRBackend:
    """
    Pure-njit integration backend for AMR + perturbed FLRW + lensing.

    Configuration is frozen at construction time (the underlying njit kernels
    hardwire tricubic + clamp + slow_roll + analytical_geodesics=False).

    Parameters
    ----------
    amr_grid : AMRGrid
        Populated AMR hierarchy with field "Phi".
    cosmology : LCDM_Cosmology
        Provides a(eta), adot(eta), conformal_hubble, conformal_hubble_prime.
    c_val : float
        Speed of light in SI.
    slow_roll : bool
        Must be True for now. (Placeholder for future full-time-derivative path.)
    lensing : bool
        If True, expects 24-component states; otherwise 8-component.
    n_eta_samples : int
        Number of samples in the eta -> (a, adot, H, Hp) lookup table.
    field_name : str
        Name of the scalar field in the AMR grid (default "Phi").
    """

    def __init__(
        self,
        amr_grid,
        cosmology,
        c_val,
        slow_roll=True,
        lensing=False,
        n_eta_samples=4096,
        field_name="Phi",
        eta_range=None,
    ):
        if not slow_roll:
            raise NotImplementedError(
                "NumbaAMRBackend currently requires slow_roll=True. "
                "Non-slow-roll needs FD in conformal time inside the RHS."
            )

        self.amr = amr_grid
        self.cosmology = cosmology
        self.c_val = float(c_val)
        self.slow_roll = slow_roll
        self.lensing = bool(lensing)
        self.field_name = field_name

        self._build_patch_arrays(amr_grid, field_name)
        self._build_eta_tables(cosmology, n_eta_samples, eta_range)

    # ----------------------------------------------------------------
    def _build_patch_arrays(self, amr_grid, field_name):
        """Flatten AMR patches into arrays + a typed list of 3D fields.

        Order: refined patches finest-first, then the root grid last.
        """
        patches = list(amr_grid.patches)  # already finest-first
        root = amr_grid.root

        P = len(patches) + 1
        origins = np.empty((P, 3), dtype=np.float64)
        uppers = np.empty((P, 3), dtype=np.float64)
        spacings = np.empty((P, 3), dtype=np.float64)
        shapes = np.empty((P, 3), dtype=np.int64)
        fields = NumbaList()

        for p, patch in enumerate(patches):
            origins[p] = np.asarray(patch.origin, dtype=np.float64)
            spacings[p] = np.asarray(patch.spacing, dtype=np.float64)
            shp = np.asarray(patch.shape, dtype=np.int64)
            shapes[p] = shp
            uppers[p] = origins[p] + spacings[p] * shp.astype(np.float64)
            fields.append(np.ascontiguousarray(patch._field_data, dtype=np.float64))

        # Root at last index
        p = P - 1
        origins[p] = np.asarray(root.origin, dtype=np.float64)
        spacings[p] = np.asarray(root.spacing, dtype=np.float64)
        shp = np.asarray(root.shape, dtype=np.int64)
        shapes[p] = shp
        uppers[p] = origins[p] + spacings[p] * shp.astype(np.float64)
        fields.append(np.ascontiguousarray(root.fields[field_name], dtype=np.float64))

        self.P = P
        self.origins = origins
        self.uppers = uppers
        self.spacings = spacings
        self.shapes = shapes
        self.fields = fields

    # ----------------------------------------------------------------
    def _build_eta_tables(self, cosmology, n_samples, eta_range):
        """Build uniform eta-grid lookup for a(eta), adot(eta), H_conf, H_prime."""
        if eta_range is not None:
            eta_min, eta_max = eta_range
        elif hasattr(cosmology, "eta_min") and hasattr(cosmology, "eta_max"):
            eta_min = float(cosmology.eta_min)
            eta_max = float(cosmology.eta_max)
        else:
            # Fall back: use cosmology's a_of_eta interpolant bounds if accessible
            a_of_eta = cosmology.a_of_eta
            # scipy interp1d exposes .x
            if hasattr(a_of_eta, "x"):
                eta_min = float(a_of_eta.x[0])
                eta_max = float(a_of_eta.x[-1])
            else:
                raise ValueError(
                    "Cannot infer eta range; pass eta_range=(eta_min, eta_max)"
                )

        eta_tab = np.linspace(eta_min, eta_max, n_samples)
        a_tab = np.empty(n_samples)
        adot_tab = np.empty(n_samples)
        H_tab = np.empty(n_samples)
        Hp_tab = np.empty(n_samples)

        a_fn = cosmology.a_of_eta
        adot_fn = getattr(cosmology, "adot_of_eta", None)

        for i, eta in enumerate(eta_tab):
            a_tab[i] = float(a_fn(eta))
            if adot_fn is not None:
                adot_tab[i] = float(adot_fn(eta))
            else:
                # central FD with eta-scaled step
                dt = max(1e-6 * abs(eta), 1.0)
                adot_tab[i] = (float(a_fn(eta + dt)) - float(a_fn(eta - dt))) / (2.0 * dt)
            H_tab[i] = float(cosmology.conformal_hubble(eta))
            Hp_tab[i] = float(cosmology.conformal_hubble_prime(eta))

        self.eta_min = float(eta_tab[0])
        self.eta_max = float(eta_tab[-1])
        self.inv_deta = (n_samples - 1) / (self.eta_max - self.eta_min)
        self.a_tab = a_tab
        self.adot_tab = adot_tab
        self.H_tab = H_tab
        self.Hp_tab = Hp_tab

    # ----------------------------------------------------------------
    def integrate_rk4(self, state0, dt, n_steps, lambda_stop=0.0, record_every=0):
        """
        Fixed-step RK4 integration.

        Parameters
        ----------
        state0 : ndarray (8,) or (24,)
        dt : float
            Affine-parameter step (signed).
        n_steps : int
            Maximum number of RK4 steps.
        lambda_stop : float
            If > 0, stop when |lambda| >= lambda_stop (final step clamped).
        record_every : int
            If > 0, record every N-th step.  0 means record only initial + final.

        Returns
        -------
        traj : ndarray (n_rec, N)
            Recorded states (N = 8 or 24).
        final_state : ndarray (N,)
        lambda_final : float
        n_steps_actual : int
        """
        state0 = np.ascontiguousarray(state0, dtype=np.float64)

        if state0.shape[0] == 8 and not self.lensing:
            traj, final, n_rec, lam, steps = _integrate_loop_8(
                state0, float(dt), int(n_steps),
                float(lambda_stop), int(record_every),
                self.origins, self.uppers, self.spacings, self.shapes,
                self.fields, self.P,
                self.eta_min, self.inv_deta, self.a_tab, self.adot_tab,
                self.c_val, self.slow_roll,
            )
        elif state0.shape[0] == 24 and self.lensing:
            traj, final, n_rec, lam, steps = _integrate_loop_24(
                state0, float(dt), int(n_steps),
                float(lambda_stop), int(record_every),
                self.origins, self.uppers, self.spacings, self.shapes,
                self.fields, self.P,
                self.eta_min, self.inv_deta, self.a_tab, self.adot_tab,
                self.H_tab, self.Hp_tab,
                self.c_val, self.slow_roll,
            )
        else:
            raise ValueError(
                f"State size {state0.shape[0]} incompatible with lensing={self.lensing}"
            )

        return traj, final, lam, steps

    # ----------------------------------------------------------------
    def warmup(self):
        """Trigger numba compilation with a dummy state (8 + 24)."""
        s8 = np.zeros(8)
        s8[0] = 0.5 * (self.eta_min + self.eta_max)
        s8[1] = float(self.origins[-1, 0] + 0.5 * (self.uppers[-1, 0] - self.origins[-1, 0]))
        s8[2] = float(self.origins[-1, 1] + 0.5 * (self.uppers[-1, 1] - self.origins[-1, 1]))
        s8[3] = float(self.origins[-1, 2] + 0.5 * (self.uppers[-1, 2] - self.origins[-1, 2]))
        s8[4] = 1.0
        _integrate_loop_8(
            s8, 1e10, 2, 0.0, 0,
            self.origins, self.uppers, self.spacings, self.shapes,
            self.fields, self.P,
            self.eta_min, self.inv_deta, self.a_tab, self.adot_tab,
            self.c_val, self.slow_roll,
        )
        if self.lensing:
            s24 = np.zeros(24)
            s24[0:8] = s8
            s24[8] = 0.0; s24[9] = 1.0           # e1 = (0,1,0,0)
            s24[12] = 0.0; s24[14] = 1.0          # e2 = (0,0,1,0)
            # D = 0, P = I
            s24[16] = 0.0; s24[19] = 0.0
            s24[20] = 1.0; s24[23] = 1.0
            _integrate_loop_24(
                s24, 1e10, 2, 0.0, 0,
                self.origins, self.uppers, self.spacings, self.shapes,
                self.fields, self.P,
                self.eta_min, self.inv_deta, self.a_tab, self.adot_tab,
                self.H_tab, self.Hp_tab,
                self.c_val, self.slow_roll,
            )


# ======================================================================
#  Photon convenience wrapper
# ======================================================================

def integrate_photon_numba(
    photon, backend, dt, n_steps,
    lambda_stop=0.0, record_every=1,
):
    """
    Integrate a single Photon with the numba backend and write trajectory
    back into photon.history / photon.x / photon.u (+ lensing fields if 24-comp).

    Parameters
    ----------
    photon : Photon
        Must expose .x (4,), .u (4,), .history (appendable), and for lensing
        .e1, .e2, .D_flat, .P_flat.
    backend : NumbaAMRBackend
    dt : float
        Affine step (signed).
    n_steps : int
    lambda_stop : float
        If > 0, stop when |lambda| reaches this value.
    record_every : int
        0 = initial + final only; 1 = every step; N = every N steps.

    Returns
    -------
    traj : ndarray (n_rec, N)
    lambda_final : float
    n_steps_actual : int
    """
    x = np.asarray(photon.x, dtype=np.float64)
    u = np.asarray(photon.u, dtype=np.float64)

    if backend.lensing:
        e1 = np.asarray(getattr(photon, "e1"), dtype=np.float64)
        e2 = np.asarray(getattr(photon, "e2"), dtype=np.float64)
        D_flat = np.asarray(getattr(photon, "D_flat", [0, 0, 0, 0]), dtype=np.float64)
        P_flat = np.asarray(getattr(photon, "P_flat", [1, 0, 0, 1]), dtype=np.float64)
        state0 = np.concatenate([x, u, e1, e2, D_flat, P_flat])
    else:
        state0 = np.concatenate([x, u])

    traj, final, lam, steps = backend.integrate_rk4(
        state0, dt, n_steps, lambda_stop, record_every,
    )

    # Write back to photon
    photon.x = final[0:4].copy()
    photon.u = final[4:8].copy()
    if backend.lensing and final.shape[0] >= 24:
        photon.e1 = final[8:12].copy()
        photon.e2 = final[12:16].copy()
        photon.D_flat = final[16:20].copy()
        photon.P_flat = final[20:24].copy()
    photon.lambda_affine = lam

    # Replay trajectory into photon.history (compatible with list / History objects)
    if hasattr(photon, "history"):
        hist = photon.history
        for row in traj:
            if hasattr(hist, "append"):
                hist.append(row.copy())
            else:
                # assume it's a list-like
                hist.append(row.copy())

    return traj, lam, steps
