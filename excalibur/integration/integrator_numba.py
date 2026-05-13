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
    - compiled RK4 / RKF45 / Dormand-Prince 5(4) schemes

The entire per-step hot-path runs under @njit: AMR patch lookup, tricubic
interpolation, Christoffel assembly, Riemann blocks, optical tidal matrix,
Jacobi RHS, and the integration schemes themselves. No Python object calls
inside the compiled stages.

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
        integrator="dopri5",
    )
    traj, final_state, lam = backend.integrate(
        state0_24, dt=dt_fine, n_steps=N,
    )
"""
from __future__ import annotations

import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from excalibur.core.constants import G
from excalibur.grid.interpolator_4d_fast import _fused_3d_tricubic_clamp
from excalibur.metrics.perturbed_flrw_metric_fast import compute_tensorial_acceleration
from excalibur.observables.riemann_perturbed_flrw import riemann_blocks_kernel
from excalibur.observables.sachs_basis import screen_projected_sachs_transport_rhs
from excalibur.observables.optical_tidal_matrix import (
    optical_tidal_matrix_optimized,
    jacobi_rhs,
)
from excalibur.integration.integrator_numba_schemes import (
    integrate_numba_scheme,
    STOP_REASON_UNKNOWN,
    normalize_numba_integrator_name,
    numba_integrator_is_adaptive,
    numba_integrator_scheme_id,
)


_SCREEN_MODE_METRIC = 0
_SCREEN_MODE_CONFORMAL = 1

_BYPASS_MODE_NONE = 0
_BYPASS_MODE_NFW = 1


# ======================================================================
#  Eta-table interpolation
# ======================================================================

@njit(cache=True, fastmath=True)
def _interp_eta_table(eta, eta_min, inv_deta, table):
    """Linear interpolation on a uniform eta grid."""
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


@njit(cache=True, fastmath=True)
def _metric_dot3(lhs, rhs, g_ij):
    s = 0.0
    for i in range(3):
        for j in range(3):
            s += lhs[i] * g_ij[i, j] * rhs[j]
    return s


@njit(cache=True, fastmath=True)
def _pick_seed_metric(direction, g_ij):
    seeds = np.eye(3)
    best_idx = 0
    best_dot = abs(_metric_dot3(seeds[0], direction, g_ij))
    for idx in range(1, 3):
        dot_val = abs(_metric_dot3(seeds[idx], direction, g_ij))
        if dot_val < best_dot:
            best_dot = dot_val
            best_idx = idx
    return seeds[best_idx].copy()


@njit(cache=True, fastmath=True)
def _cross3(lhs, rhs):
    out = np.empty(3)
    out[0] = lhs[1] * rhs[2] - lhs[2] * rhs[1]
    out[1] = lhs[2] * rhs[0] - lhs[0] * rhs[2]
    out[2] = lhs[0] * rhs[1] - lhs[1] * rhs[0]
    return out


@njit(cache=True, fastmath=True)
def _build_transverse_basis_metric_numba(k_spatial, g_ij):
    k_norm_sq = _metric_dot3(k_spatial, k_spatial, g_ij)
    if k_norm_sq <= 0.0:
        raise ValueError("Spatial photon momentum has non-positive metric norm squared.")

    n_hat = k_spatial / np.sqrt(k_norm_sq)
    best_seed = _pick_seed_metric(n_hat, g_ij)

    v1 = best_seed.copy()
    v1 -= (_metric_dot3(v1, n_hat, g_ij) / _metric_dot3(n_hat, n_hat, g_ij)) * n_hat
    v1 /= np.sqrt(_metric_dot3(v1, v1, g_ij))

    v2 = _cross3(n_hat, v1)
    v2 -= (_metric_dot3(v2, n_hat, g_ij) / _metric_dot3(n_hat, n_hat, g_ij)) * n_hat
    v2 -= (_metric_dot3(v2, v1, g_ij) / _metric_dot3(v1, v1, g_ij)) * v1
    v2_norm = np.sqrt(_metric_dot3(v2, v2, g_ij))
    if v2_norm < 1e-15:
        raise ValueError("Degenerate Sachs basis: v2 has zero metric norm.")
    v2 /= v2_norm
    return v1, v2


@njit(cache=True, fastmath=True)
def _init_sachs_basis_numba(k_mu, g_mu_nu, a, screen_mode):
    k_spatial = k_mu[1:4].copy()

    if screen_mode == _SCREEN_MODE_CONFORMAL:
        screen_g_mu_nu = g_mu_nu / (a * a)
        lift_g_mu_nu = screen_g_mu_nu
    else:
        screen_g_mu_nu = g_mu_nu
        lift_g_mu_nu = g_mu_nu

    screen_g_ij = screen_g_mu_nu[1:4, 1:4]
    v1, v2 = _build_transverse_basis_metric_numba(k_spatial, screen_g_ij)

    lift_g_ij = lift_g_mu_nu[1:4, 1:4]
    g00 = lift_g_mu_nu[0, 0]
    k0 = k_mu[0]
    denom = g00 * k0
    if abs(denom) < 1e-30:
        raise ValueError("g_00 * k0 is too small to lift Sachs vectors.")

    dot1 = _metric_dot3(v1, k_spatial, lift_g_ij)
    dot2 = _metric_dot3(v2, k_spatial, lift_g_ij)

    e1 = np.zeros(4)
    e2 = np.zeros(4)
    e1[0] = -dot1 / denom
    e2[0] = -dot2 / denom
    e1[1:4] = v1
    e2[1:4] = v2
    return e1, e2


@njit(cache=True, fastmath=True)
def _optical_tidal_matrix_local_screen_numba(
    Rd_k00l, Rd_0lki, Rd_kijl,
    k_mu, g_mu_nu, a, screen_mode,
):
    e1_mu, e2_mu = _init_sachs_basis_numba(k_mu, g_mu_nu, a, screen_mode)
    return optical_tidal_matrix_optimized(
        Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu, g_mu_nu,
    )


@njit(cache=True, fastmath=True)
def _nfw_value_gradient_hessian(x, y, z, center, r_s, rho_s):
    dx = x - center[0]
    dy = y - center[1]
    dz = z - center[2]
    r2 = dx * dx + dy * dy + dz * dz
    r_min = 1e-6 * r_s
    r_min2 = r_min * r_min
    if r2 < r_min2:
        r2 = r_min2
    r = np.sqrt(r2)
    s = r / r_s

    prefac = 4.0 * np.pi * G * rho_s * r_s * r_s * r_s
    phi = -prefac * np.log(1.0 + s) / r

    enclosed_mass = 4.0 * np.pi * rho_s * r_s * r_s * r_s * (
        np.log(1.0 + s) - s / (1.0 + s)
    )
    gm_over_r3 = G * enclosed_mass / (r2 * r)
    gx = gm_over_r3 * dx
    gy = gm_over_r3 * dy
    gz = gm_over_r3 * dz

    rho = rho_s / (s * (1.0 + s) * (1.0 + s))
    radial_coef = 4.0 * np.pi * G * rho - 3.0 * gm_over_r3

    hxx = radial_coef * dx * dx / r2 + gm_over_r3
    hyy = radial_coef * dy * dy / r2 + gm_over_r3
    hzz = radial_coef * dz * dz / r2 + gm_over_r3
    hxy = radial_coef * dx * dy / r2
    hxz = radial_coef * dx * dz / r2
    hyz = radial_coef * dy * dz / r2
    return phi, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz


@njit(cache=True, fastmath=True)
def _nfw_sum_value_gradient_hessian(x, y, z, centers, r_s_values, rho_s_values):
    phi = 0.0
    gx = 0.0
    gy = 0.0
    gz = 0.0
    hxx = 0.0
    hyy = 0.0
    hzz = 0.0
    hxy = 0.0
    hxz = 0.0
    hyz = 0.0

    n_components = centers.shape[0]
    for i in range(n_components):
        (
            phi_i,
            gx_i,
            gy_i,
            gz_i,
            hxx_i,
            hyy_i,
            hzz_i,
            hxy_i,
            hxz_i,
            hyz_i,
        ) = _nfw_value_gradient_hessian(
            x,
            y,
            z,
            centers[i],
            r_s_values[i],
            rho_s_values[i],
        )
        phi += phi_i
        gx += gx_i
        gy += gy_i
        gz += gz_i
        hxx += hxx_i
        hyy += hyy_i
        hzz += hzz_i
        hxy += hxy_i
        hxz += hxz_i
        hyz += hyz_i

    return phi, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz


@njit(cache=True, fastmath=True)
def _value_gradient_hessian_with_optional_bypass(
    x, y, z,
    origins, uppers, spacings, shapes, fields, P,
    bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
):
    if bypass_mode == _BYPASS_MODE_NFW:
        dx = x - bypass_center[0]
        dy = y - bypass_center[1]
        dz = z - bypass_center[2]
        if dx * dx + dy * dy + dz * dz < bypass_r2:
            return _nfw_sum_value_gradient_hessian(
                x, y, z, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
            )

    return _amr_tricubic_clamp(
        x, y, z, origins, uppers, spacings, shapes, fields, P,
    )


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


@njit(cache=True, fastmath=True)
def _build_metric_tensor_flrw(scale_factor, phi_n, c_val):
    g_mu_nu = np.zeros((4, 4))
    scale2 = scale_factor * scale_factor
    c2 = c_val * c_val
    g_mu_nu[0, 0] = -scale2 * (1.0 + 2.0 * phi_n) * c2
    g_mu_nu[1, 1] = scale2 * (1.0 - 2.0 * phi_n)
    g_mu_nu[2, 2] = scale2 * (1.0 - 2.0 * phi_n)
    g_mu_nu[3, 3] = scale2 * (1.0 - 2.0 * phi_n)
    return g_mu_nu


# ======================================================================
#  8-component geodesic RHS (no lensing)
# ======================================================================

@njit(cache=True, fastmath=True)
def _geodesic_rhs_8(
    state,
    origins, uppers, spacings, shapes, fields, P,
    eta_min, inv_deta, a_tab, adot_tab,
    c_val, slow_roll, screen_mode,
    bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
):
    eta = state[0]
    x = state[1]; y = state[2]; z = state[3]
    u0 = state[4]; u1 = state[5]; u2 = state[6]; u3 = state[7]

    a = _interp_eta_table(eta, eta_min, inv_deta, a_tab)
    adot = _interp_eta_table(eta, eta_min, inv_deta, adot_tab)

    val, gx, gy, gz, _hxx, _hyy, _hzz, _hxy, _hxz, _hyz = _value_gradient_hessian_with_optional_bypass(
        x, y, z, origins, uppers, spacings, shapes, fields, P,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
    )

    c2 = c_val * c_val
    phi_n = val / c2
    phi_dot_n = 0.0  # slow_roll assumed; non-slow-roll falls back to Python path

    if screen_mode == _SCREEN_MODE_CONFORMAL:
        geo_a = 1.0
        geo_adot = 0.0
    else:
        geo_a = a
        geo_adot = adot

    du0, du1, du2, du3 = compute_tensorial_acceleration(
        u0, u1, u2, u3, geo_a, geo_adot, phi_n, gx, gy, gz, phi_dot_n, c_val,
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
    c_val, slow_roll, screen_mode,
    bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
):
    eta = state[0]
    x = state[1]; y = state[2]; z = state[3]
    u0 = state[4]; u1 = state[5]; u2 = state[6]; u3 = state[7]

    a, adot, H_conf, H_prime = _a_adot_H_Hp(
        eta, eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
    )

    val, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz = _value_gradient_hessian_with_optional_bypass(
        x, y, z, origins, uppers, spacings, shapes, fields, P,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
    )

    c2 = c_val * c_val
    phi_n = val / c2
    phi_dot_n = 0.0          # slow_roll
    phi_dot_SI = 0.0
    phi_ddot = 0.0

    if screen_mode == _SCREEN_MODE_CONFORMAL:
        geo_a = 1.0
        geo_adot = 0.0
        optic_a = 1.0
        optic_adot = 0.0
        optic_H_conf = 0.0
        optic_H_prime = 0.0
    else:
        geo_a = a
        geo_adot = adot
        optic_a = a
        optic_adot = adot
        optic_H_conf = H_conf
        optic_H_prime = H_prime

    # --- Geodesic acceleration (8 comp) ---
    du0, du1, du2, du3 = compute_tensorial_acceleration(
        u0, u1, u2, u3, geo_a, geo_adot, phi_n, gx, gy, gz, phi_dot_n, c_val,
    )

    # --- Christoffel for Sachs transport ---
    k_mu = np.empty(4)
    k_mu[0] = u0; k_mu[1] = u1; k_mu[2] = u2; k_mu[3] = u3
    G = _christoffel_flrw(optic_a, optic_adot, val, gx, gy, gz, phi_dot_SI, c_val)

    # --- Metric tensor at current position ---
    g_mu_nu = _build_metric_tensor_flrw(optic_a, phi_n, c_val)
    transport_g_mu_nu = g_mu_nu

    e1 = state[8:12]
    e2 = state[12:16]
    de1 = screen_projected_sachs_transport_rhs(e1, G, k_mu, transport_g_mu_nu)
    de2 = screen_projected_sachs_transport_rhs(e2, G, k_mu, transport_g_mu_nu)

    # --- Riemann blocks ---
    grad_phi = np.empty(3); grad_phi[0] = gx; grad_phi[1] = gy; grad_phi[2] = gz
    grad_phi_dot = np.zeros(3)   # slow_roll
    hess_phi = np.empty((3, 3))
    hess_phi[0, 0] = hxx; hess_phi[1, 1] = hyy; hess_phi[2, 2] = hzz
    hess_phi[0, 1] = hxy; hess_phi[1, 0] = hxy
    hess_phi[0, 2] = hxz; hess_phi[2, 0] = hxz
    hess_phi[1, 2] = hyz; hess_phi[2, 1] = hyz

    Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
        optic_a, optic_H_conf, optic_H_prime,
        val, phi_dot_SI, phi_ddot,
        grad_phi, grad_phi_dot, hess_phi,
        c_val,
    )

    if screen_mode == _SCREEN_MODE_METRIC:
        R_AB = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g_mu_nu,
        )
    else:
        R_AB = _optical_tidal_matrix_local_screen_numba(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, g_mu_nu, optic_a, screen_mode,
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
                c_val, slow_roll, screen_mode,
                bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s):
    k1 = _geodesic_rhs_8(state, origins, uppers, spacings, shapes, fields, P,
                         eta_min, inv_deta, a_tab, adot_tab, c_val, slow_roll,
                         screen_mode,
                         bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s)
    s2 = state + 0.5 * dt * k1
    k2 = _geodesic_rhs_8(s2, origins, uppers, spacings, shapes, fields, P,
                         eta_min, inv_deta, a_tab, adot_tab, c_val, slow_roll,
                         screen_mode,
                         bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s)
    s3 = state + 0.5 * dt * k2
    k3 = _geodesic_rhs_8(s3, origins, uppers, spacings, shapes, fields, P,
                         eta_min, inv_deta, a_tab, adot_tab, c_val, slow_roll,
                         screen_mode,
                         bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s)
    s4 = state + dt * k3
    k4 = _geodesic_rhs_8(s4, origins, uppers, spacings, shapes, fields, P,
                         eta_min, inv_deta, a_tab, adot_tab, c_val, slow_roll,
                         screen_mode,
                         bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s)
    out = np.empty(8)
    for i in range(8):
        out[i] = state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
    return out


@njit(cache=True, fastmath=True)
def _rk4_step_24(state, dt,
                 origins, uppers, spacings, shapes, fields, P,
                 eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
                 c_val, slow_roll, screen_mode,
                 bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s):
    k1 = _geodesic_rhs_24(state, origins, uppers, spacings, shapes, fields, P,
                          eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
                          c_val, slow_roll, screen_mode,
                          bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s)
    s2 = state + 0.5 * dt * k1
    k2 = _geodesic_rhs_24(s2, origins, uppers, spacings, shapes, fields, P,
                          eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
                          c_val, slow_roll, screen_mode,
                          bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s)
    s3 = state + 0.5 * dt * k2
    k3 = _geodesic_rhs_24(s3, origins, uppers, spacings, shapes, fields, P,
                          eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
                          c_val, slow_roll, screen_mode,
                          bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s)
    s4 = state + dt * k3
    k4 = _geodesic_rhs_24(s4, origins, uppers, spacings, shapes, fields, P,
                          eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
                          c_val, slow_roll, screen_mode,
                          bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s)
    out = np.empty(24)
    for i in range(24):
        out[i] = state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
    return out


@njit(cache=True, fastmath=True)
def _integrate_loop_8(
    state0, dt, n_steps, lambda_stop, record_every,
    origins, uppers, spacings, shapes, fields, P,
    eta_min, inv_deta, a_tab, adot_tab,
    c_val, slow_roll, screen_mode,
    bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
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
                            c_val, slow_roll, screen_mode,
                            bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s)
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
    c_val, slow_roll, screen_mode,
    bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
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
                             c_val, slow_roll, screen_mode,
                             bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s)
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

    Configuration is frozen at construction time for the physics path
    (tricubic + clamp + slow_roll + analytical_geodesics=False), while the
    integration scheme itself is delegated to the compiled scheme module.

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
    sachs_screen_convention : str
        ``"metric"`` / ``"physical_metric"`` keep the historical online
        physical screen. ``"conformal_metric"`` evolves the full numba
        geodesic-plus-optics system in conformal variables.
    analytical_source : optional
        Analytic NFW-like source used for a local bypass inside the compiled
        kernels. This can be a single halo exposing ``center``, ``r_s``, and
        ``rho_s`` or a composite source exposing ``halos`` with those fields
        on each component.
    bypass_radius : float
        Radius of the local analytic bypass sphere in meters. Ignored when
        ``analytical_source`` is None or ``bypass_radius <= 0``.
    integrator : str
        ``"rk4"``, ``"rk45"`` (Fehlberg 4(5)), or ``"dopri5"``
        (Dormand-Prince 5(4)).
    rtol, atol : float
        Adaptive-step tolerances for ``rk45`` and ``dopri5``.
    dt_min, dt_max : float or None
        Optional step-size bounds. When omitted, adaptive schemes use the same
        defaults as the Python integrator: ``|dt|/1000`` and ``|dt|*50``.
    max_rejected : int
        Abort threshold for consecutive rejected adaptive steps.
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
        sachs_screen_convention="metric",
        analytical_source=None,
        bypass_radius=0.0,
        integrator="rk4",
        rtol=1e-9,
        atol=1e-13,
        dt_min=None,
        dt_max=None,
        max_rejected=200,
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
        self.integrator_name = normalize_numba_integrator_name(integrator)
        self.scheme_id = numba_integrator_scheme_id(self.integrator_name)
        self.is_adaptive = numba_integrator_is_adaptive(self.integrator_name)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.dt_min = None if dt_min is None else float(dt_min)
        self.dt_max = None if dt_max is None else float(dt_max)
        self.max_rejected = int(max_rejected)
        self.last_stop_reason = STOP_REASON_UNKNOWN
        self.last_n_steps_actual = 0
        self.last_rejected_total = 0
        self.last_rejected_streak_peak = 0
        self.last_min_dt_attempted = 0.0
        self.last_dt_attempted = 0.0
        self.last_dt_suggested = 0.0
        self.sachs_screen_convention = sachs_screen_convention
        if sachs_screen_convention in ("metric", "physical_metric"):
            self.sachs_screen_mode = _SCREEN_MODE_METRIC
        elif sachs_screen_convention == "conformal_metric":
            self.sachs_screen_mode = _SCREEN_MODE_CONFORMAL
        else:
            raise ValueError(
                "NumbaAMRBackend only supports 'metric', 'physical_metric', or 'conformal_metric' screen conventions"
            )

        self._build_patch_arrays(amr_grid, field_name)
        self._build_eta_tables(cosmology, n_eta_samples, eta_range)
        self._configure_bypass(analytical_source, bypass_radius)

    # ----------------------------------------------------------------
    def _configure_bypass(self, analytical_source, bypass_radius):
        self.bypass_mode = _BYPASS_MODE_NONE
        self.bypass_center = np.zeros(3, dtype=np.float64)
        self.bypass_r2 = 0.0
        self.bypass_nfw_centers = np.zeros((1, 3), dtype=np.float64)
        self.bypass_nfw_r_s = np.ones(1, dtype=np.float64)
        self.bypass_nfw_rho_s = np.zeros(1, dtype=np.float64)

        if analytical_source is None or bypass_radius <= 0.0:
            return

        if hasattr(analytical_source, "halos"):
            components = list(analytical_source.halos)
        else:
            components = [analytical_source]

        if not components:
            raise ValueError("NumbaAMRBackend analytical bypass requires at least one NFW component")

        required = ("center", "r_s", "rho_s")
        if not all(all(hasattr(component, attr) for attr in required) for component in components):
            raise ValueError(
                "NumbaAMRBackend analytical bypass requires NFW-like components exposing center, r_s, and rho_s"
            )

        self.bypass_mode = _BYPASS_MODE_NFW
        self.bypass_center = np.asarray(analytical_source.center, dtype=np.float64)
        self.bypass_r2 = float(bypass_radius) ** 2
        self.bypass_nfw_centers = np.ascontiguousarray(
            np.asarray([component.center for component in components], dtype=np.float64)
        )
        self.bypass_nfw_r_s = np.ascontiguousarray(
            np.asarray([component.r_s for component in components], dtype=np.float64)
        )
        self.bypass_nfw_rho_s = np.ascontiguousarray(
            np.asarray([component.rho_s for component in components], dtype=np.float64)
        )

    # ----------------------------------------------------------------
    def _rhs_args_8(self):
        return (
            self.origins,
            self.uppers,
            self.spacings,
            self.shapes,
            self.fields,
            self.P,
            self.eta_min,
            self.inv_deta,
            self.a_tab,
            self.adot_tab,
            self.c_val,
            self.slow_roll,
            self.sachs_screen_mode,
            self.bypass_mode,
            self.bypass_center,
            self.bypass_r2,
            self.bypass_nfw_centers,
            self.bypass_nfw_r_s,
            self.bypass_nfw_rho_s,
        )

    def _rhs_args_24(self):
        return (
            self.origins,
            self.uppers,
            self.spacings,
            self.shapes,
            self.fields,
            self.P,
            self.eta_min,
            self.inv_deta,
            self.a_tab,
            self.adot_tab,
            self.H_tab,
            self.Hp_tab,
            self.c_val,
            self.slow_roll,
            self.sachs_screen_mode,
            self.bypass_mode,
            self.bypass_center,
            self.bypass_r2,
            self.bypass_nfw_centers,
            self.bypass_nfw_r_s,
            self.bypass_nfw_rho_s,
        )

    def _resolve_dt_bounds(self, dt, is_adaptive):
        abs_dt = abs(float(dt))
        if self.dt_min is not None:
            dt_min = self.dt_min
        else:
            dt_min = abs_dt / 1000.0 if is_adaptive else abs_dt

        if self.dt_max is not None:
            dt_max = self.dt_max
        else:
            dt_max = abs_dt * 50.0 if is_adaptive else abs_dt

        return float(dt_min), float(dt_max)

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
    def integrate(
        self,
        state0,
        dt,
        n_steps,
        lambda_stop=0.0,
        record_every=0,
        integrator=None,
        rtol=None,
        atol=None,
        dt_min=None,
        dt_max=None,
    ):
        """
        Integrate using the configured numba scheme.

        Parameters
        ----------
        state0 : ndarray (8,) or (24,)
        dt : float
            Initial affine-parameter step (signed).
        n_steps : int
            Maximum number of accepted steps.
        lambda_stop : float
            If > 0, stop when |lambda| >= lambda_stop (final step clamped).
        record_every : int
            If > 0, record every N-th accepted step.  0 means record only the
            trajectory buffer default from the compiled backend.
        integrator : str or None
            Optional per-call scheme override.
        rtol, atol : float or None
            Optional per-call adaptive tolerances.
        dt_min, dt_max : float or None
            Optional per-call step-size bounds.

        Returns
        -------
        traj : ndarray (n_rec, N)
        final_state : ndarray (N,)
        lambda_final : float
        n_steps_actual : int
        """
        state0 = np.ascontiguousarray(state0, dtype=np.float64)

        scheme_name = self.integrator_name if integrator is None else normalize_numba_integrator_name(integrator)
        scheme_id = numba_integrator_scheme_id(scheme_name)
        is_adaptive = numba_integrator_is_adaptive(scheme_name)
        dt_min_eff, dt_max_eff = self._resolve_dt_bounds(dt, is_adaptive)
        if dt_min is not None:
            dt_min_eff = float(dt_min)
        if dt_max is not None:
            dt_max_eff = float(dt_max)
        rtol_eff = self.rtol if rtol is None else float(rtol)
        atol_eff = self.atol if atol is None else float(atol)

        if state0.shape[0] == 8 and not self.lensing:
            (
                traj,
                final,
                n_rec,
                lam,
                steps,
                stop_reason,
                rejected_total,
                rejected_streak_peak,
                min_dt_attempted,
                last_dt_attempted,
                last_dt_suggested,
            ) = integrate_numba_scheme(
                _geodesic_rhs_8,
                scheme_id,
                state0,
                float(dt),
                int(n_steps),
                float(lambda_stop),
                int(record_every),
                rtol_eff,
                atol_eff,
                dt_min_eff,
                dt_max_eff,
                self.max_rejected,
                *self._rhs_args_8(),
            )
        elif state0.shape[0] == 24 and self.lensing:
            (
                traj,
                final,
                n_rec,
                lam,
                steps,
                stop_reason,
                rejected_total,
                rejected_streak_peak,
                min_dt_attempted,
                last_dt_attempted,
                last_dt_suggested,
            ) = integrate_numba_scheme(
                _geodesic_rhs_24,
                scheme_id,
                state0,
                float(dt),
                int(n_steps),
                float(lambda_stop),
                int(record_every),
                rtol_eff,
                atol_eff,
                dt_min_eff,
                dt_max_eff,
                self.max_rejected,
                *self._rhs_args_24(),
            )
        else:
            raise ValueError(
                f"State size {state0.shape[0]} incompatible with lensing={self.lensing}"
            )

        self.last_stop_reason = int(stop_reason)
        self.last_n_steps_actual = int(steps)
        self.last_rejected_total = int(rejected_total)
        self.last_rejected_streak_peak = int(rejected_streak_peak)
        self.last_min_dt_attempted = float(min_dt_attempted)
        self.last_dt_attempted = float(last_dt_attempted)
        self.last_dt_suggested = float(last_dt_suggested)
        return traj, final, lam, steps

    # ----------------------------------------------------------------
    def integrate_rk4(self, state0, dt, n_steps, lambda_stop=0.0, record_every=0):
        """Backward-compatible alias for fixed-step RK4."""
        return self.integrate(
            state0,
            dt,
            n_steps,
            lambda_stop=lambda_stop,
            record_every=record_every,
            integrator="rk4",
        )

    # ----------------------------------------------------------------
    def warmup(self):
        """Trigger numba compilation with a dummy state (8 + 24)."""
        s8 = np.zeros(8)
        s8[0] = 0.5 * (self.eta_min + self.eta_max)
        s8[1] = float(self.origins[-1, 0] + 0.5 * (self.uppers[-1, 0] - self.origins[-1, 0]))
        s8[2] = float(self.origins[-1, 1] + 0.5 * (self.uppers[-1, 1] - self.origins[-1, 1]))
        s8[3] = float(self.origins[-1, 2] + 0.5 * (self.uppers[-1, 2] - self.origins[-1, 2]))
        s8[4] = 1.0
        s8[5] = 1.0
        if not self.lensing:
            self.integrate(s8, 1e10, 2, 0.0, 0)
        else:
            s24 = np.zeros(24)
            s24[0:8] = s8
            s24[8] = 0.0; s24[9] = 1.0           # e1 = (0,1,0,0)
            s24[12] = 0.0; s24[14] = 1.0          # e2 = (0,0,1,0)
            # D = 0, P = I
            s24[16] = 0.0; s24[19] = 0.0
            s24[20] = 1.0; s24[23] = 1.0
            self.integrate(s24, 1e10, 2, 0.0, 0)


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

    traj, final, lam, steps = backend.integrate(
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
    photon.integration_n_steps_actual = int(getattr(backend, "last_n_steps_actual", steps))
    photon.integration_stop_reason = int(getattr(backend, "last_stop_reason", STOP_REASON_UNKNOWN))
    photon.integration_rejected_total = int(getattr(backend, "last_rejected_total", 0))
    photon.integration_rejected_streak_peak = int(getattr(backend, "last_rejected_streak_peak", 0))
    photon.integration_dt_min_attempted = float(getattr(backend, "last_min_dt_attempted", abs(dt)))
    photon.integration_dt_last_attempted = float(getattr(backend, "last_dt_attempted", dt))
    photon.integration_dt_last_suggested = float(getattr(backend, "last_dt_suggested", dt))

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
