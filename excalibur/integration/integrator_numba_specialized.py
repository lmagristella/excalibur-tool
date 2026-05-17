"""Regime-specialized Numba integrator kernel.

Restricted to ``sachs_screen_convention = "conformal_metric"`` (the configuration
used by ``run_lensing_nfw_analytic.py``).  In that regime the wrapping
``lowalloc`` kernel already passes
    optic_a = 1.0,  geo_a = 1.0,  geo_adot = 0.0,
    optic_H_conf = 0.0,  optic_H_prime = 0.0,
and (matching the lowalloc reference) ``phi_dot_SI = 0.0``.  Several terms in
the Christoffel, Riemann blocks, and optical tidal matrix assembly are
therefore identically zero.  This kernel drops those branches; the physics is
unchanged.

Specifically, compared to ``integrator_numba_lowalloc``:
    * ``H_tab`` and ``Hp_tab`` look-ups are skipped (their contributions vanish).
    * Christoffel entries that are exactly zero in this regime
      (``G[0,0,0]``, ``G[0,i,i]``, ``G[i,i,0]=G[i,0,i]``) are dropped.
    * The metric tensor is built with ``a = 1``.
    * The Riemann block ``R_{0lki}`` is identically zero and is skipped
      both in the kernel and in the R_down assembly.
    * The diagonal scalar piece of ``R_{k00l}`` and the ``second_scalar``
      piece of ``R_{kijl}`` are dropped.
    * The Sachs basis is re-initialised every step into pre-allocated buffers
      (replicating the arithmetic of ``_init_sachs_basis_numba`` for
      ``screen_mode = CONFORMAL`` with ``a = 1``).

If the backend is configured with a non-conformal screen convention the
kernel raises and the caller should fall back to ``lowalloc``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from excalibur.integration.integrator_numba import (
    NumbaAMRBackend as _BaseNumbaAMRBackend,
    _SCREEN_MODE_CONFORMAL,
    _integrate_loop_8,
    _value_gradient_hessian_with_optional_bypass,
    integrate_photon_numba,
)
from excalibur.metrics.perturbed_flrw_metric_fast import compute_tensorial_acceleration
from excalibur.observables.optical_tidal_matrix import _set_riemann_sym


# ----------------------------------------------------------------------
#  Christoffel symbols  (a = 1, adot = 0, phi_dot = 0)
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _christoffel_specialized_inplace(G, gx, gy, gz, c_val):
    for mu in range(4):
        for nu in range(4):
            for sigma in range(4):
                G[mu, nu, sigma] = 0.0

    c_inv2 = 1.0 / (c_val * c_val)
    gxc = gx * c_inv2
    gyc = gy * c_inv2
    gzc = gz * c_inv2

    G[1, 0, 0] = gx
    G[2, 0, 0] = gy
    G[3, 0, 0] = gz

    G[1, 1, 1] = -gxc
    G[2, 2, 2] = -gyc
    G[3, 3, 3] = -gzc

    G[0, 0, 1] = gxc; G[0, 1, 0] = gxc
    G[0, 0, 2] = gyc; G[0, 2, 0] = gyc
    G[0, 0, 3] = gzc; G[0, 3, 0] = gzc

    G[1, 2, 2] = gxc; G[1, 3, 3] = gxc
    G[2, 1, 1] = gyc; G[2, 3, 3] = gyc
    G[3, 1, 1] = gzc; G[3, 2, 2] = gzc
    G[1, 1, 2] = -gyc; G[1, 2, 1] = -gyc
    G[1, 1, 3] = -gzc; G[1, 3, 1] = -gzc
    G[2, 2, 1] = -gxc; G[2, 1, 2] = -gxc
    G[2, 2, 3] = -gzc; G[2, 3, 2] = -gzc
    G[3, 3, 1] = -gxc; G[3, 1, 3] = -gxc
    G[3, 3, 2] = -gyc; G[3, 2, 3] = -gyc


# ----------------------------------------------------------------------
#  Metric tensor  (a = 1)
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _build_metric_a1_inplace(g_mu_nu, phi_n, c_val):
    for mu in range(4):
        for nu in range(4):
            g_mu_nu[mu, nu] = 0.0
    c2 = c_val * c_val
    diag = 1.0 - 2.0 * phi_n
    g_mu_nu[0, 0] = -(1.0 + 2.0 * phi_n) * c2
    g_mu_nu[1, 1] = diag
    g_mu_nu[2, 2] = diag
    g_mu_nu[3, 3] = diag


# ----------------------------------------------------------------------
#  Sachs basis  --  in-place re-init (conformal screen, a = 1)
# ----------------------------------------------------------------------
#
# Replicates the arithmetic of ``_init_sachs_basis_numba`` for
# ``screen_mode = CONFORMAL`` with ``a = 1`` using only pre-allocated buffers.
# Since ``screen_g_mu_nu = g_mu_nu / 1.0`` and ``lift_g_mu_nu = screen_g_mu_nu``,
# the active spatial sub-matrix is ``g_mu_nu[1:4, 1:4]`` itself.
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _metric_dot3_local(lx, ly, lz, rx, ry, rz, g11, g12, g13, g22, g23, g33):
    return (lx * g11 * rx + lx * g12 * ry + lx * g13 * rz
            + ly * g12 * rx + ly * g22 * ry + ly * g23 * rz
            + lz * g13 * rx + lz * g23 * ry + lz * g33 * rz)


@njit(cache=True, fastmath=True)
def _init_sachs_basis_conformal_a1_inplace(e1, e2, k_mu, g_mu_nu, v1, v2, n_hat):
    g11 = g_mu_nu[1, 1]
    g12 = g_mu_nu[1, 2]
    g13 = g_mu_nu[1, 3]
    g22 = g_mu_nu[2, 2]
    g23 = g_mu_nu[2, 3]
    g33 = g_mu_nu[3, 3]
    g00 = g_mu_nu[0, 0]
    k0 = k_mu[0]

    kx = k_mu[1]
    ky = k_mu[2]
    kz = k_mu[3]

    k_norm_sq = _metric_dot3_local(kx, ky, kz, kx, ky, kz, g11, g12, g13, g22, g23, g33)
    if k_norm_sq <= 0.0:
        raise ValueError("Spatial photon momentum has non-positive metric norm squared.")
    inv_kn = 1.0 / np.sqrt(k_norm_sq)
    nx = kx * inv_kn
    ny = ky * inv_kn
    nz = kz * inv_kn
    n_hat[0] = nx
    n_hat[1] = ny
    n_hat[2] = nz

    # pick_seed: argmin of |seed @ g @ n_hat| for seed in canonical axes.
    s0 = abs(g11 * nx + g12 * ny + g13 * nz)
    s1 = abs(g12 * nx + g22 * ny + g23 * nz)
    s2 = abs(g13 * nx + g23 * ny + g33 * nz)
    best = 0
    bv = s0
    if s1 < bv:
        bv = s1
        best = 1
    if s2 < bv:
        best = 2
    if best == 0:
        sx, sy, sz = 1.0, 0.0, 0.0
    elif best == 1:
        sx, sy, sz = 0.0, 1.0, 0.0
    else:
        sx, sy, sz = 0.0, 0.0, 1.0

    # v1 = best_seed -  (v1.g.n_hat / n_hat.g.n_hat) * n_hat,  then normalise.
    nn = _metric_dot3_local(nx, ny, nz, nx, ny, nz, g11, g12, g13, g22, g23, g33)
    s_dot_n = _metric_dot3_local(sx, sy, sz, nx, ny, nz, g11, g12, g13, g22, g23, g33)
    coef = s_dot_n / nn
    vx = sx - coef * nx
    vy = sy - coef * ny
    vz = sz - coef * nz
    v_norm_sq = _metric_dot3_local(vx, vy, vz, vx, vy, vz, g11, g12, g13, g22, g23, g33)
    inv_v = 1.0 / np.sqrt(v_norm_sq)
    vx *= inv_v
    vy *= inv_v
    vz *= inv_v
    v1[0] = vx
    v1[1] = vy
    v1[2] = vz

    # v2 = n_hat x v1 (Euclidean), then Gram-Schmidt against n_hat and v1.
    wx = ny * vz - nz * vy
    wy = nz * vx - nx * vz
    wz = nx * vy - ny * vx
    w_dot_n = _metric_dot3_local(wx, wy, wz, nx, ny, nz, g11, g12, g13, g22, g23, g33)
    coef = w_dot_n / nn
    wx -= coef * nx
    wy -= coef * ny
    wz -= coef * nz
    vv = _metric_dot3_local(vx, vy, vz, vx, vy, vz, g11, g12, g13, g22, g23, g33)
    w_dot_v = _metric_dot3_local(wx, wy, wz, vx, vy, vz, g11, g12, g13, g22, g23, g33)
    coef = w_dot_v / vv
    wx -= coef * vx
    wy -= coef * vy
    wz -= coef * vz
    w_norm_sq = _metric_dot3_local(wx, wy, wz, wx, wy, wz, g11, g12, g13, g22, g23, g33)
    if w_norm_sq < 1e-30:
        raise ValueError("Degenerate Sachs basis: v2 has zero metric norm.")
    inv_w = 1.0 / np.sqrt(w_norm_sq)
    wx *= inv_w
    wy *= inv_w
    wz *= inv_w
    v2[0] = wx
    v2[1] = wy
    v2[2] = wz

    # Lift to 4D: e_0 = -(g_ij v^j k^i) / (g_00 k^0).
    denom = g00 * k0
    if abs(denom) < 1e-30:
        raise ValueError("g_00 * k0 is too small to lift Sachs vectors.")
    dot1 = _metric_dot3_local(vx, vy, vz, kx, ky, kz, g11, g12, g13, g22, g23, g33)
    dot2 = _metric_dot3_local(wx, wy, wz, kx, ky, kz, g11, g12, g13, g22, g23, g33)
    inv_denom = 1.0 / denom
    e1[0] = -dot1 * inv_denom
    e1[1] = vx
    e1[2] = vy
    e1[3] = vz
    e2[0] = -dot2 * inv_denom
    e2[1] = wx
    e2[2] = wy
    e2[3] = wz


# ----------------------------------------------------------------------
#  Sachs transport with screen projection  (in-place)
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _screen_projected_transport_specialized_inplace(out, e_mu, G, k_mu, g_mu_nu, nabla_k_u):
    for mu in range(4):
        s = 0.0
        for nu in range(4):
            for sigma in range(4):
                s += G[mu, nu, sigma] * k_mu[sigma] * e_mu[nu]
        out[mu] = -s

    for mu in range(4):
        s = 0.0
        for sigma in range(4):
            s += G[mu, 0, sigma] * k_mu[sigma]
        nabla_k_u[mu] = s

    u_dot_k = 0.0
    for nu in range(4):
        u_dot_k += g_mu_nu[0, nu] * k_mu[nu]
    if abs(u_dot_k) < 1e-30:
        return

    e_dot_nabla_u = 0.0
    for mu in range(4):
        e_cov = 0.0
        for nu in range(4):
            e_cov += g_mu_nu[mu, nu] * e_mu[nu]
        e_dot_nabla_u += e_cov * nabla_k_u[mu]

    alpha = -e_dot_nabla_u / u_dot_k
    for mu in range(4):
        out[mu] += alpha * k_mu[mu]


# ----------------------------------------------------------------------
#  Riemann blocks  --  a = 1, H = 0, H' = 0
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _riemann_blocks_specialized_inplace(Rd_k00l, Rd_kijl, hess_phi, c_val):
    # Rd_k00l[k, l] = a^2 * (-hess_phi[k, l] + delta_kl * diag_scalar);
    # with a=1 and diag_scalar=0:
    for k in range(3):
        for l in range(3):
            Rd_k00l[k, l] = -hess_phi[k, l]

    c_inv2 = 1.0 / (c_val * c_val)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    val = 0.0
                    if k == j:
                        val += hess_phi[i, l]
                    if k == l:
                        val -= hess_phi[i, j]
                    if i == j:
                        val -= hess_phi[k, l]
                    if i == l:
                        val += hess_phi[k, j]
                    Rd_kijl[k, i, j, l] = c_inv2 * val


# ----------------------------------------------------------------------
#  Optical tidal matrix  --  block 2 ≡ 0, skipped.
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _optical_tidal_specialized_inplace(R_AB, R_down, T_down, Rd_k00l, Rd_kijl,
                                       k_mu, e1_mu, e2_mu):
    for a in range(4):
        for b in range(4):
            T_down[a, b] = 0.0
            for c in range(4):
                for d in range(4):
                    R_down[a, b, c, d] = 0.0

    for k in range(3):
        for l in range(3):
            _set_riemann_sym(R_down, k + 1, 0, 0, l + 1, Rd_k00l[k, l])

    for k in range(3):
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    _set_riemann_sym(R_down, k + 1, i + 1, j + 1, l + 1, Rd_kijl[k, i, j, l])

    for mu in range(4):
        for nu in range(4):
            s = 0.0
            for alpha in range(4):
                for beta in range(4):
                    s += R_down[mu, alpha, nu, beta] * k_mu[alpha] * k_mu[beta]
            T_down[mu, nu] = s

    s11 = 0.0
    s12 = 0.0
    s21 = 0.0
    s22 = 0.0
    for mu in range(4):
        e1m = e1_mu[mu]
        e2m = e2_mu[mu]
        for nu in range(4):
            t = T_down[mu, nu]
            s11 += t * e1m * e1_mu[nu]
            s12 += t * e1m * e2_mu[nu]
            s21 += t * e2m * e1_mu[nu]
            s22 += t * e2m * e2_mu[nu]
    R_AB[0, 0] = s11
    R_AB[0, 1] = s12
    R_AB[1, 0] = s21
    R_AB[1, 1] = s22


# ----------------------------------------------------------------------
#  Jacobi RHS  (same as lowalloc -- kept local for clarity)
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _jacobi_rhs_specialized_inplace(out, D_flat, R_AB):
    D11 = D_flat[0]; D12 = D_flat[1]
    D21 = D_flat[2]; D22 = D_flat[3]
    P11 = D_flat[4]; P12 = D_flat[5]
    P21 = D_flat[6]; P22 = D_flat[7]
    out[0] = P11
    out[1] = P12
    out[2] = P21
    out[3] = P22
    out[4] = -(R_AB[0, 0] * D11 + R_AB[0, 1] * D21)
    out[5] = -(R_AB[0, 0] * D12 + R_AB[0, 1] * D22)
    out[6] = -(R_AB[1, 0] * D11 + R_AB[1, 1] * D21)
    out[7] = -(R_AB[1, 0] * D12 + R_AB[1, 1] * D22)


# ----------------------------------------------------------------------
#  Geodesic + lensing RHS  (24-state)
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _geodesic_rhs_24_specialized(
    out,
    state,
    origins,
    uppers,
    spacings,
    shapes,
    fields,
    P,
    c_val,
    bypass_mode,
    bypass_center,
    bypass_r2,
    bypass_nfw_centers,
    bypass_nfw_r_s,
    bypass_nfw_rho_s,
    k_mu,
    G,
    g_mu_nu,
    de1,
    de2,
    nabla_k_u,
    hess_phi,
    Rd_k00l,
    Rd_kijl,
    R_down,
    T_down,
    R_AB,
    jrhs,
    sachs_v1,
    sachs_v2,
    sachs_nhat,
    e1_local,
    e2_local,
):
    # eta = state[0]; spatial = state[1:4]; u^mu = state[4:8]
    x = state[1]
    y = state[2]
    z = state[3]
    u0 = state[4]
    u1 = state[5]
    u2 = state[6]
    u3 = state[7]

    val, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz = _value_gradient_hessian_with_optional_bypass(
        x, y, z, origins, uppers, spacings, shapes, fields, P,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
    )

    c2 = c_val * c_val
    phi_n = val / c2

    du0, du1, du2, du3 = compute_tensorial_acceleration(
        u0, u1, u2, u3, 1.0, 0.0, phi_n, gx, gy, gz, 0.0, c_val,
    )

    k_mu[0] = u0
    k_mu[1] = u1
    k_mu[2] = u2
    k_mu[3] = u3

    _christoffel_specialized_inplace(G, gx, gy, gz, c_val)
    _build_metric_a1_inplace(g_mu_nu, phi_n, c_val)

    _screen_projected_transport_specialized_inplace(de1, state[8:12], G, k_mu, g_mu_nu, nabla_k_u)
    _screen_projected_transport_specialized_inplace(de2, state[12:16], G, k_mu, g_mu_nu, nabla_k_u)

    hess_phi[0, 0] = hxx
    hess_phi[1, 1] = hyy
    hess_phi[2, 2] = hzz
    hess_phi[0, 1] = hxy
    hess_phi[1, 0] = hxy
    hess_phi[0, 2] = hxz
    hess_phi[2, 0] = hxz
    hess_phi[1, 2] = hyz
    hess_phi[2, 1] = hyz

    _riemann_blocks_specialized_inplace(Rd_k00l, Rd_kijl, hess_phi, c_val)

    _init_sachs_basis_conformal_a1_inplace(
        e1_local, e2_local, k_mu, g_mu_nu, sachs_v1, sachs_v2, sachs_nhat,
    )

    _optical_tidal_specialized_inplace(
        R_AB, R_down, T_down, Rd_k00l, Rd_kijl,
        k_mu, e1_local, e2_local,
    )

    _jacobi_rhs_specialized_inplace(jrhs, state[16:24], R_AB)

    out[0] = u0
    out[1] = u1
    out[2] = u2
    out[3] = u3
    out[4] = du0
    out[5] = du1
    out[6] = du2
    out[7] = du3
    for i in range(4):
        out[8 + i] = de1[i]
        out[12 + i] = de2[i]
        out[16 + i] = jrhs[i]
        out[20 + i] = jrhs[4 + i]


# ----------------------------------------------------------------------
#  RK4 step  (24-state)
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _rk4_step_24_specialized(
    out,
    state,
    dt,
    origins,
    uppers,
    spacings,
    shapes,
    fields,
    P,
    c_val,
    bypass_mode,
    bypass_center,
    bypass_r2,
    bypass_nfw_centers,
    bypass_nfw_r_s,
    bypass_nfw_rho_s,
    k1, k2, k3, k4, s2, s3, s4,
    k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
    Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
    sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
):
    half_dt = 0.5 * dt
    sixth_dt = dt / 6.0

    _geodesic_rhs_24_specialized(
        k1, state,
        origins, uppers, spacings, shapes, fields, P, c_val,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
        sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
    )
    for i in range(24):
        s2[i] = state[i] + half_dt * k1[i]

    _geodesic_rhs_24_specialized(
        k2, s2,
        origins, uppers, spacings, shapes, fields, P, c_val,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
        sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
    )
    for i in range(24):
        s3[i] = state[i] + half_dt * k2[i]

    _geodesic_rhs_24_specialized(
        k3, s3,
        origins, uppers, spacings, shapes, fields, P, c_val,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
        sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
    )
    for i in range(24):
        s4[i] = state[i] + dt * k3[i]

    _geodesic_rhs_24_specialized(
        k4, s4,
        origins, uppers, spacings, shapes, fields, P, c_val,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
        sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
    )

    for i in range(24):
        out[i] = state[i] + sixth_dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


# ----------------------------------------------------------------------
#  Main integration loop  (24-state)
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _integrate_loop_24_specialized(
    state0,
    dt,
    n_steps,
    lambda_stop,
    record_every,
    origins,
    uppers,
    spacings,
    shapes,
    fields,
    P,
    c_val,
    bypass_mode,
    bypass_center,
    bypass_r2,
    bypass_nfw_centers,
    bypass_nfw_r_s,
    bypass_nfw_rho_s,
):
    state = state0.copy()
    state_next = np.empty(24)
    lam = 0.0

    if record_every <= 0:
        n_rec = 1
    else:
        n_rec = 1 + n_steps // record_every + 2
    traj = np.empty((n_rec, 24))
    traj[0] = state
    rec_idx = 1

    k1 = np.empty(24)
    k2 = np.empty(24)
    k3 = np.empty(24)
    k4 = np.empty(24)
    s2 = np.empty(24)
    s3 = np.empty(24)
    s4 = np.empty(24)
    k_mu = np.empty(4)
    G = np.empty((4, 4, 4))
    g_mu_nu = np.empty((4, 4))
    de1 = np.empty(4)
    de2 = np.empty(4)
    nabla_k_u = np.empty(4)
    hess_phi = np.empty((3, 3))
    Rd_k00l = np.empty((3, 3))
    Rd_kijl = np.empty((3, 3, 3, 3))
    R_down = np.empty((4, 4, 4, 4))
    T_down = np.empty((4, 4))
    R_AB = np.empty((2, 2))
    jrhs = np.empty(8)
    sachs_v1 = np.empty(3)
    sachs_v2 = np.empty(3)
    sachs_nhat = np.empty(3)
    e1_local = np.empty(4)
    e2_local = np.empty(4)

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

        _rk4_step_24_specialized(
            state_next, state, dt_eff,
            origins, uppers, spacings, shapes, fields, P, c_val,
            bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
            k1, k2, k3, k4, s2, s3, s4,
            k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
            Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
            sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
        )
        lam += dt_eff
        step += 1

        state_tmp = state
        state = state_next
        state_next = state_tmp

        if record_every > 0 and (step % record_every) == 0 and rec_idx < n_rec:
            traj[rec_idx] = state
            rec_idx += 1

    if rec_idx < n_rec:
        traj[rec_idx] = state
        rec_idx += 1

    return traj[:rec_idx], state, rec_idx, lam, step


# ----------------------------------------------------------------------
#  Dormand-Prince 5(4) adaptive integrator (FSAL) -- 24-state
# ----------------------------------------------------------------------
#
# Butcher tableau (Dormand & Prince, J. Comp. Appl. Math. 6 (1980) 19).
# 7-stage explicit RK with first-same-as-last (FSAL): after an accepted
# step k7 = f(t+h, y_new) is reused as k1 of the next step.
#
# 5th-order solution coefficients (b) coincide with the 7th-row a coefficients
# (FSAL). The error estimate uses the difference (b - b_hat) where b_hat are
# the 4th-order embedded coefficients.

# c_i
_DP_C2 = 1.0 / 5.0
_DP_C3 = 3.0 / 10.0
_DP_C4 = 4.0 / 5.0
_DP_C5 = 8.0 / 9.0

# a_ij
_DP_A21 = 1.0 / 5.0

_DP_A31 = 3.0 / 40.0
_DP_A32 = 9.0 / 40.0

_DP_A41 = 44.0 / 45.0
_DP_A42 = -56.0 / 15.0
_DP_A43 = 32.0 / 9.0

_DP_A51 = 19372.0 / 6561.0
_DP_A52 = -25360.0 / 2187.0
_DP_A53 = 64448.0 / 6561.0
_DP_A54 = -212.0 / 729.0

_DP_A61 = 9017.0 / 3168.0
_DP_A62 = -355.0 / 33.0
_DP_A63 = 46732.0 / 5247.0
_DP_A64 = 49.0 / 176.0
_DP_A65 = -5103.0 / 18656.0

_DP_A71 = 35.0 / 384.0
_DP_A73 = 500.0 / 1113.0
_DP_A74 = 125.0 / 192.0
_DP_A75 = -2187.0 / 6784.0
_DP_A76 = 11.0 / 84.0

# b - b_hat (error coefficients).
_DP_E1 = 71.0 / 57600.0
_DP_E3 = -71.0 / 16695.0
_DP_E4 = 71.0 / 1920.0
_DP_E5 = -17253.0 / 339200.0
_DP_E6 = 22.0 / 525.0
_DP_E7 = -1.0 / 40.0


@njit(cache=True, fastmath=True)
def _dopri5_step_24_specialized(
    state, dt,
    origins, uppers, spacings, shapes, fields, P, c_val,
    bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
    k1, k2, k3, k4, k5, k6, k7,
    s2, s3, s4, s5, s6, y_new, err,
    k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
    Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
    sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
    have_k1,
):
    """Single DOPRI5 step.

    If ``have_k1`` is True, ``k1`` is assumed to already hold f(t, state)
    (FSAL reuse from the previous accepted step). Otherwise ``k1`` is computed.

    Writes ``y_new`` (5th-order solution) and ``err`` (b - b_hat) * dt.
    Final ``k7 = f(t+dt, y_new)`` is left in place for FSAL reuse.
    """
    if not have_k1:
        _geodesic_rhs_24_specialized(
            k1, state,
            origins, uppers, spacings, shapes, fields, P, c_val,
            bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
            k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
            Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
            sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
        )

    for i in range(24):
        s2[i] = state[i] + dt * (_DP_A21 * k1[i])
    _geodesic_rhs_24_specialized(
        k2, s2,
        origins, uppers, spacings, shapes, fields, P, c_val,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
        sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
    )

    for i in range(24):
        s3[i] = state[i] + dt * (_DP_A31 * k1[i] + _DP_A32 * k2[i])
    _geodesic_rhs_24_specialized(
        k3, s3,
        origins, uppers, spacings, shapes, fields, P, c_val,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
        sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
    )

    for i in range(24):
        s4[i] = state[i] + dt * (_DP_A41 * k1[i] + _DP_A42 * k2[i] + _DP_A43 * k3[i])
    _geodesic_rhs_24_specialized(
        k4, s4,
        origins, uppers, spacings, shapes, fields, P, c_val,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
        sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
    )

    for i in range(24):
        s5[i] = state[i] + dt * (_DP_A51 * k1[i] + _DP_A52 * k2[i]
                                 + _DP_A53 * k3[i] + _DP_A54 * k4[i])
    _geodesic_rhs_24_specialized(
        k5, s5,
        origins, uppers, spacings, shapes, fields, P, c_val,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
        sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
    )

    for i in range(24):
        s6[i] = state[i] + dt * (_DP_A61 * k1[i] + _DP_A62 * k2[i]
                                 + _DP_A63 * k3[i] + _DP_A64 * k4[i]
                                 + _DP_A65 * k5[i])
    _geodesic_rhs_24_specialized(
        k6, s6,
        origins, uppers, spacings, shapes, fields, P, c_val,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
        sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
    )

    # 5th-order solution: b = a_7 (FSAL)
    for i in range(24):
        y_new[i] = state[i] + dt * (_DP_A71 * k1[i]
                                    + _DP_A73 * k3[i] + _DP_A74 * k4[i]
                                    + _DP_A75 * k5[i] + _DP_A76 * k6[i])

    _geodesic_rhs_24_specialized(
        k7, y_new,
        origins, uppers, spacings, shapes, fields, P, c_val,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
        sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
    )

    # err = dt * (b - b_hat) . k_i
    for i in range(24):
        err[i] = dt * (_DP_E1 * k1[i] + _DP_E3 * k3[i] + _DP_E4 * k4[i]
                       + _DP_E5 * k5[i] + _DP_E6 * k6[i] + _DP_E7 * k7[i])


@njit(cache=True, fastmath=True)
def _integrate_dopri5_24_specialized(
    state0,
    dt_init,
    lambda_stop,
    rtol,
    atol,
    dt_min,
    dt_max,
    max_steps,
    record_every,
    origins,
    uppers,
    spacings,
    shapes,
    fields,
    P,
    c_val,
    bypass_mode,
    bypass_center,
    bypass_r2,
    bypass_nfw_centers,
    bypass_nfw_r_s,
    bypass_nfw_rho_s,
):
    """Adaptive 24-state lensing integration with PI step controller.

    Error norm is computed over the Jacobi block ``state[16:24]`` only --
    those are the lensing observables we care to control. The geodesic
    sub-block evolves smoothly and never drives the step.

    Returns
    -------
    traj : (n_rec, 24) ndarray
    final : (24,) ndarray
    n_rec : int
    lam_final : float
    n_accept : int
    n_reject : int
    """
    state = state0.copy()
    state_next = np.empty(24)
    lam = 0.0
    sign = 1.0 if dt_init >= 0.0 else -1.0
    h = abs(dt_init)
    h_max = abs(dt_max) if dt_max > 0.0 else 1.0 * abs(lambda_stop) if lambda_stop > 0.0 else 1.0e30
    h_min = abs(dt_min) if dt_min > 0.0 else 0.0

    # Trajectory recording (every N accepted steps).
    if record_every <= 0:
        n_rec_max = 2
    else:
        # Bound by max_steps in the worst case.
        n_rec_max = 1 + max_steps // max(record_every, 1) + 2
    traj = np.empty((n_rec_max, 24))
    traj[0] = state
    rec_idx = 1

    # Scratch arrays (all heap-resident across the whole integration).
    k1 = np.empty(24); k2 = np.empty(24); k3 = np.empty(24); k4 = np.empty(24)
    k5 = np.empty(24); k6 = np.empty(24); k7 = np.empty(24)
    s2 = np.empty(24); s3 = np.empty(24); s4 = np.empty(24)
    s5 = np.empty(24); s6 = np.empty(24)
    err = np.empty(24)
    k_mu = np.empty(4)
    G = np.empty((4, 4, 4))
    g_mu_nu = np.empty((4, 4))
    de1 = np.empty(4); de2 = np.empty(4); nabla_k_u = np.empty(4)
    hess_phi = np.empty((3, 3))
    Rd_k00l = np.empty((3, 3))
    Rd_kijl = np.empty((3, 3, 3, 3))
    R_down = np.empty((4, 4, 4, 4))
    T_down = np.empty((4, 4))
    R_AB = np.empty((2, 2))
    jrhs = np.empty(8)
    sachs_v1 = np.empty(3); sachs_v2 = np.empty(3); sachs_nhat = np.empty(3)
    e1_local = np.empty(4); e2_local = np.empty(4)

    # PI controller parameters (Hairer, Solving ODEs I, eq IV.2.4).
    p_order = 5
    safety = 0.9
    fac_min = 0.2
    fac_max = 5.0
    alpha = 0.7 / p_order
    beta = 0.4 / p_order
    err_prev = 1.0  # used by the PI controller across steps

    have_k1 = False
    n_accept = 0
    n_reject = 0
    step_total = 0
    use_lam_stop = lambda_stop > 0.0

    while step_total < max_steps:
        if use_lam_stop and lam >= lambda_stop:
            break

        # Clamp h to remaining interval. The RHS is effectively autonomous in
        # the affine parameter (eta is a state component), so changing h does
        # not invalidate a cached k1.
        if use_lam_stop:
            remaining = lambda_stop - lam
            if h > remaining:
                h = remaining
        if h < h_min and h_min > 0.0:
            h = h_min
        if h > h_max:
            h = h_max

        dt = sign * h

        _dopri5_step_24_specialized(
            state, dt,
            origins, uppers, spacings, shapes, fields, P, c_val,
            bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
            k1, k2, k3, k4, k5, k6, k7,
            s2, s3, s4, s5, s6, state_next, err,
            k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
            Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
            sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
            have_k1,
        )
        step_total += 1

        # Error norm on the Jacobi block only (indices 16..23).
        err_sum = 0.0
        for i in range(16, 24):
            yi = state[i]
            yi_new = state_next[i]
            ai = abs(yi)
            bi = abs(yi_new)
            mx = ai if ai > bi else bi
            sc = atol + rtol * mx
            r = err[i] / sc
            err_sum += r * r
        err_norm = np.sqrt(err_sum / 8.0)

        # Treat NaN/inf as a failed step so we shrink and retry instead of
        # silently accepting a diverged state.
        if not (err_norm == err_norm) or err_norm > 1e30:
            err_norm = 1e30

        if err_norm <= 1.0:
            # Accepted.
            n_accept += 1
            lam += h

            # FSAL: copy k7 into k1 for next step.
            for i in range(24):
                k1[i] = k7[i]
                state[i] = state_next[i]
            have_k1 = True

            if record_every > 0 and (n_accept % record_every) == 0 and rec_idx < n_rec_max:
                traj[rec_idx] = state
                rec_idx += 1

            # PI step-size update.
            if err_norm < 1e-12:
                err_norm = 1e-12
            factor = safety * (err_norm ** (-alpha)) * (err_prev ** beta)
            if factor < fac_min:
                factor = fac_min
            elif factor > fac_max:
                factor = fac_max
            h *= factor
            err_prev = err_norm
        else:
            # Rejected. k1 stays valid (state unchanged), but FSAL k7 was for
            # the rejected y_new, so it cannot be reused as new k1 next time.
            # k1 = f(state) is still correct, however.
            n_reject += 1
            have_k1 = True

            factor = safety * (err_norm ** (-1.0 / p_order))
            if factor < fac_min:
                factor = fac_min
            h *= factor
            if h_min > 0.0 and h < h_min:
                h = h_min
            # If the step is still rejected at the dt floor, retrying with the
            # same h cannot make progress. Bail out so the caller can lower the
            # floor or discard the ray instead of burning the full step budget.
            if h_min > 0.0 and h <= h_min:
                break

    if rec_idx < n_rec_max:
        traj[rec_idx] = state
        rec_idx += 1

    return traj[:rec_idx], state, rec_idx, lam, n_accept, n_reject


# ----------------------------------------------------------------------
#  Parallel batch DOPRI5  (numba.prange over photons)
# ----------------------------------------------------------------------
#
# Each photon integration is fully independent: the only shared inputs are
# the grid (origins / uppers / spacings / shapes / fields / P) and the
# analytical-bypass parameters, which are read-only during integration.
# We dispatch one prange iteration per photon and allocate all scratch
# buffers inside the iteration body so they are thread-local.
#
# Trajectory recording is supported through a caller-provided fixed-size
# buffer ``traj_buf`` of shape ``(n_photons, traj_capacity, 24)``.  Set
# ``record_every <= 0`` to record only the final state.

@njit(cache=True, parallel=True, fastmath=True)
def _integrate_batch_dopri5_24_specialized(
    states0,
    dt_init,
    lambda_stop,
    rtol,
    atol,
    dt_min,
    dt_max,
    max_steps,
    record_every,
    traj_buf,
    origins,
    uppers,
    spacings,
    shapes,
    fields,
    P,
    c_val,
    bypass_mode,
    bypass_center,
    bypass_r2,
    bypass_nfw_centers,
    bypass_nfw_r_s,
    bypass_nfw_rho_s,
):
    """Parallel batch DOPRI5 integration over photons.

    Parameters
    ----------
    states0 : (n_photons, 24) float64
        Initial states.
    traj_buf : (n_photons, traj_capacity, 24) float64
        Pre-allocated trajectory buffer.  When ``record_every <= 0`` this is
        only written at the final state (slot 0).

    Returns
    -------
    finals : (n_photons, 24)
    lams : (n_photons,)
    n_accepts, n_rejects : (n_photons,) int64
    n_pts : (n_photons,) int64
        Number of trajectory slots actually filled in ``traj_buf``.
    """
    n_photons = states0.shape[0]
    traj_capacity = traj_buf.shape[1]

    finals = np.empty((n_photons, 24))
    lams = np.empty(n_photons)
    n_accepts = np.empty(n_photons, dtype=np.int64)
    n_rejects = np.empty(n_photons, dtype=np.int64)
    n_pts = np.empty(n_photons, dtype=np.int64)

    sign = 1.0 if dt_init >= 0.0 else -1.0
    h_init = abs(dt_init)
    h_max_in = abs(dt_max) if dt_max > 0.0 else (abs(lambda_stop) if lambda_stop > 0.0 else 1.0e30)
    h_min_in = abs(dt_min) if dt_min > 0.0 else 0.0
    use_lam_stop = lambda_stop > 0.0
    p_order = 5
    safety = 0.9
    fac_min = 0.2
    fac_max = 5.0
    alpha = 0.7 / p_order
    beta = 0.4 / p_order

    for p_idx in prange(n_photons):
        # ------------------------------------------------------------------
        # Thread-local scratch.
        # ------------------------------------------------------------------
        state = states0[p_idx].copy()
        state_next = np.empty(24)

        k1 = np.empty(24); k2 = np.empty(24); k3 = np.empty(24); k4 = np.empty(24)
        k5 = np.empty(24); k6 = np.empty(24); k7 = np.empty(24)
        s2 = np.empty(24); s3 = np.empty(24); s4 = np.empty(24)
        s5 = np.empty(24); s6 = np.empty(24)
        err = np.empty(24)
        k_mu = np.empty(4)
        G = np.empty((4, 4, 4))
        g_mu_nu = np.empty((4, 4))
        de1 = np.empty(4); de2 = np.empty(4); nabla_k_u = np.empty(4)
        hess_phi = np.empty((3, 3))
        Rd_k00l = np.empty((3, 3))
        Rd_kijl = np.empty((3, 3, 3, 3))
        R_down = np.empty((4, 4, 4, 4))
        T_down = np.empty((4, 4))
        R_AB = np.empty((2, 2))
        jrhs = np.empty(8)
        sachs_v1 = np.empty(3); sachs_v2 = np.empty(3); sachs_nhat = np.empty(3)
        e1_local = np.empty(4); e2_local = np.empty(4)

        lam = 0.0
        h = h_init
        err_prev = 1.0
        have_k1 = False
        n_acc = 0
        n_rej = 0
        step_total = 0
        rec_idx = 0

        # Slot 0 always carries the initial state.
        if traj_capacity > 0:
            for i in range(24):
                traj_buf[p_idx, 0, i] = state[i]
            rec_idx = 1

        while step_total < max_steps:
            if use_lam_stop and lam >= lambda_stop:
                break

            if use_lam_stop:
                remaining = lambda_stop - lam
                if h > remaining:
                    h = remaining
            if h < h_min_in and h_min_in > 0.0:
                h = h_min_in
            if h > h_max_in:
                h = h_max_in

            dt = sign * h

            _dopri5_step_24_specialized(
                state, dt,
                origins, uppers, spacings, shapes, fields, P, c_val,
                bypass_mode, bypass_center, bypass_r2, bypass_nfw_centers, bypass_nfw_r_s, bypass_nfw_rho_s,
                k1, k2, k3, k4, k5, k6, k7,
                s2, s3, s4, s5, s6, state_next, err,
                k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
                Rd_k00l, Rd_kijl, R_down, T_down, R_AB, jrhs,
                sachs_v1, sachs_v2, sachs_nhat, e1_local, e2_local,
                have_k1,
            )
            step_total += 1

            # Error norm on the Jacobi block only.
            err_sum = 0.0
            for i in range(16, 24):
                yi = state[i]
                yi_new = state_next[i]
                ai = abs(yi)
                bi = abs(yi_new)
                mx = ai if ai > bi else bi
                sc = atol + rtol * mx
                r = err[i] / sc
                err_sum += r * r
            err_norm = np.sqrt(err_sum / 8.0)

            if not (err_norm == err_norm) or err_norm > 1e30:
                err_norm = 1e30

            if err_norm <= 1.0:
                n_acc += 1
                lam += h

                for i in range(24):
                    k1[i] = k7[i]
                    state[i] = state_next[i]
                have_k1 = True

                if record_every > 0 and (n_acc % record_every) == 0 and rec_idx < traj_capacity:
                    for i in range(24):
                        traj_buf[p_idx, rec_idx, i] = state[i]
                    rec_idx += 1

                if err_norm < 1e-12:
                    err_norm = 1e-12
                factor = safety * (err_norm ** (-alpha)) * (err_prev ** beta)
                if factor < fac_min:
                    factor = fac_min
                elif factor > fac_max:
                    factor = fac_max
                h *= factor
                err_prev = err_norm
            else:
                n_rej += 1
                have_k1 = True

                factor = safety * (err_norm ** (-1.0 / p_order))
                if factor < fac_min:
                    factor = fac_min
                h *= factor
                if h_min_in > 0.0 and h < h_min_in:
                    h = h_min_in
                if h_min_in > 0.0 and h <= h_min_in:
                    break

        # Trailing final-state slot (mirrors the sequential kernel).
        if rec_idx < traj_capacity:
            for i in range(24):
                traj_buf[p_idx, rec_idx, i] = state[i]
            rec_idx += 1

        for i in range(24):
            finals[p_idx, i] = state[i]
        lams[p_idx] = lam
        n_accepts[p_idx] = n_acc
        n_rejects[p_idx] = n_rej
        n_pts[p_idx] = rec_idx

    return finals, lams, n_accepts, n_rejects, n_pts


# ----------------------------------------------------------------------
#  Photon wrapper for adaptive DOPRI5
# ----------------------------------------------------------------------

def integrate_photon_numba_dopri5(
    photon, backend, dt_init,
    lambda_stop, rtol=1e-5, atol=1e-8,
    dt_min=0.0, dt_max=0.0,
    max_steps=10_000_000, record_every=0,
):
    """Adaptive DOPRI5 wrapper mirroring ``integrate_photon_numba``."""
    x = np.asarray(photon.x, dtype=np.float64)
    u = np.asarray(photon.u, dtype=np.float64)

    if backend.lensing:
        e1 = np.asarray(getattr(photon, "e1"), dtype=np.float64)
        e2 = np.asarray(getattr(photon, "e2"), dtype=np.float64)
        D_flat = np.asarray(getattr(photon, "D_flat", [0, 0, 0, 0]), dtype=np.float64)
        P_flat = np.asarray(getattr(photon, "P_flat", [1, 0, 0, 1]), dtype=np.float64)
        state0 = np.concatenate([x, u, e1, e2, D_flat, P_flat])
    else:
        raise ValueError("integrate_photon_numba_dopri5 requires a lensing photon (24-state).")

    traj, final, lam, n_accept, n_reject = backend.integrate_dopri5(
        state0, dt_init, lambda_stop,
        rtol=rtol, atol=atol, dt_min=dt_min, dt_max=dt_max,
        max_steps=max_steps, record_every=record_every,
    )

    photon.x = final[0:4].copy()
    photon.u = final[4:8].copy()
    photon.e1 = final[8:12].copy()
    photon.e2 = final[12:16].copy()
    photon.D_flat = final[16:20].copy()
    photon.P_flat = final[20:24].copy()
    photon.lambda_affine = lam

    if hasattr(photon, "history"):
        hist = photon.history
        for row in traj:
            if hasattr(hist, "append"):
                hist.append(row.copy())
            else:
                hist += [row.copy()]

    return traj, lam, n_accept, n_reject


def integrate_photons_batch_dopri5(
    photons, backend, dt_init,
    lambda_stop, rtol=1e-6, atol=1e-9,
    dt_min=0.0, dt_max=0.0,
    max_steps=10_000_000, record_every=0,
    traj_capacity=None,
):
    """Adaptive DOPRI5 wrapper over a list of photons using ``numba.prange``.

    All photons are integrated in parallel in a single batch call.  Final
    state, history (if ``record_every > 0``) and ``lambda_affine`` are
    written back into each :class:`Photon`.

    Parameters
    ----------
    traj_capacity : int or None
        Maximum number of trajectory rows recorded per photon, including the
        leading slot for the initial state and the trailing slot for the
        final state.  When ``None`` it is derived from ``max_steps`` and
        ``record_every`` (1 + max_steps // record_every + 2 when recording,
        2 otherwise) -- the upper bound used by the sequential kernel.
    """
    if not backend.lensing:
        raise ValueError("integrate_photons_batch_dopri5 requires a lensing backend.")

    n_photons = len(photons)
    if n_photons == 0:
        return np.empty((0, 0, 24)), np.empty(0, dtype=np.int64), np.empty(0), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    states0 = np.empty((n_photons, 24), dtype=np.float64)
    for i, photon in enumerate(photons):
        x = np.asarray(photon.x, dtype=np.float64)
        u = np.asarray(photon.u, dtype=np.float64)
        e1 = np.asarray(getattr(photon, "e1"), dtype=np.float64)
        e2 = np.asarray(getattr(photon, "e2"), dtype=np.float64)
        D_flat = np.asarray(getattr(photon, "D_flat", [0, 0, 0, 0]), dtype=np.float64)
        P_flat = np.asarray(getattr(photon, "P_flat", [1, 0, 0, 1]), dtype=np.float64)
        states0[i, 0:4] = x
        states0[i, 4:8] = u
        states0[i, 8:12] = e1
        states0[i, 12:16] = e2
        states0[i, 16:20] = D_flat
        states0[i, 20:24] = P_flat

    if traj_capacity is None:
        if record_every > 0:
            traj_capacity = 1 + max_steps // max(record_every, 1) + 2
        else:
            traj_capacity = 2

    finals, lams, n_accepts, n_rejects, traj_buf, n_pts = backend.integrate_dopri5_batch(
        states0, dt_init, lambda_stop,
        rtol=rtol, atol=atol, dt_min=dt_min, dt_max=dt_max,
        max_steps=max_steps, record_every=record_every,
        traj_capacity=traj_capacity,
    )

    for i, photon in enumerate(photons):
        photon.x = finals[i, 0:4].copy()
        photon.u = finals[i, 4:8].copy()
        photon.e1 = finals[i, 8:12].copy()
        photon.e2 = finals[i, 12:16].copy()
        photon.D_flat = finals[i, 16:20].copy()
        photon.P_flat = finals[i, 20:24].copy()
        photon.lambda_affine = lams[i]

        if hasattr(photon, "history"):
            hist = photon.history
            n = int(n_pts[i])
            for k in range(n):
                row = traj_buf[i, k].copy()
                if hasattr(hist, "append"):
                    hist.append(row)
                else:
                    hist += [row]

    return traj_buf, n_pts, lams, n_accepts, n_rejects


# ----------------------------------------------------------------------
#  Backend wrapper
# ----------------------------------------------------------------------

class NumbaAMRBackend(_BaseNumbaAMRBackend):
    """Conformal-screen specialised 24-state lensing kernel."""

    def integrate_rk4(self, state0, dt, n_steps, lambda_stop=0.0, record_every=0):
        state0 = np.ascontiguousarray(state0, dtype=np.float64)

        if state0.shape[0] == 24 and self.lensing:
            if self.sachs_screen_mode != _SCREEN_MODE_CONFORMAL:
                raise ValueError(
                    "integrator_numba_specialized requires sachs_screen_convention "
                    "= 'conformal_metric'. Use the lowalloc backend for other modes."
                )
            traj, final, _n_rec, lam, steps = _integrate_loop_24_specialized(
                state0,
                float(dt),
                int(n_steps),
                float(lambda_stop),
                int(record_every),
                self.origins,
                self.uppers,
                self.spacings,
                self.shapes,
                self.fields,
                self.P,
                self.c_val,
                self.bypass_mode,
                self.bypass_center,
                self.bypass_r2,
                self.bypass_nfw_centers,
                self.bypass_nfw_r_s,
                self.bypass_nfw_rho_s,
            )
            return traj, final, lam, steps

        return super().integrate_rk4(state0, dt, n_steps, lambda_stop=lambda_stop, record_every=record_every)

    def integrate_dopri5(self, state0, dt_init, lambda_stop,
                         rtol=1e-6, atol=1e-9,
                         dt_min=0.0, dt_max=0.0,
                         max_steps=10_000_000, record_every=0):
        state0 = np.ascontiguousarray(state0, dtype=np.float64)
        if state0.shape[0] != 24 or not self.lensing:
            raise ValueError("integrate_dopri5 is only implemented for the 24-state lensing path.")
        if self.sachs_screen_mode != _SCREEN_MODE_CONFORMAL:
            raise ValueError(
                "integrator_numba_specialized requires sachs_screen_convention "
                "= 'conformal_metric'. Use the lowalloc backend for other modes."
            )
        traj, final, _n_rec, lam, n_accept, n_reject = _integrate_dopri5_24_specialized(
            state0,
            float(dt_init),
            float(lambda_stop),
            float(rtol),
            float(atol),
            float(dt_min),
            float(dt_max),
            int(max_steps),
            int(record_every),
            self.origins,
            self.uppers,
            self.spacings,
            self.shapes,
            self.fields,
            self.P,
            self.c_val,
            self.bypass_mode,
            self.bypass_center,
            self.bypass_r2,
            self.bypass_nfw_centers,
            self.bypass_nfw_r_s,
            self.bypass_nfw_rho_s,
        )
        return traj, final, lam, n_accept, n_reject

    def integrate_dopri5_batch(self, states0, dt_init, lambda_stop,
                               rtol=1e-6, atol=1e-9,
                               dt_min=0.0, dt_max=0.0,
                               max_steps=10_000_000, record_every=0,
                               traj_capacity=2):
        states0 = np.ascontiguousarray(states0, dtype=np.float64)
        if states0.ndim != 2 or states0.shape[1] != 24 or not self.lensing:
            raise ValueError("integrate_dopri5_batch expects (n_photons, 24) for the lensing path.")
        if self.sachs_screen_mode != _SCREEN_MODE_CONFORMAL:
            raise ValueError(
                "integrator_numba_specialized requires sachs_screen_convention "
                "= 'conformal_metric'. Use the lowalloc backend for other modes."
            )
        n_photons = states0.shape[0]
        traj_capacity = max(int(traj_capacity), 2)
        traj_buf = np.empty((n_photons, traj_capacity, 24), dtype=np.float64)
        finals, lams, n_accepts, n_rejects, n_pts = _integrate_batch_dopri5_24_specialized(
            states0,
            float(dt_init),
            float(lambda_stop),
            float(rtol),
            float(atol),
            float(dt_min),
            float(dt_max),
            int(max_steps),
            int(record_every),
            traj_buf,
            self.origins,
            self.uppers,
            self.spacings,
            self.shapes,
            self.fields,
            self.P,
            self.c_val,
            self.bypass_mode,
            self.bypass_center,
            self.bypass_r2,
            self.bypass_nfw_centers,
            self.bypass_nfw_r_s,
            self.bypass_nfw_rho_s,
        )
        return finals, lams, n_accepts, n_rejects, traj_buf, n_pts

    def warmup(self):
        s8 = np.zeros(8)
        s8[0] = 0.5 * (self.eta_min + self.eta_max)
        s8[1] = float(self.origins[-1, 0] + 0.5 * (self.uppers[-1, 0] - self.origins[-1, 0]))
        s8[2] = float(self.origins[-1, 1] + 0.5 * (self.uppers[-1, 1] - self.origins[-1, 1]))
        s8[3] = float(self.origins[-1, 2] + 0.5 * (self.uppers[-1, 2] - self.origins[-1, 2]))
        s8[4] = 1.0
        s8[5] = 1.0
        _integrate_loop_8(
            s8,
            1e10, 2, 0.0, 0,
            self.origins, self.uppers, self.spacings, self.shapes, self.fields, self.P,
            self.eta_min, self.inv_deta,
            self.a_tab, self.adot_tab,
            self.c_val, self.slow_roll, self.sachs_screen_mode,
            self.bypass_mode, self.bypass_center, self.bypass_r2, self.bypass_nfw_centers,
            self.bypass_nfw_r_s, self.bypass_nfw_rho_s,
        )

        if self.lensing and self.sachs_screen_mode == _SCREEN_MODE_CONFORMAL:
            s24 = np.zeros(24)
            s24[0:8] = s8
            s24[8] = 0.0
            s24[9] = 1.0
            s24[12] = 0.0
            s24[14] = 1.0
            s24[20] = 1.0
            s24[23] = 1.0
            _integrate_loop_24_specialized(
                s24,
                1e10, 2, 0.0, 0,
                self.origins, self.uppers, self.spacings, self.shapes, self.fields, self.P,
                self.c_val,
                self.bypass_mode, self.bypass_center, self.bypass_r2, self.bypass_nfw_centers,
                self.bypass_nfw_r_s, self.bypass_nfw_rho_s,
            )
            _integrate_dopri5_24_specialized(
                s24,
                1e10, 0.0, 1e-6, 1e-9, 0.0, 0.0, 2, 0,
                self.origins, self.uppers, self.spacings, self.shapes, self.fields, self.P,
                self.c_val,
                self.bypass_mode, self.bypass_center, self.bypass_r2, self.bypass_nfw_centers,
                self.bypass_nfw_r_s, self.bypass_nfw_rho_s,
            )
            batch_states0 = np.tile(s24, (2, 1))
            batch_traj = np.empty((2, 2, 24))
            _integrate_batch_dopri5_24_specialized(
                batch_states0,
                1e10, 0.0, 1e-6, 1e-9, 0.0, 0.0, 2, 0,
                batch_traj,
                self.origins, self.uppers, self.spacings, self.shapes, self.fields, self.P,
                self.c_val,
                self.bypass_mode, self.bypass_center, self.bypass_r2, self.bypass_nfw_centers,
                self.bypass_nfw_r_s, self.bypass_nfw_rho_s,
            )
