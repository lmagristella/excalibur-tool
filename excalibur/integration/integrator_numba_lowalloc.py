from __future__ import annotations

import numpy as np
from numba import njit

from excalibur.integration.integrator_numba import (
    NumbaAMRBackend as _BaseNumbaAMRBackend,
    _SCREEN_MODE_METRIC,
    _a_adot_H_Hp,
    _init_sachs_basis_numba,
    _integrate_loop_8,
    _value_gradient_hessian_with_optional_bypass,
    integrate_photon_numba,
)
from excalibur.metrics.perturbed_flrw_metric_fast import compute_tensorial_acceleration
from excalibur.observables.optical_tidal_matrix import _set_riemann_sym


@njit(cache=True, fastmath=True)
def _select3(index, value0, value1, value2):
    if index == 0:
        return value0
    if index == 1:
        return value1
    return value2


@njit(cache=True, fastmath=True)
def _christoffel_flrw_inplace(G, a, adot, phi_SI, gx, gy, gz, phi_dot_SI, c_val):
    c_inv = 1.0 / c_val
    c_inv2 = c_inv * c_inv
    c_inv4 = c_inv2 * c_inv2
    a_inv = 1.0 / a
    a_inv2 = a_inv * a_inv
    adot_over_a = adot * a_inv
    phi_plus_psi = 2.0 * phi_SI
    a_adot_c_inv2 = a * adot * c_inv2
    a2_phi_dot_c_inv4 = a * a * phi_dot_SI * c_inv4

    for mu in range(4):
        for nu in range(4):
            for sigma in range(4):
                G[mu, nu, sigma] = 0.0

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
    G[0, 0, 1] = gxc
    G[0, 1, 0] = gxc
    G[0, 0, 2] = gyc
    G[0, 2, 0] = gyc
    G[0, 0, 3] = gzc
    G[0, 3, 0] = gzc

    tmix = adot_over_a - phi_dot_SI * c_inv2
    G[1, 1, 0] = tmix
    G[1, 0, 1] = tmix
    G[2, 2, 0] = tmix
    G[2, 0, 2] = tmix
    G[3, 3, 0] = tmix
    G[3, 0, 3] = tmix

    G[1, 2, 2] = gxc
    G[1, 3, 3] = gxc
    G[2, 1, 1] = gyc
    G[2, 3, 3] = gyc
    G[3, 1, 1] = gzc
    G[3, 2, 2] = gzc
    G[1, 1, 2] = -gyc
    G[1, 2, 1] = -gyc
    G[1, 1, 3] = -gzc
    G[1, 3, 1] = -gzc
    G[2, 2, 1] = -gxc
    G[2, 1, 2] = -gxc
    G[2, 2, 3] = -gzc
    G[2, 3, 2] = -gzc
    G[3, 3, 1] = -gxc
    G[3, 1, 3] = -gxc
    G[3, 3, 2] = -gyc
    G[3, 2, 3] = -gyc


@njit(cache=True, fastmath=True)
def _build_metric_tensor_flrw_inplace(g_mu_nu, scale_factor, phi_n, c_val):
    for mu in range(4):
        for nu in range(4):
            g_mu_nu[mu, nu] = 0.0

    scale2 = scale_factor * scale_factor
    c2 = c_val * c_val
    diag = scale2 * (1.0 - 2.0 * phi_n)
    g_mu_nu[0, 0] = -scale2 * (1.0 + 2.0 * phi_n) * c2
    g_mu_nu[1, 1] = diag
    g_mu_nu[2, 2] = diag
    g_mu_nu[3, 3] = diag


@njit(cache=True, fastmath=True)
def _sachs_transport_rhs_inplace(out, e_mu, christoffel, k_mu):
    for mu in range(4):
        s = 0.0
        for nu in range(4):
            for sigma in range(4):
                s += christoffel[mu, nu, sigma] * k_mu[sigma] * e_mu[nu]
        out[mu] = -s


@njit(cache=True, fastmath=True)
def _screen_projected_sachs_transport_rhs_inplace(out, e_mu, christoffel, k_mu, g_mu_nu, nabla_k_u):
    _sachs_transport_rhs_inplace(out, e_mu, christoffel, k_mu)

    for mu in range(4):
        s = 0.0
        for sigma in range(4):
            s += christoffel[mu, 0, sigma] * k_mu[sigma]
        nabla_k_u[mu] = s

    u_dot_k = 0.0
    for nu in range(4):
        u_dot_k += g_mu_nu[0, nu] * k_mu[nu]

    if abs(u_dot_k) < 1e-30:
        return

    e_dot_nabla_u = 0.0
    for mu in range(4):
        e_cov_mu = 0.0
        for nu in range(4):
            e_cov_mu += g_mu_nu[mu, nu] * e_mu[nu]
        e_dot_nabla_u += e_cov_mu * nabla_k_u[mu]

    alpha = -e_dot_nabla_u / u_dot_k
    for mu in range(4):
        out[mu] += alpha * k_mu[mu]


@njit(cache=True, fastmath=True)
def _riemann_blocks_kernel_inplace(
    Rd_k00l,
    Rd_0lki,
    Rd_kijl,
    a,
    H,
    Hprime,
    phi,
    gx,
    gy,
    gz,
    hess_phi,
    c_val,
):
    c2 = c_val * c_val
    a2 = a * a

    diag_scalar = Hprime * (1.0 - 2.0 * phi / c2)
    for k in range(3):
        for l in range(3):
            val = -hess_phi[k, l]
            if k == l:
                val += diag_scalar
            Rd_k00l[k, l] = a2 * val

    combo0 = H * gx
    combo1 = H * gy
    combo2 = H * gz
    fac_0 = a2 / c2
    for l in range(3):
        for k in range(3):
            for i in range(3):
                val = 0.0
                if i == l:
                    val += _select3(k, combo0, combo1, combo2)
                if l == k:
                    val -= _select3(i, combo0, combo1, combo2)
                Rd_0lki[l, k, i] = fac_0 * val

    second_scalar = Hprime - 6.0 * H * H * phi / c2
    fac_k = a2 / c2
    for k_idx in range(3):
        for i_idx in range(3):
            for j_idx in range(3):
                for l_idx in range(3):
                    val = 0.0
                    if k_idx == j_idx:
                        val += hess_phi[i_idx, l_idx]
                    if k_idx == l_idx:
                        val -= hess_phi[i_idx, j_idx]
                    if i_idx == j_idx:
                        val -= hess_phi[k_idx, l_idx]
                    if i_idx == l_idx:
                        val += hess_phi[k_idx, j_idx]

                    kron = 0.0
                    if l_idx == i_idx and k_idx == j_idx:
                        kron += 1.0
                    if l_idx == k_idx and i_idx == j_idx:
                        kron -= 1.0
                    val += second_scalar * kron
                    Rd_kijl[k_idx, i_idx, j_idx, l_idx] = fac_k * val


@njit(cache=True, fastmath=True)
def _optical_tidal_matrix_optimized_inplace(R_AB, R_down, T_down, Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu):
    for a in range(4):
        for b in range(4):
            T_down[a, b] = 0.0
            for c in range(4):
                for d in range(4):
                    R_down[a, b, c, d] = 0.0

    for k in range(3):
        for l in range(3):
            _set_riemann_sym(R_down, k + 1, 0, 0, l + 1, Rd_k00l[k, l])

    for l in range(3):
        for k in range(3):
            for i in range(3):
                _set_riemann_sym(R_down, 0, l + 1, k + 1, i + 1, Rd_0lki[l, k, i])

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
        e1_mu_val = e1_mu[mu]
        e2_mu_val = e2_mu[mu]
        for nu in range(4):
            t_val = T_down[mu, nu]
            s11 += t_val * e1_mu_val * e1_mu[nu]
            s12 += t_val * e1_mu_val * e2_mu[nu]
            s21 += t_val * e2_mu_val * e1_mu[nu]
            s22 += t_val * e2_mu_val * e2_mu[nu]

    R_AB[0, 0] = s11
    R_AB[0, 1] = s12
    R_AB[1, 0] = s21
    R_AB[1, 1] = s22


@njit(cache=True, fastmath=True)
def _jacobi_rhs_inplace(out, D_flat, R_AB):
    D11 = D_flat[0]
    D12 = D_flat[1]
    D21 = D_flat[2]
    D22 = D_flat[3]
    P11 = D_flat[4]
    P12 = D_flat[5]
    P21 = D_flat[6]
    P22 = D_flat[7]

    out[0] = P11
    out[1] = P12
    out[2] = P21
    out[3] = P22
    out[4] = -(R_AB[0, 0] * D11 + R_AB[0, 1] * D21)
    out[5] = -(R_AB[0, 0] * D12 + R_AB[0, 1] * D22)
    out[6] = -(R_AB[1, 0] * D11 + R_AB[1, 1] * D21)
    out[7] = -(R_AB[1, 0] * D12 + R_AB[1, 1] * D22)


@njit(cache=True, fastmath=True)
def _geodesic_rhs_24_lowalloc(
    out,
    state,
    origins,
    uppers,
    spacings,
    shapes,
    fields,
    P,
    eta_min,
    inv_deta,
    a_tab,
    adot_tab,
    H_tab,
    Hp_tab,
    c_val,
    slow_roll,
    screen_mode,
    bypass_mode,
    bypass_center,
    bypass_r2,
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
    Rd_0lki,
    Rd_kijl,
    R_down,
    T_down,
    R_AB,
    jrhs,
):
    eta = state[0]
    x = state[1]
    y = state[2]
    z = state[3]
    u0 = state[4]
    u1 = state[5]
    u2 = state[6]
    u3 = state[7]

    a, adot, H_conf, H_prime = _a_adot_H_Hp(
        eta, eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
    )

    val, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz = _value_gradient_hessian_with_optional_bypass(
        x, y, z, origins, uppers, spacings, shapes, fields, P,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_r_s, bypass_nfw_rho_s,
    )

    c2 = c_val * c_val
    phi_n = val / c2

    if screen_mode == _SCREEN_MODE_METRIC:
        geo_a = a
        geo_adot = adot
        optic_a = a
        optic_H_conf = H_conf
        optic_H_prime = H_prime
    else:
        geo_a = 1.0
        geo_adot = 0.0
        optic_a = 1.0
        optic_H_conf = 0.0
        optic_H_prime = 0.0

    du0, du1, du2, du3 = compute_tensorial_acceleration(
        u0, u1, u2, u3, geo_a, geo_adot, phi_n, gx, gy, gz, 0.0, c_val,
    )

    k_mu[0] = u0
    k_mu[1] = u1
    k_mu[2] = u2
    k_mu[3] = u3

    _christoffel_flrw_inplace(G, optic_a, geo_adot if screen_mode != _SCREEN_MODE_METRIC else adot, val, gx, gy, gz, 0.0, c_val)
    _build_metric_tensor_flrw_inplace(g_mu_nu, optic_a, phi_n, c_val)

    _screen_projected_sachs_transport_rhs_inplace(de1, state[8:12], G, k_mu, g_mu_nu, nabla_k_u)
    _screen_projected_sachs_transport_rhs_inplace(de2, state[12:16], G, k_mu, g_mu_nu, nabla_k_u)

    hess_phi[0, 0] = hxx
    hess_phi[1, 1] = hyy
    hess_phi[2, 2] = hzz
    hess_phi[0, 1] = hxy
    hess_phi[1, 0] = hxy
    hess_phi[0, 2] = hxz
    hess_phi[2, 0] = hxz
    hess_phi[1, 2] = hyz
    hess_phi[2, 1] = hyz

    _riemann_blocks_kernel_inplace(
        Rd_k00l,
        Rd_0lki,
        Rd_kijl,
        optic_a,
        optic_H_conf,
        optic_H_prime,
        val,
        gx,
        gy,
        gz,
        hess_phi,
        c_val,
    )

    if screen_mode == _SCREEN_MODE_METRIC:
        _optical_tidal_matrix_optimized_inplace(
            R_AB, R_down, T_down,
            Rd_k00l, Rd_0lki, Rd_kijl,
            k_mu, state[8:12], state[12:16],
        )
    else:
        e1_local, e2_local = _init_sachs_basis_numba(k_mu, g_mu_nu, optic_a, screen_mode)
        _optical_tidal_matrix_optimized_inplace(
            R_AB, R_down, T_down,
            Rd_k00l, Rd_0lki, Rd_kijl,
            k_mu, e1_local, e2_local,
        )

    _jacobi_rhs_inplace(jrhs, state[16:24], R_AB)

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


@njit(cache=True, fastmath=True)
def _rk4_step_24_lowalloc(
    out,
    state,
    dt,
    origins,
    uppers,
    spacings,
    shapes,
    fields,
    P,
    eta_min,
    inv_deta,
    a_tab,
    adot_tab,
    H_tab,
    Hp_tab,
    c_val,
    slow_roll,
    screen_mode,
    bypass_mode,
    bypass_center,
    bypass_r2,
    bypass_nfw_r_s,
    bypass_nfw_rho_s,
    k1,
    k2,
    k3,
    k4,
    s2,
    s3,
    s4,
    k_mu,
    G,
    g_mu_nu,
    de1,
    de2,
    nabla_k_u,
    hess_phi,
    Rd_k00l,
    Rd_0lki,
    Rd_kijl,
    R_down,
    T_down,
    R_AB,
    jrhs,
):
    half_dt = 0.5 * dt
    sixth_dt = dt / 6.0

    _geodesic_rhs_24_lowalloc(
        k1, state,
        origins, uppers, spacings, shapes, fields, P,
        eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
        c_val, slow_roll, screen_mode,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_0lki, Rd_kijl, R_down, T_down, R_AB, jrhs,
    )
    for i in range(24):
        s2[i] = state[i] + half_dt * k1[i]

    _geodesic_rhs_24_lowalloc(
        k2, s2,
        origins, uppers, spacings, shapes, fields, P,
        eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
        c_val, slow_roll, screen_mode,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_0lki, Rd_kijl, R_down, T_down, R_AB, jrhs,
    )
    for i in range(24):
        s3[i] = state[i] + half_dt * k2[i]

    _geodesic_rhs_24_lowalloc(
        k3, s3,
        origins, uppers, spacings, shapes, fields, P,
        eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
        c_val, slow_roll, screen_mode,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_0lki, Rd_kijl, R_down, T_down, R_AB, jrhs,
    )
    for i in range(24):
        s4[i] = state[i] + dt * k3[i]

    _geodesic_rhs_24_lowalloc(
        k4, s4,
        origins, uppers, spacings, shapes, fields, P,
        eta_min, inv_deta, a_tab, adot_tab, H_tab, Hp_tab,
        c_val, slow_roll, screen_mode,
        bypass_mode, bypass_center, bypass_r2, bypass_nfw_r_s, bypass_nfw_rho_s,
        k_mu, G, g_mu_nu, de1, de2, nabla_k_u, hess_phi,
        Rd_k00l, Rd_0lki, Rd_kijl, R_down, T_down, R_AB, jrhs,
    )

    for i in range(24):
        out[i] = state[i] + sixth_dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


@njit(cache=True, fastmath=True)
def _integrate_loop_24_lowalloc(
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
    eta_min,
    inv_deta,
    a_tab,
    adot_tab,
    H_tab,
    Hp_tab,
    c_val,
    slow_roll,
    screen_mode,
    bypass_mode,
    bypass_center,
    bypass_r2,
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
    Rd_0lki = np.empty((3, 3, 3))
    Rd_kijl = np.empty((3, 3, 3, 3))
    R_down = np.empty((4, 4, 4, 4))
    T_down = np.empty((4, 4))
    R_AB = np.empty((2, 2))
    jrhs = np.empty(8)

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

        _rk4_step_24_lowalloc(
            state_next,
            state,
            dt_eff,
            origins,
            uppers,
            spacings,
            shapes,
            fields,
            P,
            eta_min,
            inv_deta,
            a_tab,
            adot_tab,
            H_tab,
            Hp_tab,
            c_val,
            slow_roll,
            screen_mode,
            bypass_mode,
            bypass_center,
            bypass_r2,
            bypass_nfw_r_s,
            bypass_nfw_rho_s,
            k1,
            k2,
            k3,
            k4,
            s2,
            s3,
            s4,
            k_mu,
            G,
            g_mu_nu,
            de1,
            de2,
            nabla_k_u,
            hess_phi,
            Rd_k00l,
            Rd_0lki,
            Rd_kijl,
            R_down,
            T_down,
            R_AB,
            jrhs,
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


class NumbaAMRBackend(_BaseNumbaAMRBackend):
    """Alternative numba backend that reuses scratch arrays in the 24-state RK4 path."""

    def integrate_rk4(self, state0, dt, n_steps, lambda_stop=0.0, record_every=0):
        state0 = np.ascontiguousarray(state0, dtype=np.float64)

        if state0.shape[0] == 24 and self.lensing:
            traj, final, _n_rec, lam, steps = _integrate_loop_24_lowalloc(
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
                self.bypass_nfw_r_s,
                self.bypass_nfw_rho_s,
            )
            return traj, final, lam, steps

        return super().integrate_rk4(state0, dt, n_steps, lambda_stop=lambda_stop, record_every=record_every)

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
            1e10,
            2,
            0.0,
            0,
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
            self.bypass_nfw_r_s,
            self.bypass_nfw_rho_s,
        )

        if self.lensing:
            s24 = np.zeros(24)
            s24[0:8] = s8
            s24[8] = 0.0
            s24[9] = 1.0
            s24[12] = 0.0
            s24[14] = 1.0
            s24[20] = 1.0
            s24[23] = 1.0
            _integrate_loop_24_lowalloc(
                s24,
                1e10,
                2,
                0.0,
                0,
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
                self.bypass_nfw_r_s,
                self.bypass_nfw_rho_s,
            )