#!/usr/bin/env python3
"""Audit the optical solver on a representative analytical NFW ray.

This script does two things on the same b = 1 Mpc trajectory:

1. Pointwise audit of the optical tidal matrix:
   compare the production ``R_AB`` implementation to both a full 4D
   contraction and the equivalent spatial formula written in the stored
   Riemann-block conventions.

2. Frozen-operator Jacobi replay:
   after integrating the full 24-state system, rebuild ``R_AB`` at every
   recorded step and re-integrate only the Jacobi equation on that frozen
   trajectory/basis.  If this replay reproduces the final kappa, the bias is
   carried by the optical operator along the ray rather than by some hidden
   mismatch in the online solver plumbing.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c, one_Mpc
from excalibur.integration.integrator import Integrator
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.observables.optical_tidal_matrix import (
    distance_comparison,
    jacobi_rhs,
    lensing_from_jacobi,
    optical_tidal_matrix_from_blocks,
    optical_tidal_matrix_optimized,
)
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.observables.riemann_perturbed_flrw import riemann_blocks_kernel
from run_lensing_nfw_analytic import make_photon
from run_nfw_independent_benchmark import (
    build_setup,
    scalar_kappa_on_real_path,
    scalar_kappa_straight_ray,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit the NFW optical solver on an analytical bypass geometry.")
    parser.add_argument("--b-mpc", type=float, default=1.0)
    parser.add_argument("--z-lens", type=float, default=0.24652)
    parser.add_argument("--z-source", type=float, default=0.50203)
    parser.add_argument("--mass-msun", type=float, default=2e15)
    parser.add_argument("--c-nfw", type=float, default=7.0)
    parser.add_argument("--box-mpc", type=float, default=None)
    parser.add_argument("--obs-z-mpc", type=float, default=5.0)
    parser.add_argument("--n-root", type=int, default=16)
    parser.add_argument("--dt-per-rs", type=float, default=8.0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-b-mpc", type=float, nargs="*", default=[0.3, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--sweep-z-lens", type=float, nargs="*", default=[0.1, 0.2, 0.3])
    parser.add_argument("--sweep-z-source", type=float, nargs="*", default=[0.3, 0.5, 0.8])
    return parser.parse_args()


def _build_setup_from_args(args, *, z_lens=None, z_source=None):
    return build_setup(
        z_lens=args.z_lens if z_lens is None else z_lens,
        z_source=args.z_source if z_source is None else z_source,
        mass_msun=args.mass_msun,
        c_nfw=args.c_nfw,
        box_mpc=args.box_mpc,
        obs_z_mpc=args.obs_z_mpc,
        n_root=args.n_root,
    )


def spatial_formula_rab(Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu):
    """Spatial form of R_AB in the stored block conventions.

    The stored blocks are ``R_{k00l}``, ``R_{0lki}``, ``R_{kijl}``, whereas the
    4D contraction uses ``R_{k0l0}``, ``R_{k0li}``, ``R_{kil0}``, ``R_{kilj}``.
    Rewriting the latter in terms of the stored blocks introduces one
    antisymmetry per term, hence the overall minus sign below.
    """
    k0 = k_mu[0]
    ki = k_mu[1:4]
    s = np.vstack([e1_mu[1:4], e2_mu[1:4]])

    R_AB = np.zeros((2, 2))
    for A in range(2):
        for B in range(2):
            val = 0.0
            for k_idx in range(3):
                for l_idx in range(3):
                    val += s[A, k_idx] * s[B, l_idx] * Rd_k00l[k_idx, l_idx] * k0 * k0
                    for i_idx in range(3):
                        val += s[A, k_idx] * s[B, l_idx] * (
                            Rd_0lki[l_idx, k_idx, i_idx] - Rd_0lki[k_idx, i_idx, l_idx]
                        ) * k0 * ki[i_idx]
                    for i_idx in range(3):
                        for j_idx in range(3):
                            val += s[A, k_idx] * s[B, l_idx] * Rd_kijl[k_idx, i_idx, j_idx, l_idx] * ki[i_idx] * ki[j_idx]
            R_AB[A, B] = -val
    return R_AB


def build_metric_tensor(a, phi_si):
    phi_norm = phi_si / (c * c)
    g_mu_nu = np.zeros((4, 4))
    g_mu_nu[0, 0] = -a * a * (1.0 + 2.0 * phi_norm) * c * c
    g_mu_nu[1, 1] = a * a * (1.0 - 2.0 * phi_norm)
    g_mu_nu[2, 2] = a * a * (1.0 - 2.0 * phi_norm)
    g_mu_nu[3, 3] = a * a * (1.0 - 2.0 * phi_norm)
    return g_mu_nu


def normalize_spatial(vec):
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero spatial vector")
    return vec / norm


def make_fixed_euclidean_screen(direction, seed):
    direction_hat = normalize_spatial(direction)
    screen1 = seed - np.dot(seed, direction_hat) * direction_hat
    if np.linalg.norm(screen1) < 1e-15:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, direction_hat)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        screen1 = fallback - np.dot(fallback, direction_hat) * direction_hat
    screen1 = normalize_spatial(screen1)
    screen2 = normalize_spatial(np.cross(direction_hat, screen1))
    return screen1, screen2


def sample_local_fields(metric, state):
    x_mu = state[0:4]
    eta = x_mu[0]
    pos = x_mu[1:4]

    a, _ = metric._get_scale_factor_and_derivative(eta)
    H_conf = metric.cosmology.conformal_hubble(eta)
    H_prime = metric.cosmology.conformal_hubble_prime(eta)

    phi_si, grad3_tuple, hess3_tuple, phi_dot_si = metric.interp.value_gradient_hessian_and_time_derivative(
        pos, "Phi", eta
    )
    gx, gy, gz = grad3_tuple
    hxx, hyy, hzz, hxy, hxz, hyz = hess3_tuple

    if metric.slow_roll:
        grad_phi_dot = np.zeros(3)
        phi_ddot = 0.0
        phi_dot_si = 0.0
    else:
        dt_fd = max(1e-6 * abs(eta), 1.0)
        _, grad3_p, _, phi_dot_p = metric.interp.value_gradient_hessian_and_time_derivative(pos, "Phi", eta + dt_fd)
        _, grad3_m, _, phi_dot_m = metric.interp.value_gradient_hessian_and_time_derivative(pos, "Phi", eta - dt_fd)
        grad_phi_dot = np.array([
            (grad3_p[0] - grad3_m[0]) / (2.0 * dt_fd),
            (grad3_p[1] - grad3_m[1]) / (2.0 * dt_fd),
            (grad3_p[2] - grad3_m[2]) / (2.0 * dt_fd),
        ])
        phi_ddot = (phi_dot_p - phi_dot_m) / (2.0 * dt_fd)

    grad_phi = np.array([gx, gy, gz])
    hess_phi = np.array([
        [hxx, hxy, hxz],
        [hxy, hyy, hyz],
        [hxz, hyz, hzz],
    ])
    return {
        "a": a,
        "H_conf": H_conf,
        "H_prime": H_prime,
        "phi_si": phi_si,
        "phi_dot_si": phi_dot_si,
        "phi_ddot": phi_ddot,
        "grad_phi": grad_phi,
        "grad_phi_dot": grad_phi_dot,
        "hess_phi": hess_phi,
    }


def rebuild_riemann_blocks_from_fields(fields):
    return riemann_blocks_kernel(
        fields["a"],
        fields["H_conf"],
        fields["H_prime"],
        fields["phi_si"],
        fields["phi_dot_si"],
        fields["phi_ddot"],
        fields["grad_phi"],
        fields["grad_phi_dot"],
        fields["hess_phi"],
        c,
    )


def decompose_riemann_blocks(fields):
    a = fields["a"]
    H_conf = fields["H_conf"]
    H_prime = fields["H_prime"]
    phi_si = fields["phi_si"]
    phi_dot_si = fields["phi_dot_si"]
    phi_ddot = fields["phi_ddot"]
    grad_phi = fields["grad_phi"]
    grad_phi_dot = fields["grad_phi_dot"]
    hess_phi = fields["hess_phi"]

    c2 = c * c
    a2 = a * a

    block1_hess = -a2 * hess_phi
    diag_scalar = H_prime * (1.0 - 2.0 * phi_si / c2) + phi_ddot / c2 + 2.0 * H_conf * phi_dot_si / c2
    block1_scalar = a2 * diag_scalar * np.eye(3)

    combo = grad_phi_dot + H_conf * grad_phi
    block2 = np.zeros((3, 3, 3))
    fac = a2 / c2
    for l_idx in range(3):
        for k_idx in range(3):
            for i_idx in range(3):
                if i_idx == l_idx:
                    block2[l_idx, k_idx, i_idx] += fac * combo[k_idx]
                if l_idx == k_idx:
                    block2[l_idx, k_idx, i_idx] -= fac * combo[i_idx]

    term_a = np.zeros((3, 3, 3, 3))
    term_b = np.zeros((3, 3, 3, 3))
    term_c = np.zeros((3, 3, 3, 3))
    term_d = np.zeros((3, 3, 3, 3))
    kron_scalar = np.zeros((3, 3, 3, 3))

    second_scalar = H_prime - (2.0 * H_conf * phi_dot_si + 6.0 * H_conf * H_conf * phi_si) / c2
    for k_idx in range(3):
        for i_idx in range(3):
            for j_idx in range(3):
                for l_idx in range(3):
                    if k_idx == j_idx:
                        term_a[k_idx, i_idx, j_idx, l_idx] = fac * hess_phi[i_idx, l_idx]
                    if k_idx == l_idx:
                        term_b[k_idx, i_idx, j_idx, l_idx] = -fac * hess_phi[i_idx, j_idx]
                    if i_idx == j_idx:
                        term_c[k_idx, i_idx, j_idx, l_idx] = -fac * hess_phi[k_idx, l_idx]
                    if i_idx == l_idx:
                        term_d[k_idx, i_idx, j_idx, l_idx] = fac * hess_phi[k_idx, j_idx]
                    kron = 0.0
                    if l_idx == i_idx and k_idx == j_idx:
                        kron += 1.0
                    if l_idx == k_idx and i_idx == j_idx:
                        kron -= 1.0
                    kron_scalar[k_idx, i_idx, j_idx, l_idx] = fac * second_scalar * kron

    block3_hess = term_a + term_b + term_c + term_d
    return {
        "block1_hess": block1_hess,
        "block1_scalar": block1_scalar,
        "block2": block2,
        "block3_hess": block3_hess,
        "block3_scalar": kron_scalar,
        "block3_hess_A": term_a,
        "block3_hess_B": term_b,
        "block3_hess_C": term_c,
        "block3_hess_D": term_d,
    }


def direct_hessian_optical_terms(a, hess_phi, k_mu, e1_mu, e2_mu):
    r"""Direct Hessian-only optical operator on the current ray/basis.

    This is the static weak-field Hessian sector written directly in terms of
    the local spatial Hessian, the spatial part of k^mu, and the Sachs screen
    basis.  It is independent from ``riemann_blocks_kernel`` and from the
    4-index Riemann assembly used elsewhere in this script.

    The split below is not the same combinatorial A/B/C/D split used for the
    stored ``R_{kijl}`` block.  Here we organize the Hessian contribution in a
    more geometric basis: a lapse-like screen Hessian term, a projected-screen
    term, a longitudinal-trace term, and two screen-longitudinal mix terms.
    """
    q = k_mu[1:4]
    k0 = k_mu[0]
    S = np.vstack([e1_mu[1:4], e2_mu[1:4]])
    a2 = a * a
    qq = np.dot(q, q)
    qHq = q @ hess_phi @ q

    direct_lapse = np.zeros((2, 2))
    direct_mix_left = np.zeros((2, 2))
    direct_los_trace = np.zeros((2, 2))
    direct_screen = np.zeros((2, 2))
    direct_mix_right = np.zeros((2, 2))

    for A in range(2):
        sA = S[A]
        sAq = np.dot(sA, q)
        sAHq = sA @ hess_phi @ q
        for B in range(2):
            sB = S[B]
            sBq = np.dot(sB, q)
            qHsB = q @ hess_phi @ sB
            sAHsB = sA @ hess_phi @ sB
            sAsB = np.dot(sA, sB)

            direct_lapse[A, B] = a2 * (k0 * k0) * sAHsB
            direct_mix_left[A, B] = -(a2 / (c * c)) * sAq * qHsB
            direct_los_trace[A, B] = +(a2 / (c * c)) * sAsB * qHq
            direct_screen[A, B] = +(a2 / (c * c)) * qq * sAHsB
            direct_mix_right[A, B] = -(a2 / (c * c)) * sBq * sAHq

    direct_total = (
        direct_lapse
        + direct_mix_left
        + direct_los_trace
        + direct_screen
        + direct_mix_right
    )
    return {
        "direct_lapse": direct_lapse,
        "direct_mix_left": direct_mix_left,
        "direct_los_trace": direct_los_trace,
        "direct_screen": direct_screen,
        "direct_mix_right": direct_mix_right,
        "direct_total": direct_total,
    }


def rebuild_riemann_blocks(metric, state):
    fields = sample_local_fields(metric, state)
    return fields["a"], fields["phi_si"], rebuild_riemann_blocks_from_fields(fields)


def integrate_full_state_history(metric, state0, dt, lambda_total, *, rtol=1e-8, atol=1e-13):
    """Manual fixed-step replay of the production 24-state RK4 solver."""
    solver = Integrator(metric=metric, dt=dt, mode="sequential", integrator="rk4", rtol=rtol, atol=atol)
    state = state0.copy()
    history = [state.copy()]
    dt_history = []
    lambda_current = 0.0

    while abs(lambda_current) < abs(lambda_total):
        remaining = abs(lambda_total) - abs(lambda_current)
        dt_eff = min(abs(dt), remaining)
        dt_eff = dt_eff if dt >= 0 else -dt_eff
        state, _, accepted = solver.integrator.step(metric, state, dt_eff, solver.rtol, solver.atol)
        if not accepted:
            raise RuntimeError("Unexpected rejected step in fixed-step RK4 replay")
        lambda_current += solver.integrator.dt_actual
        dt_history.append(solver.integrator.dt_actual)
        history.append(state.copy())

    return np.asarray(history), np.asarray(dt_history), lambda_current


def replay_jacobi_on_frozen_operator(R_series, dt_history):
    """Re-integrate only the Jacobi equation on a frozen recorded operator."""
    DP = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    for idx, dt in enumerate(dt_history):
        R0 = R_series[idx]
        R1 = R_series[idx + 1]

        def rhs(dp_state, frac):
            return jacobi_rhs(dp_state, (1.0 - frac) * R0 + frac * R1)

        k1 = rhs(DP, 0.0)
        k2 = rhs(DP + 0.5 * dt * k1, 0.5)
        k3 = rhs(DP + 0.5 * dt * k2, 0.5)
        k4 = rhs(DP + dt * k3, 1.0)
        DP = DP + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return DP


def replay_jacobi_first_order(R_series, dt_history, lambda_final):
    r"""Independent first-order Sachs solution on the recorded operator.

    Linearizing the Jacobi equation around the unlensed solution

        D_0(lambda) = lambda I,

    gives

        D(lambda_S) = lambda_S I - \int_0^{lambda_S} (lambda_S - s) s R(s) ds

    up to O(R^2).  This is independent from the online 24-state solver and from
    the frozen RK4 Jacobi replay above.
    """
    lam = np.concatenate(([0.0], np.cumsum(dt_history)))
    weights = (lambda_final - lam) * lam / lambda_final
    correction = np.trapezoid(weights[:, None, None] * R_series, x=lam, axis=0)

    A = np.eye(2) - correction
    return np.array([A[0, 0], A[0, 1], A[1, 0], A[1, 1]])


def replay_alpha(R_series, dt_history, lambda_final, kappa_analytic):
    DP = replay_jacobi_on_frozen_operator(R_series, dt_history)
    kappa, _, _ = lensing_from_jacobi(DP[:4] / lambda_final)
    alpha = kappa / kappa_analytic if abs(kappa_analytic) > 0.0 else np.nan
    return kappa, alpha


def compute_residual_metrics(args, *, b_mpc, z_lens=None, z_source=None):
    setup = _build_setup_from_args(args, z_lens=z_lens, z_source=z_source)
    metric = setup["metric"]
    halo = setup["halo"]
    target = setup["center"] + b_mpc * one_Mpc * setup["e_perp1"]
    dt = halo.r_s / (args.dt_per_rs * c)

    photon = make_photon(setup["obs_pos"], target, metric, setup["eta_0"], setup["a_0"])
    solver = Integrator(metric=metric, dt=dt, mode="sequential", integrator="rk4", rtol=1e-8, atol=1e-13)
    solver.integrate_single(photon, stop_mode="affine", stop_value=setup["lambda_total"])

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        kappa_num, _, gamma_num = lensing_from_jacobi(photon.D_flat / photon.lambda_affine)
    kappa_analytic = float(halo.kappa_analytic(np.array([b_mpc * one_Mpc]), setup["sigma_cr_comoving"])[0])
    gamma_analytic = float(halo.gamma_analytic(np.array([b_mpc * one_Mpc]), setup["sigma_cr_comoving"])[0])
    kappa_born = scalar_kappa_straight_ray(
        halo,
        setup["obs_pos"],
        target,
        setup["d_s"],
        setup["sigma_cr_comoving"],
    )
    kappa_path = scalar_kappa_on_real_path(halo, photon, setup["sigma_cr_comoving"])
    distance_diag = distance_comparison(photon.D_flat * c, setup["z_s"], setup["cosmo"])
    da_ratio = distance_diag["D_A_ray"] / distance_diag["D_A_FLRW"]

    alpha_num_raw = kappa_num / kappa_analytic if abs(kappa_analytic) > 0.0 else np.nan
    gamma_ratio_raw = gamma_num / gamma_analytic if abs(gamma_analytic) > 0.0 else np.nan
    stable = (
        np.isfinite(kappa_num)
        and np.isfinite(gamma_num)
        and np.isfinite(alpha_num_raw)
        and np.isfinite(gamma_ratio_raw)
        and np.isfinite(distance_diag["D_A_ray"])
        and np.isfinite(distance_diag["D_A_FLRW"])
        and np.isfinite(da_ratio)
        and abs(kappa_num) < 10.0
        and abs(gamma_num) < 10.0
        and abs(da_ratio) < 10.0
    )

    return {
        "z_l": setup["z_l"],
        "z_s": setup["z_s"],
        "b_mpc": b_mpc,
        "d_l_mpc": setup["d_l"] / one_Mpc,
        "d_s_mpc": setup["d_s"] / one_Mpc,
        "kappa_num": kappa_num,
        "gamma_num": gamma_num,
        "kappa_analytic": kappa_analytic,
        "gamma_analytic": gamma_analytic,
        "kappa_born": kappa_born,
        "kappa_path": kappa_path,
        "alpha_num_raw": alpha_num_raw,
        "alpha_num": alpha_num_raw if stable else np.nan,
        "alpha_born": kappa_born / kappa_analytic if abs(kappa_analytic) > 0.0 else np.nan,
        "alpha_path": kappa_path / kappa_analytic if abs(kappa_analytic) > 0.0 else np.nan,
        "gamma_ratio_raw": gamma_ratio_raw,
        "gamma_ratio": gamma_ratio_raw if stable else np.nan,
        "da_ratio": da_ratio,
        "status": "ok" if stable else "unstable",
    }


def run_sweep(args):
    rows = []
    for z_lens in args.sweep_z_lens:
        for z_source in args.sweep_z_source:
            if z_source <= z_lens:
                continue
            for b_mpc in args.sweep_b_mpc:
                rows.append(compute_residual_metrics(args, b_mpc=b_mpc, z_lens=z_lens, z_source=z_source))

    print("=" * 96)
    print("  NFW Optical Solver Sweep  --  explicit-redshift conformal/comoving residual map")
    print("=" * 96)
    print("  z_l      z_s      b[Mpc]    alpha_num    alpha_path   alpha_born   gamma_ratio   status")
    for row in rows:
        alpha_num_txt = f"{row['alpha_num']:0.6f}" if np.isfinite(row["alpha_num"]) else "nan"
        gamma_ratio_txt = f"{row['gamma_ratio']:0.6f}" if np.isfinite(row["gamma_ratio"]) else "nan"
        print(
            f"  {row['z_l']:0.3f}    {row['z_s']:0.3f}    {row['b_mpc']:6.3f}    "
            f"{alpha_num_txt:>10}    {row['alpha_path']:0.6f}    {row['alpha_born']:0.6f}    {gamma_ratio_txt:>11}   {row['status']}"
        )

    stable_rows = [row for row in rows if row["status"] == "ok"]
    unstable_rows = [row for row in rows if row["status"] != "ok"]
    alpha_vals = np.array([row["alpha_num"] for row in stable_rows], dtype=float)
    gamma_vals = np.array([row["gamma_ratio"] for row in stable_rows], dtype=float)
    print()
    print(f"  Stable points = {len(stable_rows)} / {len(rows)}")
    print(f"  alpha_num mean/std = {np.nanmean(alpha_vals):.6f} +/- {np.nanstd(alpha_vals):.6f}")
    print(f"  gamma_ratio mean/std = {np.nanmean(gamma_vals):.6f} +/- {np.nanstd(gamma_vals):.6f}")

    z_pairs = {}
    for row in stable_rows:
        z_pairs.setdefault((row["z_l"], row["z_s"]), []).append(row["alpha_num"])
    print("  Pair averages:")
    for (z_lens, z_source), values in sorted(z_pairs.items()):
        values = np.asarray(values, dtype=float)
        print(f"    z_l={z_lens:.3f}, z_s={z_source:.3f}: alpha_num = {np.nanmean(values):.6f} +/- {np.nanstd(values):.6f}")

    b_groups = {}
    for row in stable_rows:
        b_groups.setdefault(row["b_mpc"], []).append(row["alpha_num"])
    print("  Impact averages:")
    for b_mpc, values in sorted(b_groups.items()):
        values = np.asarray(values, dtype=float)
        print(f"    b={b_mpc:.3f} Mpc: alpha_num = {np.nanmean(values):.6f} +/- {np.nanstd(values):.6f}")
    if unstable_rows:
        print("  Unstable points:")
        for row in unstable_rows:
            print(
                f"    z_l={row['z_l']:.3f}, z_s={row['z_s']:.3f}, b={row['b_mpc']:.3f} Mpc: "
                f"alpha_path={row['alpha_path']:.6f}, alpha_born={row['alpha_born']:.6f}, D_A/D_A_FLRW={row['da_ratio']:.6g}"
            )
    print("=" * 96)


def main():
    args = parse_args()
    if args.sweep:
        run_sweep(args)
        return

    b_mpc = args.b_mpc
    setup = _build_setup_from_args(args)
    metric = setup["metric"]
    halo = setup["halo"]
    target = setup["center"] + b_mpc * one_Mpc * setup["e_perp1"]
    dt = halo.r_s / (args.dt_per_rs * c)

    print("=" * 88)
    print("  NFW Optical Solver Audit  --  real ray, pointwise R_AB + frozen Jacobi replay")
    print("=" * 88)
    print(
        f"  Geometry: b = {b_mpc:.2f} Mpc, z_l = {setup['z_l']:.4f}, z_s = {setup['z_s']:.4f}, "
        f"D_l = {setup['d_l']/one_Mpc:.1f} Mpc"
    )

    # Production solver result (for reference)
    photon_solver = make_photon(setup["obs_pos"], target, metric, setup["eta_0"], setup["a_0"])
    prod_solver = Integrator(metric=metric, dt=dt, mode="sequential", integrator="rk4", rtol=1e-8, atol=1e-13)
    prod_solver.integrate_single(photon_solver, stop_mode="affine", stop_value=setup["lambda_total"])
    kappa_solver, _, _ = lensing_from_jacobi(photon_solver.D_flat / photon_solver.lambda_affine)
    kappa_target_norm, mu_target_norm, gamma_target_norm = lensing_from_jacobi(
        photon_solver.D_flat / setup["lambda_total"]
    )
    distance_diag = distance_comparison(photon_solver.D_flat * c, setup["z_s"], setup["cosmo"])
    da_deficit = 1.0 - distance_diag["D_A_ray"] / distance_diag["D_A_FLRW"]
    det_target = np.linalg.det((photon_solver.D_flat / setup["lambda_total"]).reshape(2, 2))
    da_deficit_from_det = 1.0 - np.sqrt(abs(det_target))

    # Full-state manual replay to record x, k, e1, e2, D, P at every step.
    photon_replay = make_photon(setup["obs_pos"], target, metric, setup["eta_0"], setup["a_0"])
    state0 = np.concatenate([
        photon_replay.x,
        photon_replay.u,
        photon_replay.e1,
        photon_replay.e2,
        photon_replay.D_flat,
        photon_replay.P_flat,
    ])
    history, dt_history, lambda_final = integrate_full_state_history(metric, state0, dt, setup["lambda_total"])
    kappa_manual, _, _ = lensing_from_jacobi(history[-1, 16:20] / lambda_final)

    n_states = history.shape[0]
    print(f"  Recorded states: {n_states}  (steps = {n_states - 1}, lambda_final = {lambda_final:.6e} s)")
    print(f"  kappa_solver = {kappa_solver:+.8e}")
    print(f"  kappa_manual = {kappa_manual:+.8e}   rel.diff = {abs(kappa_manual/kappa_solver - 1.0):.3e}")

    kappa_analytic = float(halo.kappa_analytic(np.array([b_mpc * one_Mpc]), setup["sigma_cr_physical"])[0])
    print(f"  kappa_analytic = {kappa_analytic:+.8e}   alpha_full = {kappa_solver / kappa_analytic:.6f}")
    print()
    print("  Jacobi normalisation / observable readout audit:")
    print(
        f"    lambda_solver = {photon_solver.lambda_affine:.6e} s, "
        f"lambda_target = {setup['lambda_total']:.6e} s, "
        f"rel.diff = {(photon_solver.lambda_affine - setup['lambda_total']) / setup['lambda_total']:.3e}"
    )
    print(f"    kappa(lambda_solver) = {kappa_solver:+.8e}")
    print(
        f"    kappa(lambda_target) = {kappa_target_norm:+.8e}   "
        f"delta = {kappa_target_norm - kappa_solver:+.3e}"
    )
    print(f"    |gamma|(lambda_target) = {gamma_target_norm:.8e}")
    print(f"    mu(lambda_target)      = {mu_target_norm:.8e}")
    print(f"    1 - D_A_ray / D_A_FLRW = {da_deficit:+.8e}")
    print(f"    1 - sqrt(det A_target) = {da_deficit_from_det:+.8e}")
    print(f"    [1 - D_A/D_A_FLRW] - kappa(lambda_target) = {da_deficit - kappa_target_norm:+.3e}")
    print()

    R_total = np.zeros((n_states, 2, 2))
    R_block1 = np.zeros((n_states, 2, 2))
    R_block2 = np.zeros((n_states, 2, 2))
    R_block3 = np.zeros((n_states, 2, 2))
    R_block1_hess = np.zeros((n_states, 2, 2))
    R_block1_scalar = np.zeros((n_states, 2, 2))
    R_block3_hess = np.zeros((n_states, 2, 2))
    R_block3_scalar = np.zeros((n_states, 2, 2))
    R_block3_hess_A = np.zeros((n_states, 2, 2))
    R_block3_hess_B = np.zeros((n_states, 2, 2))
    R_block3_hess_C = np.zeros((n_states, 2, 2))
    R_block3_hess_D = np.zeros((n_states, 2, 2))
    R_direct_hess = np.zeros((n_states, 2, 2))
    R_direct_lapse = np.zeros((n_states, 2, 2))
    R_direct_mix_left = np.zeros((n_states, 2, 2))
    R_direct_los_trace = np.zeros((n_states, 2, 2))
    R_direct_screen = np.zeros((n_states, 2, 2))
    R_direct_mix_right = np.zeros((n_states, 2, 2))
    R_total_conformalbasis = np.zeros((n_states, 2, 2))
    radii = np.zeros(n_states)
    closure_err_b1 = 0.0
    closure_err_b3 = 0.0
    direct_hess_closure = 0.0
    max_rel_direct_hess = 0.0

    max_rel_code_vs_full = 0.0
    max_rel_code_vs_spatial = 0.0
    max_rel_trace_code_vs_spatial = 0.0
    max_abs_asym = 0.0
    max_abs_e0 = 0.0
    worst_full_idx = 0
    worst_spatial_idx = 0

    for idx, state in enumerate(history):
        fields = sample_local_fields(metric, state)
        a = fields["a"]
        phi_si = fields["phi_si"]
        blocks = rebuild_riemann_blocks_from_fields(fields)
        Rd_k00l, Rd_0lki, Rd_kijl = blocks
        pieces = decompose_riemann_blocks(fields)
        g_mu_nu = build_metric_tensor(a, phi_si)
        k_mu = state[4:8]
        e1_mu = state[8:12]
        e2_mu = state[12:16]
        radii[idx] = np.linalg.norm(state[1:4] - setup["center"])

        zeros_33 = np.zeros_like(Rd_k00l)
        zeros_333 = np.zeros_like(Rd_0lki)
        zeros_3333 = np.zeros_like(Rd_kijl)

        R_code = optical_tidal_matrix_optimized(Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu, g_mu_nu)
        R_full = optical_tidal_matrix_from_blocks(Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu, g_mu_nu)
        R_spatial = spatial_formula_rab(Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu)
        e1_mu_conformal, e2_mu_conformal = init_sachs_basis(k_mu, g_mu_nu, a, convention="conformal_metric")
        R_total_conformalbasis[idx] = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu_conformal, e2_mu_conformal, g_mu_nu
        )

        R_total[idx] = R_code
        R_block1[idx] = optical_tidal_matrix_optimized(Rd_k00l, zeros_333, zeros_3333, k_mu, e1_mu, e2_mu, g_mu_nu)
        R_block2[idx] = optical_tidal_matrix_optimized(zeros_33, Rd_0lki, zeros_3333, k_mu, e1_mu, e2_mu, g_mu_nu)
        R_block3[idx] = optical_tidal_matrix_optimized(zeros_33, zeros_333, Rd_kijl, k_mu, e1_mu, e2_mu, g_mu_nu)
        R_block1_hess[idx] = optical_tidal_matrix_optimized(pieces["block1_hess"], zeros_333, zeros_3333, k_mu, e1_mu, e2_mu, g_mu_nu)
        R_block1_scalar[idx] = optical_tidal_matrix_optimized(pieces["block1_scalar"], zeros_333, zeros_3333, k_mu, e1_mu, e2_mu, g_mu_nu)
        R_block3_hess[idx] = optical_tidal_matrix_optimized(zeros_33, zeros_333, pieces["block3_hess"], k_mu, e1_mu, e2_mu, g_mu_nu)
        R_block3_scalar[idx] = optical_tidal_matrix_optimized(zeros_33, zeros_333, pieces["block3_scalar"], k_mu, e1_mu, e2_mu, g_mu_nu)
        R_block3_hess_A[idx] = optical_tidal_matrix_optimized(zeros_33, zeros_333, pieces["block3_hess_A"], k_mu, e1_mu, e2_mu, g_mu_nu)
        R_block3_hess_B[idx] = optical_tidal_matrix_optimized(zeros_33, zeros_333, pieces["block3_hess_B"], k_mu, e1_mu, e2_mu, g_mu_nu)
        R_block3_hess_C[idx] = optical_tidal_matrix_optimized(zeros_33, zeros_333, pieces["block3_hess_C"], k_mu, e1_mu, e2_mu, g_mu_nu)
        R_block3_hess_D[idx] = optical_tidal_matrix_optimized(zeros_33, zeros_333, pieces["block3_hess_D"], k_mu, e1_mu, e2_mu, g_mu_nu)
        direct_terms = direct_hessian_optical_terms(a, fields["hess_phi"], k_mu, e1_mu, e2_mu)
        R_direct_hess[idx] = direct_terms["direct_total"]
        R_direct_lapse[idx] = direct_terms["direct_lapse"]
        R_direct_mix_left[idx] = direct_terms["direct_mix_left"]
        R_direct_los_trace[idx] = direct_terms["direct_los_trace"]
        R_direct_screen[idx] = direct_terms["direct_screen"]
        R_direct_mix_right[idx] = direct_terms["direct_mix_right"]
        closure_err_b1 = max(closure_err_b1, np.max(np.abs(R_block1[idx] - (R_block1_hess[idx] + R_block1_scalar[idx]))))
        closure_err_b3 = max(closure_err_b3, np.max(np.abs(R_block3[idx] - (R_block3_hess[idx] + R_block3_scalar[idx]))))
        direct_hess_closure = max(
            direct_hess_closure,
            np.max(
                np.abs(
                    R_direct_hess[idx]
                    - (
                        R_direct_lapse[idx]
                        + R_direct_mix_left[idx]
                        + R_direct_los_trace[idx]
                        + R_direct_screen[idx]
                        + R_direct_mix_right[idx]
                    )
                )
            ),
        )
        direct_scale = max(np.max(np.abs(R_block1_hess[idx] + R_block3_hess[idx])), 1e-300)
        max_rel_direct_hess = max(
            max_rel_direct_hess,
            np.max(np.abs(R_direct_hess[idx] - (R_block1_hess[idx] + R_block3_hess[idx]))) / direct_scale,
        )

        scale = max(np.max(np.abs(R_full)), 1e-300)
        rel_code_full = np.max(np.abs(R_code - R_full)) / scale
        rel_code_spatial = np.max(np.abs(R_code - R_spatial)) / scale
        if rel_code_full > max_rel_code_vs_full:
            max_rel_code_vs_full = rel_code_full
            worst_full_idx = idx
        if rel_code_spatial > max_rel_code_vs_spatial:
            max_rel_code_vs_spatial = rel_code_spatial
            worst_spatial_idx = idx
        trace_scale = max(abs(np.trace(R_full)), 1e-300)
        max_rel_trace_code_vs_spatial = max(
            max_rel_trace_code_vs_spatial,
            abs(np.trace(R_code) - np.trace(R_spatial)) / trace_scale,
        )
        max_abs_asym = max(max_abs_asym, abs(R_code[0, 1] - R_code[1, 0]))
        max_abs_e0 = max(max_abs_e0, abs(e1_mu[0]), abs(e2_mu[0]))

    print("  Pointwise operator audit along the recorded real ray:")
    print(f"    max |R_code - R_full| / |R_full|    = {max_rel_code_vs_full:.3e}   (step {worst_full_idx})")
    print(f"    max |R_code - R_spatial| / |R_full| = {max_rel_code_vs_spatial:.3e}   (step {worst_spatial_idx})")
    print(f"    max |tr(R_code)-tr(R_spatial)| / |tr(R_full)| = {max_rel_trace_code_vs_spatial:.3e}")
    print(f"    max |R12 - R21|                      = {max_abs_asym:.3e}")
    print(f"    max |e_A^0| along the ray           = {max_abs_e0:.3e}")
    print(f"    block1 decomposition closure        = {closure_err_b1:.3e}")
    print(f"    block3 decomposition closure        = {closure_err_b3:.3e}")
    print(f"    direct Hessian closure              = {direct_hess_closure:.3e}")
    print(f"    max |R_direct_hess - R_code_hess| / |R_code_hess| = {max_rel_direct_hess:.3e}")
    print()

    born_hat = normalize_spatial(target - setup["obs_pos"])
    fixed_screen1, fixed_screen2 = make_fixed_euclidean_screen(born_hat, setup["e_perp1"])
    k_mu_born = np.array([-1.0, *(c * born_hat)])
    e1_mu_fixed = np.array([0.0, *fixed_screen1])
    e2_mu_fixed = np.array([0.0, *fixed_screen2])
    lambda_samples = np.concatenate(([0.0], np.cumsum(dt_history)))

    R_born_fixed = np.zeros((n_states, 2, 2))
    R_realpos_bornfixed = np.zeros((n_states, 2, 2))
    R_realpos_localq_fixed = np.zeros((n_states, 2, 2))
    R_actualk_localscreen = np.zeros((n_states, 2, 2))
    R_actualk_conformalmetric = np.zeros((n_states, 2, 2))
    R_actualk_metricnormscreen = np.zeros((n_states, 2, 2))
    R_actualk_conformal_pred = np.zeros((n_states, 2, 2))
    R_actualk_metricnorm_pred = np.zeros((n_states, 2, 2))
    R_actualk_physical_from_conformal_pred = np.zeros((n_states, 2, 2))
    R_actualk_initbasis = np.zeros((n_states, 2, 2))
    a_series = np.zeros(n_states)
    phi_series = np.zeros(n_states)
    conformal_factor_series = np.zeros(n_states)
    metricnorm_factor_series = np.zeros(n_states)
    for idx, state in enumerate(history):
        fields = sample_local_fields(metric, state)
        a_series[idx] = fields["a"]
        phi_series[idx] = fields["phi_si"]
        pos_born = setup["obs_pos"] + lambda_samples[idx] * c * born_hat
        hess_born = halo.potential_hessian(pos_born[0], pos_born[1], pos_born[2])
        direct_born = direct_hessian_optical_terms(1.0, hess_born, k_mu_born, e1_mu_fixed, e2_mu_fixed)
        R_born_fixed[idx] = direct_born["direct_total"]

        pos_real = state[1:4]
        hess_real = halo.potential_hessian(pos_real[0], pos_real[1], pos_real[2])
        direct_real_bornfixed = direct_hessian_optical_terms(1.0, hess_real, k_mu_born, e1_mu_fixed, e2_mu_fixed)
        R_realpos_bornfixed[idx] = direct_real_bornfixed["direct_total"]

        q_real = normalize_spatial(state[4:8][1:4])
        local_screen1, local_screen2 = make_fixed_euclidean_screen(q_real, fixed_screen1)
        k_mu_local = np.array([-1.0, *(c * q_real)])
        e1_mu_local = np.array([0.0, *local_screen1])
        e2_mu_local = np.array([0.0, *local_screen2])
        direct_real_localq = direct_hessian_optical_terms(1.0, hess_real, k_mu_local, e1_mu_local, e2_mu_local)
        R_realpos_localq_fixed[idx] = direct_real_localq["direct_total"]

        k_mu_actual = state[4:8]
        direct_actualk_localscreen = direct_hessian_optical_terms(
            fields["a"], hess_real, k_mu_actual, e1_mu_local, e2_mu_local
        )
        R_actualk_localscreen[idx] = direct_actualk_localscreen["direct_total"]

        g_mu_nu = build_metric_tensor(fields["a"], fields["phi_si"])
        conformal_factor_series[idx] = 1.0 / (1.0 - 2.0 * fields["phi_si"] / (c * c))
        e1_mu_conformal, e2_mu_conformal = init_sachs_basis(
            k_mu_actual, g_mu_nu, fields["a"], convention="conformal_metric"
        )
        direct_actualk_conformal = direct_hessian_optical_terms(
            fields["a"], hess_real, k_mu_actual, e1_mu_conformal, e2_mu_conformal
        )
        R_actualk_conformalmetric[idx] = direct_actualk_conformal["direct_total"]
        R_actualk_conformal_pred[idx] = conformal_factor_series[idx] * R_actualk_localscreen[idx]

        spatial_scale = np.sqrt(g_mu_nu[1, 1])
        metricnorm_factor_series[idx] = 1.0 / g_mu_nu[1, 1]
        e1_mu_metricnorm = np.array([0.0, *(local_screen1 / spatial_scale)])
        e2_mu_metricnorm = np.array([0.0, *(local_screen2 / spatial_scale)])
        direct_actualk_metricnormscreen = direct_hessian_optical_terms(
            fields["a"], hess_real, k_mu_actual, e1_mu_metricnorm, e2_mu_metricnorm
        )
        R_actualk_metricnormscreen[idx] = direct_actualk_metricnormscreen["direct_total"]
        R_actualk_metricnorm_pred[idx] = metricnorm_factor_series[idx] * R_actualk_localscreen[idx]
        R_actualk_physical_from_conformal_pred[idx] = (1.0 / (fields["a"] * fields["a"])) * R_actualk_conformalmetric[idx]

        e1_mu_init, e2_mu_init = init_sachs_basis(k_mu_actual, g_mu_nu, fields["a"])
        direct_actualk_initbasis = direct_hessian_optical_terms(
            fields["a"], hess_real, k_mu_actual, e1_mu_init, e2_mu_init
        )
        R_actualk_initbasis[idx] = direct_actualk_initbasis["direct_total"]

    conformal_scale = np.maximum(
        np.max(np.abs(R_actualk_conformalmetric), axis=(1, 2)),
        1e-300,
    )[:, None, None]
    max_rel_conformal_factor_closure = np.max(
        np.abs(R_actualk_conformalmetric - R_actualk_conformal_pred) / conformal_scale
    )
    metricnorm_scale = np.maximum(
        np.max(np.abs(R_actualk_metricnormscreen), axis=(1, 2)),
        1e-300,
    )[:, None, None]
    max_rel_metricnorm_factor_closure = np.max(
        np.abs(R_actualk_metricnormscreen - R_actualk_metricnorm_pred) / metricnorm_scale
    )
    max_rel_conformal_factor_closure_rot = 0.0
    max_rel_physical_from_conformal_closure_rot = 0.0
    max_rel_initbasis_vs_metricnorm_rot = 0.0
    for idx, state in enumerate(history):
        fields = sample_local_fields(metric, state)
        g_mu_nu = build_metric_tensor(fields["a"], fields["phi_si"])
        spatial_scale = np.sqrt(g_mu_nu[1, 1])

        q_real = normalize_spatial(state[4:8][1:4])
        local_screen1, local_screen2 = make_fixed_euclidean_screen(q_real, fixed_screen1)
        e1_mu_conformal, e2_mu_conformal = init_sachs_basis(
            state[4:8], g_mu_nu, fields["a"], convention="conformal_metric"
        )
        conformal_spatial_scale = np.sqrt(conformal_factor_series[idx])
        Q_conformal = np.array([
            [np.dot(e1_mu_conformal[1:4] / conformal_spatial_scale, local_screen1), np.dot(e1_mu_conformal[1:4] / conformal_spatial_scale, local_screen2)],
            [np.dot(e2_mu_conformal[1:4] / conformal_spatial_scale, local_screen1), np.dot(e2_mu_conformal[1:4] / conformal_spatial_scale, local_screen2)],
        ])
        R_conformal_rot_to_local = Q_conformal.T @ R_actualk_conformalmetric[idx] @ Q_conformal
        scale_conformal = max(np.max(np.abs(R_actualk_conformal_pred[idx])), 1e-300)
        max_rel_conformal_factor_closure_rot = max(
            max_rel_conformal_factor_closure_rot,
            np.max(np.abs(R_conformal_rot_to_local - R_actualk_conformal_pred[idx])) / scale_conformal,
        )
        R_physical_from_conformal_rot_to_local = (
            Q_conformal.T @ R_actualk_physical_from_conformal_pred[idx] @ Q_conformal
        )
        scale_physical = max(np.max(np.abs(R_actualk_metricnormscreen[idx])), 1e-300)
        max_rel_physical_from_conformal_closure_rot = max(
            max_rel_physical_from_conformal_closure_rot,
            np.max(
                np.abs(R_physical_from_conformal_rot_to_local - R_actualk_metricnormscreen[idx])
            ) / scale_physical,
        )

        e1_mu_metricnorm = np.array([0.0, *(local_screen1 / spatial_scale)])
        e2_mu_metricnorm = np.array([0.0, *(local_screen2 / spatial_scale)])

        e1_mu_init, e2_mu_init = init_sachs_basis(state[4:8], g_mu_nu, fields["a"])
        Q = np.array([
            [e1_mu_init @ g_mu_nu @ e1_mu_metricnorm, e1_mu_init @ g_mu_nu @ e2_mu_metricnorm],
            [e2_mu_init @ g_mu_nu @ e1_mu_metricnorm, e2_mu_init @ g_mu_nu @ e2_mu_metricnorm],
        ])
        R_metricnorm_rot = Q @ R_actualk_metricnormscreen[idx] @ Q.T
        scale = max(np.max(np.abs(R_actualk_initbasis[idx])), 1e-300)
        max_rel_initbasis_vs_metricnorm_rot = max(
            max_rel_initbasis_vs_metricnorm_rot,
            np.max(np.abs(R_actualk_initbasis[idx] - R_metricnorm_rot)) / scale,
        )

    DP_total = replay_jacobi_on_frozen_operator(R_total, dt_history)
    if metric.sachs_screen_convention == "metric":
        DP_total_online = DP_total
        A1_total_online = None
    else:
        DP_total_online = replay_jacobi_on_frozen_operator(R_total_conformalbasis, dt_history)
        A1_total_online = replay_jacobi_first_order(R_total_conformalbasis, dt_history, lambda_final)
    DP_block1 = replay_jacobi_on_frozen_operator(R_block1, dt_history)
    DP_block2 = replay_jacobi_on_frozen_operator(R_block2, dt_history)
    DP_block3 = replay_jacobi_on_frozen_operator(R_block3, dt_history)
    DP_block1_hess = replay_jacobi_on_frozen_operator(R_block1_hess, dt_history)
    DP_block1_scalar = replay_jacobi_on_frozen_operator(R_block1_scalar, dt_history)
    DP_block3_hess = replay_jacobi_on_frozen_operator(R_block3_hess, dt_history)
    DP_block3_scalar = replay_jacobi_on_frozen_operator(R_block3_scalar, dt_history)
    DP_block3_hess_A = replay_jacobi_on_frozen_operator(R_block3_hess_A, dt_history)
    DP_block3_hess_B = replay_jacobi_on_frozen_operator(R_block3_hess_B, dt_history)
    DP_block3_hess_C = replay_jacobi_on_frozen_operator(R_block3_hess_C, dt_history)
    DP_block3_hess_D = replay_jacobi_on_frozen_operator(R_block3_hess_D, dt_history)
    A1_total = replay_jacobi_first_order(R_total, dt_history, lambda_final)
    A1_block1_hess = replay_jacobi_first_order(R_block1_hess, dt_history, lambda_final)
    A1_block3_hess = replay_jacobi_first_order(R_block3_hess, dt_history, lambda_final)
    A1_block3_A = replay_jacobi_first_order(R_block3_hess_A, dt_history, lambda_final)
    A1_direct_hess = replay_jacobi_first_order(R_direct_hess, dt_history, lambda_final)
    A1_direct_lapse = replay_jacobi_first_order(R_direct_lapse, dt_history, lambda_final)
    A1_direct_mix_left = replay_jacobi_first_order(R_direct_mix_left, dt_history, lambda_final)
    A1_direct_los_trace = replay_jacobi_first_order(R_direct_los_trace, dt_history, lambda_final)
    A1_direct_screen = replay_jacobi_first_order(R_direct_screen, dt_history, lambda_final)
    A1_direct_mix_right = replay_jacobi_first_order(R_direct_mix_right, dt_history, lambda_final)
    A1_born_fixed = replay_jacobi_first_order(R_born_fixed, dt_history, lambda_final)
    A1_realpos_bornfixed = replay_jacobi_first_order(R_realpos_bornfixed, dt_history, lambda_final)
    A1_realpos_localq_fixed = replay_jacobi_first_order(R_realpos_localq_fixed, dt_history, lambda_final)
    A1_actualk_localscreen = replay_jacobi_first_order(R_actualk_localscreen, dt_history, lambda_final)
    A1_actualk_conformalmetric = replay_jacobi_first_order(R_actualk_conformalmetric, dt_history, lambda_final)
    A1_actualk_metricnormscreen = replay_jacobi_first_order(R_actualk_metricnormscreen, dt_history, lambda_final)
    A1_total_conformalbasis = replay_jacobi_first_order(R_total_conformalbasis, dt_history, lambda_final)
    A1_actualk_conformal_pred = replay_jacobi_first_order(R_actualk_conformal_pred, dt_history, lambda_final)
    A1_actualk_metricnorm_pred = replay_jacobi_first_order(R_actualk_metricnorm_pred, dt_history, lambda_final)
    A1_actualk_physical_from_conformal_pred = replay_jacobi_first_order(
        R_actualk_physical_from_conformal_pred, dt_history, lambda_final
    )
    A1_actualk_initbasis = replay_jacobi_first_order(R_actualk_initbasis, dt_history, lambda_final)

    kappa_replay_total, _, _ = lensing_from_jacobi(DP_total[:4] / lambda_final)
    kappa_replay_total_online, _, _ = lensing_from_jacobi(DP_total_online[:4] / lambda_final)
    kappa_replay_b1, _, _ = lensing_from_jacobi(DP_block1[:4] / lambda_final)
    kappa_replay_b2, _, _ = lensing_from_jacobi(DP_block2[:4] / lambda_final)
    kappa_replay_b3, _, _ = lensing_from_jacobi(DP_block3[:4] / lambda_final)
    kappa_replay_b1_hess, _, _ = lensing_from_jacobi(DP_block1_hess[:4] / lambda_final)
    kappa_replay_b1_scalar, _, _ = lensing_from_jacobi(DP_block1_scalar[:4] / lambda_final)
    kappa_replay_b3_hess, _, _ = lensing_from_jacobi(DP_block3_hess[:4] / lambda_final)
    kappa_replay_b3_scalar, _, _ = lensing_from_jacobi(DP_block3_scalar[:4] / lambda_final)
    kappa_replay_b3_A, _, _ = lensing_from_jacobi(DP_block3_hess_A[:4] / lambda_final)
    kappa_replay_b3_B, _, _ = lensing_from_jacobi(DP_block3_hess_B[:4] / lambda_final)
    kappa_replay_b3_C, _, _ = lensing_from_jacobi(DP_block3_hess_C[:4] / lambda_final)
    kappa_replay_b3_D, _, _ = lensing_from_jacobi(DP_block3_hess_D[:4] / lambda_final)
    kappa_first_order_total, _, _ = lensing_from_jacobi(A1_total)
    if A1_total_online is None:
        kappa_first_order_total_online = kappa_first_order_total
    else:
        kappa_first_order_total_online, _, _ = lensing_from_jacobi(A1_total_online)
    kappa_first_order_b1_hess, _, _ = lensing_from_jacobi(A1_block1_hess)
    kappa_first_order_b3_hess, _, _ = lensing_from_jacobi(A1_block3_hess)
    kappa_first_order_b3_A, _, _ = lensing_from_jacobi(A1_block3_A)
    kappa_first_order_direct_hess, _, _ = lensing_from_jacobi(A1_direct_hess)
    kappa_first_order_direct_lapse, _, _ = lensing_from_jacobi(A1_direct_lapse)
    kappa_first_order_direct_mix_left, _, _ = lensing_from_jacobi(A1_direct_mix_left)
    kappa_first_order_direct_los_trace, _, _ = lensing_from_jacobi(A1_direct_los_trace)
    kappa_first_order_direct_screen, _, _ = lensing_from_jacobi(A1_direct_screen)
    kappa_first_order_direct_mix_right, _, _ = lensing_from_jacobi(A1_direct_mix_right)
    kappa_first_order_born_fixed, _, _ = lensing_from_jacobi(A1_born_fixed)
    kappa_first_order_realpos_bornfixed, _, _ = lensing_from_jacobi(A1_realpos_bornfixed)
    kappa_first_order_realpos_localq_fixed, _, _ = lensing_from_jacobi(A1_realpos_localq_fixed)
    kappa_first_order_actualk_localscreen, _, _ = lensing_from_jacobi(A1_actualk_localscreen)
    kappa_first_order_actualk_conformalmetric, _, _ = lensing_from_jacobi(A1_actualk_conformalmetric)
    kappa_first_order_actualk_metricnormscreen, _, _ = lensing_from_jacobi(A1_actualk_metricnormscreen)
    kappa_first_order_total_conformalbasis, _, _ = lensing_from_jacobi(A1_total_conformalbasis)
    kappa_first_order_actualk_conformal_pred, _, _ = lensing_from_jacobi(A1_actualk_conformal_pred)
    kappa_first_order_actualk_metricnorm_pred, _, _ = lensing_from_jacobi(A1_actualk_metricnorm_pred)
    kappa_first_order_actualk_physical_from_conformal_pred, _, _ = lensing_from_jacobi(
        A1_actualk_physical_from_conformal_pred
    )
    kappa_first_order_actualk_initbasis, _, _ = lensing_from_jacobi(A1_actualk_initbasis)
    sigma_cr_physical = setup["sigma_cr_physical"]
    sigma_cr_comoving = setup["sigma_cr_comoving"]
    kappa_scalar_born_physical = scalar_kappa_straight_ray(halo, setup["obs_pos"], target, setup["d_s"], sigma_cr_physical)
    kappa_scalar_born_comoving = scalar_kappa_straight_ray(halo, setup["obs_pos"], target, setup["d_s"], sigma_cr_comoving)
    kappa_analytic_physical = float(halo.kappa_analytic(np.array([b_mpc * one_Mpc]), sigma_cr_physical)[0])
    kappa_analytic_comoving = float(halo.kappa_analytic(np.array([b_mpc * one_Mpc]), sigma_cr_comoving)[0])

    idx_min_r = int(np.argmin(radii))
    trace_total = np.trace(R_total[idx_min_r])
    local_traces = {
        "block1_total": np.trace(R_block1[idx_min_r]),
        "block1_hess": np.trace(R_block1_hess[idx_min_r]),
        "block1_scalar": np.trace(R_block1_scalar[idx_min_r]),
        "block2": np.trace(R_block2[idx_min_r]),
        "block3_total": np.trace(R_block3[idx_min_r]),
        "block3_hess": np.trace(R_block3_hess[idx_min_r]),
        "block3_scalar": np.trace(R_block3_scalar[idx_min_r]),
        "block3_A": np.trace(R_block3_hess_A[idx_min_r]),
        "block3_B": np.trace(R_block3_hess_B[idx_min_r]),
        "block3_C": np.trace(R_block3_hess_C[idx_min_r]),
        "block3_D": np.trace(R_block3_hess_D[idx_min_r]),
    }

    int_trace_total = np.trapezoid(np.trace(R_total, axis1=1, axis2=2), dx=dt)
    int_trace_b1_hess = np.trapezoid(np.trace(R_block1_hess, axis1=1, axis2=2), dx=dt)
    int_trace_b1_scalar = np.trapezoid(np.trace(R_block1_scalar, axis1=1, axis2=2), dx=dt)
    int_trace_b3_hess = np.trapezoid(np.trace(R_block3_hess, axis1=1, axis2=2), dx=dt)
    int_trace_b3_scalar = np.trapezoid(np.trace(R_block3_scalar, axis1=1, axis2=2), dx=dt)
    trace_b3_A = np.trace(R_block3_hess_A, axis1=1, axis2=2)
    trace_b3_B = np.trace(R_block3_hess_B, axis1=1, axis2=2)
    trace_b3_C = np.trace(R_block3_hess_C, axis1=1, axis2=2)
    trace_b3_D = np.trace(R_block3_hess_D, axis1=1, axis2=2)
    int_trace_b3_A = np.trapezoid(trace_b3_A, dx=dt)
    int_trace_b3_B = np.trapezoid(trace_b3_B, dx=dt)
    int_trace_b3_C = np.trapezoid(trace_b3_C, dx=dt)
    int_trace_b3_D = np.trapezoid(trace_b3_D, dx=dt)
    idx_peak_b3_A = int(np.argmax(np.abs(trace_b3_A)))
    idx_peak_b3_B = int(np.argmax(np.abs(trace_b3_B)))
    idx_peak_b3_C = int(np.argmax(np.abs(trace_b3_C)))
    idx_peak_b3_D = int(np.argmax(np.abs(trace_b3_D)))

    ablations = {
        "minus_block1_hess": R_total - R_block1_hess,
        "minus_block1_scalar": R_total - R_block1_scalar,
        "minus_block3_hess": R_total - R_block3_hess,
        "minus_block3_scalar": R_total - R_block3_scalar,
        "minus_block3_A": R_total - R_block3_hess_A,
        "minus_block3_D": R_total - R_block3_hess_D,
        "minus_block1_hess_block3_A": R_total - R_block1_hess - R_block3_hess_A,
        "flip_block1_hess": R_total - 2.0 * R_block1_hess,
        "flip_block3_A": R_total - 2.0 * R_block3_hess_A,
        "flip_block1_hess_and_block3_A": R_total - 2.0 * R_block1_hess - 2.0 * R_block3_hess_A,
    }
    ablation_results = {
        name: replay_alpha(series, dt_history, lambda_final, kappa_analytic)
        for name, series in ablations.items()
    }

    print("  Frozen-operator Jacobi replay diagnostics:")
    print(f"    kappa_replay_transport_basis = {kappa_replay_total:+.8e}   rel.to.full = {abs(kappa_replay_total/kappa_manual - 1.0):.3e}")
    if metric.sachs_screen_convention != "metric":
        print(
            f"    kappa_replay_online_{metric.sachs_screen_convention} = {kappa_replay_total_online:+.8e}   "
            f"rel.to.full = {abs(kappa_replay_total_online/kappa_manual - 1.0):.3e}"
        )
        print("    note: the non-metric online solver rebuilds a local screen at each RHS call;")
        print("          the transported-basis replay is therefore a diagnostic mismatch, not the online operator.")
    print(f"    kappa_replay_block1 = {kappa_replay_b1:+.8e}")
    print(f"    kappa_replay_block2 = {kappa_replay_b2:+.8e}")
    print(f"    kappa_replay_block3 = {kappa_replay_b3:+.8e}")
    print(f"    kappa_replay_block1_hess   = {kappa_replay_b1_hess:+.8e}")
    print(f"    kappa_replay_block1_scalar = {kappa_replay_b1_scalar:+.8e}")
    print(f"    kappa_replay_block3_hess   = {kappa_replay_b3_hess:+.8e}")
    print(f"    kappa_replay_block3_scalar = {kappa_replay_b3_scalar:+.8e}")
    print(f"    kappa_replay_block3_A      = {kappa_replay_b3_A:+.8e}")
    print(f"    kappa_replay_block3_B      = {kappa_replay_b3_B:+.8e}")
    print(f"    kappa_replay_block3_C      = {kappa_replay_b3_C:+.8e}")
    print(f"    kappa_replay_block3_D      = {kappa_replay_b3_D:+.8e}")
    print()

    print("  Independent first-order Sachs replay on the recorded real ray:")
    print(
        f"    kappa_first_order_transport_basis = {kappa_first_order_total:+.8e}   "
        f"delta.to.transport = {kappa_first_order_total - kappa_replay_total:+.3e}"
    )
    if metric.sachs_screen_convention != "metric":
        print(
            f"    kappa_first_order_online_{metric.sachs_screen_convention} = {kappa_first_order_total_online:+.8e}   "
            f"delta.to.full = {kappa_first_order_total_online - kappa_manual:+.3e}"
        )
    print(
        f"    kappa_first_order_block1_hess = {kappa_first_order_b1_hess:+.8e}   "
        f"delta.to.full = {kappa_first_order_b1_hess - kappa_replay_b1_hess:+.3e}"
    )
    print(
        f"    kappa_first_order_block3_hess = {kappa_first_order_b3_hess:+.8e}   "
        f"delta.to.full = {kappa_first_order_b3_hess - kappa_replay_b3_hess:+.3e}"
    )
    print(
        f"    kappa_first_order_block3_A    = {kappa_first_order_b3_A:+.8e}   "
        f"delta.to.full = {kappa_first_order_b3_A - kappa_replay_b3_A:+.3e}"
    )
    print()

    print("  Direct Hessian-only first-order reference on the recorded real ray:")
    print(
        f"    kappa_direct_hess_total   = {kappa_first_order_direct_hess:+.8e}   "
        f"delta.to.code_hess = {kappa_first_order_direct_hess - kappa_first_order_b3_hess - kappa_first_order_b1_hess:+.3e}"
    )
    print(
        f"    kappa_direct_lapse        = {kappa_first_order_direct_lapse:+.8e}"
    )
    print(
        f"    kappa_direct_screen       = {kappa_first_order_direct_screen:+.8e}"
    )
    print(
        f"    kappa_direct_los_trace    = {kappa_first_order_direct_los_trace:+.8e}"
    )
    print(
        f"    kappa_direct_mix_left     = {kappa_first_order_direct_mix_left:+.8e}"
    )
    print(
        f"    kappa_direct_mix_right    = {kappa_first_order_direct_mix_right:+.8e}"
    )
    print("    note: this direct basis is geometric and does not coincide term-by-term")
    print("          with the combinatorial block3 A/B/C/D split printed above.")
    print()

    print("  Observable-comparison ladder on the same NFW case:")
    print("    variant                          kappa              alpha_to_analytic")
    print(f"    analytic_nfw_physical_scr    {kappa_analytic_physical:+.8e}   {kappa_analytic_physical / kappa_analytic:.6f}")
    print(f"    analytic_nfw_comoving_scr    {kappa_analytic_comoving:+.8e}   {kappa_analytic_comoving / kappa_analytic:.6f}")
    print(f"    scalar_born_physical_scr     {kappa_scalar_born_physical:+.8e}   {kappa_scalar_born_physical / kappa_analytic:.6f}")
    print(f"    scalar_born_comoving_scr     {kappa_scalar_born_comoving:+.8e}   {kappa_scalar_born_comoving / kappa_analytic:.6f}")
    print(f"    born_ray_euclidean_local     {kappa_first_order_born_fixed:+.8e}   {kappa_first_order_born_fixed / kappa_analytic:.6f}")
    print(f"    real_pos_born_screen         {kappa_first_order_realpos_bornfixed:+.8e}   {kappa_first_order_realpos_bornfixed / kappa_analytic:.6f}")
    print(f"    real_pos_local_tangent       {kappa_first_order_realpos_localq_fixed:+.8e}   {kappa_first_order_realpos_localq_fixed / kappa_analytic:.6f}")
    print(f"    real_k_euclidean_local       {kappa_first_order_actualk_localscreen:+.8e}   {kappa_first_order_actualk_localscreen / kappa_analytic:.6f}")
    print(f"    real_k_conformal_sachs_dir   {kappa_first_order_actualk_conformalmetric:+.8e}   {kappa_first_order_actualk_conformalmetric / kappa_analytic:.6f}")
    print(f"    real_k_conformal_sachs_full  {kappa_first_order_total_conformalbasis:+.8e}   {kappa_first_order_total_conformalbasis / kappa_analytic:.6f}")
    print(f"    real_k_physical_sachs_dir    {kappa_first_order_actualk_metricnormscreen:+.8e}   {kappa_first_order_actualk_metricnormscreen / kappa_analytic:.6f}")
    print(f"    real_k_physical_sachs_init   {kappa_first_order_actualk_initbasis:+.8e}   {kappa_first_order_actualk_initbasis / kappa_analytic:.6f}")
    print(f"    real_ray_transported_screen  {kappa_first_order_direct_hess:+.8e}   {kappa_first_order_direct_hess / kappa_analytic:.6f}")
    print("    reading:")
    print("      - The physical-screen references use Sigma_cr/(1+z_l), the usual weak-lensing normalization in this repo.")
    print("      - The conformal-screen references use the comoving Sigma_cr without the extra /(1+z_l) factor.")
    print("      - On the straight Born ray, the Euclidean local screen and the comoving-Sigma_cr scalar reference close almost exactly.")
    print("      - Replacing the physical Sachs screen by Fleury's conformal Sachs screen drops the high branch to the conformal branch.")
    print("      - The remaining conformal-vs-analytic difference on the real ray is small, and is due to the real path/tangent rather than to the screen convention.")
    print()

    screen_boost = kappa_first_order_actualk_metricnormscreen / kappa_first_order_actualk_conformalmetric
    conformal_boost_min = 1.0 / (a_series[idx_min_r] * a_series[idx_min_r])
    potential_boost_min = 1.0 / (1.0 - 2.0 * phi_series[idx_min_r] / (c * c))
    print("  Screen-convention audit:")
    print("    isotropic metric law: e_conf = e_euclid / sqrt(1 - 2 Phi/c^2), e_phys = e_conf / a")
    print("    corresponding operator law: R_conf = R_euclid / (1 - 2 Phi/c^2), R_phys = R_conf / a^2")
    print(f"    max |R_conf(local) - R_euclid/(1-2Phi/c^2)| / |R_conf(local)| = {max_rel_conformal_factor_closure_rot:.3e}")
    print(f"    max |R_phys(local) - R_conf(local)/a^2| / |R_phys(local)|     = {max_rel_physical_from_conformal_closure_rot:.3e}")
    print(f"    max |R_metricnorm - R_local/g_perp| / |R_metricnorm| = {max_rel_metricnorm_factor_closure:.3e}")
    print(f"    max |R_initbasis - Q R_metricnorm Q^T| / |R_initbasis| = {max_rel_initbasis_vs_metricnorm_rot:.3e}")
    print(
        f"    kappa_conf_pred_from_euclid      = {kappa_first_order_actualk_conformal_pred:+.8e}   "
        f"delta = {kappa_first_order_actualk_conformal_pred - kappa_first_order_actualk_conformalmetric:+.3e}"
    )
    print(
        f"    kappa_phys_pred_from_conformal   = {kappa_first_order_actualk_physical_from_conformal_pred:+.8e}   "
        f"delta = {kappa_first_order_actualk_physical_from_conformal_pred - kappa_first_order_actualk_metricnormscreen:+.3e}"
    )
    print(f"    kappa_boost_physical/conformal   = {screen_boost:.6f}")
    print(f"    1/(1 - 2 Phi/c^2) at r_min       = {potential_boost_min:.6f}")
    print(f"    1/a(r_min)^2                     = {conformal_boost_min:.6f}")
    print(f"    1/g_perp at r_min                = {metricnorm_factor_series[idx_min_r]:.6f}")
    print("    reading:")
    print("      - In this diagonal isotropic metric, the exact GR screen split is conformal Sachs then physical Sachs, not Euclidean then physical directly.")
    print("      - Along this NFW ray, the physical-vs-conformal jump is almost entirely the conformal factor 1/a^2; the Phi/c^2 correction is tiny.")
    print("      - The high branch is therefore a physical-screen projection of a Jacobi/operator chain that is otherwise being read in conformal-normalized variables.")
    print()

    residual_conformal_ratio = kappa_first_order_total_conformalbasis / kappa_analytic_comoving
    path_effect_ratio = kappa_first_order_total_conformalbasis / kappa_first_order_born_fixed
    print("  Conformal-screen residual audit:")
    print(f"    analytic_nfw_comoving_scr        = {kappa_analytic_comoving:+.8e}")
    print(f"    scalar_born_comoving_scr         = {kappa_scalar_born_comoving:+.8e}   delta = {kappa_scalar_born_comoving - kappa_analytic_comoving:+.3e}")
    print(f"    born_ray_conformal_equivalent    = {kappa_first_order_born_fixed:+.8e}   delta = {kappa_first_order_born_fixed - kappa_analytic_comoving:+.3e}")
    print(f"    real_ray_conformal_sachs_full    = {kappa_first_order_total_conformalbasis:+.8e}   delta = {kappa_first_order_total_conformalbasis - kappa_analytic_comoving:+.3e}")
    print(f"    real_ray_conformal_sachs_dir     = {kappa_first_order_actualk_conformalmetric:+.8e}   delta = {kappa_first_order_actualk_conformalmetric - kappa_analytic_comoving:+.3e}")
    print(f"    ratio real_conformal / analytic_comoving = {residual_conformal_ratio:.6f}")
    print(f"    ratio real_conformal / born_conformal    = {path_effect_ratio:.6f}")
    print("    reading:")
    print("      - Once Sigma_cr is made comoving, the straight-ray scalar and Born-screen references close on the conformal branch.")
    print("      - The remaining real-ray conformal deficit is only a small path/tangent effect, not a second large convention mismatch.")
    print()

    print(f"  Closest approach: r_min = {radii[idx_min_r]/one_Mpc:.6f} Mpc at step {idx_min_r}")
    print(f"    tr(R_total)         = {trace_total:+.8e}")
    print(f"    tr(R_block1_total)  = {local_traces['block1_total']:+.8e}")
    print(f"    tr(R_block1_hess)   = {local_traces['block1_hess']:+.8e}")
    print(f"    tr(R_block1_scalar) = {local_traces['block1_scalar']:+.8e}")
    print(f"    tr(R_block2)        = {local_traces['block2']:+.8e}")
    print(f"    tr(R_block3_total)  = {local_traces['block3_total']:+.8e}")
    print(f"    tr(R_block3_hess)   = {local_traces['block3_hess']:+.8e}")
    print(f"    tr(R_block3_scalar) = {local_traces['block3_scalar']:+.8e}")
    print(f"    tr(R_block3_A)      = {local_traces['block3_A']:+.8e}")
    print(f"    tr(R_block3_B)      = {local_traces['block3_B']:+.8e}")
    print(f"    tr(R_block3_C)      = {local_traces['block3_C']:+.8e}")
    print(f"    tr(R_block3_D)      = {local_traces['block3_D']:+.8e}")
    print()

    print("  Simple trace integrals along the recorded real ray:")
    print(f"    int tr(R_total)         dλ = {int_trace_total:+.8e}")
    print(f"    int tr(R_block1_hess)   dλ = {int_trace_b1_hess:+.8e}")
    print(f"    int tr(R_block1_scalar) dλ = {int_trace_b1_scalar:+.8e}")
    print(f"    int tr(R_block3_hess)   dλ = {int_trace_b3_hess:+.8e}")
    print(f"    int tr(R_block3_scalar) dλ = {int_trace_b3_scalar:+.8e}")
    print(f"    int tr(R_block3_A)      dλ = {int_trace_b3_A:+.8e}")
    print(f"    int tr(R_block3_B)      dλ = {int_trace_b3_B:+.8e}")
    print(f"    int tr(R_block3_C)      dλ = {int_trace_b3_C:+.8e}")
    print(f"    int tr(R_block3_D)      dλ = {int_trace_b3_D:+.8e}")
    print()
    print("  Peak absolute trace by block3 Hessian subterm:")
    print(
        f"    A: peak tr = {trace_b3_A[idx_peak_b3_A]:+.8e} at step {idx_peak_b3_A}, "
        f"r = {radii[idx_peak_b3_A]/one_Mpc:.6f} Mpc"
    )
    print(
        f"    B: peak tr = {trace_b3_B[idx_peak_b3_B]:+.8e} at step {idx_peak_b3_B}, "
        f"r = {radii[idx_peak_b3_B]/one_Mpc:.6f} Mpc"
    )
    print(
        f"    C: peak tr = {trace_b3_C[idx_peak_b3_C]:+.8e} at step {idx_peak_b3_C}, "
        f"r = {radii[idx_peak_b3_C]/one_Mpc:.6f} Mpc"
    )
    print(
        f"    D: peak tr = {trace_b3_D[idx_peak_b3_D]:+.8e} at step {idx_peak_b3_D}, "
        f"r = {radii[idx_peak_b3_D]/one_Mpc:.6f} Mpc"
    )
    print()

    print("  Ablation replay on the recorded real ray:")
    print("    remove term                     kappa_ablated     alpha_ablated   delta_alpha")
    for name in (
        "minus_block1_hess",
        "minus_block1_scalar",
        "minus_block3_hess",
        "minus_block3_scalar",
        "minus_block3_A",
        "minus_block3_D",
        "minus_block1_hess_block3_A",
    ):
        kappa_ab, alpha_ab = ablation_results[name]
        print(
            f"    {name:26s} {kappa_ab:+.8e}   {alpha_ab:.6f}    {alpha_ab - (kappa_solver / kappa_analytic):+.6f}"
        )
    print()

    print("  Patched-sign replay on the recorded real ray:")
    print("    patch                           kappa_patched     alpha_patched   delta_alpha")
    for name in (
        "flip_block1_hess",
        "flip_block3_A",
        "flip_block1_hess_and_block3_A",
    ):
        kappa_ab, alpha_ab = ablation_results[name]
        print(
            f"    {name:31s} {kappa_ab:+.8e}   {alpha_ab:.6f}    {alpha_ab - (kappa_solver / kappa_analytic):+.6f}"
        )
    print()

    print("  Interpretation:")
    print("    - If the pointwise R_AB mismatches were large, the contraction/formula would be suspect.")
    print("    - If the frozen-operator replay tracks the full solution, the bias is already encoded in R_AB along the ray.")
    print("=" * 88)


if __name__ == "__main__":
    main()