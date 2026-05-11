#!/usr/bin/env python3
r"""
Multi-photon weak-lensing simulation through an **NFW halo**.

Realistic-ish parameters:
    M_200  = 10^1^5 Msun   (massive cluster)
    c_NFW  = 5
    R_200  ~ 2 Mpc,  r_s ~ 413 kpc

Grid: 10 Mpc box, 512^3  -> dx ~ 20 kpc  (r_s / dx ~ 21  --  well-resolved).

Geometry:  observer at z = 0.5 Mpc, halo centre at z = 5 Mpc.
           D_l = 4.5 Mpc, D_s = 9 Mpc   -> thin-lens D_s = 2 D_l.

Outputs ``lensing_nfw_results.npz`` for the analysis script.
"""

import numpy as np
from scipy import interpolate
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# -- excalibur imports ----------------------------------------------
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_4d_fast import InterpolatorFast as Interpolator4DFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.core.constants import c, G, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.objects.nfw_halo import NFWHalo
from excalibur.observables.lensing_conventions import (
    DEFAULT_LENSING_REFERENCE_CONVENTION,
    lensing_convention_label,
    sigma_cr_conventions,
)
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from excalibur.io.filename_utils import RunNamer


# =====================================================================
#  HELPER: build a photon aimed at (halo_center + offset)
# =====================================================================
def make_photon(obs_pos, target, metric, eta_0, a_0):
    """Create a Photon with null-condition k^mu and Sachs basis."""
    obs_4d = np.array([eta_0, *obs_pos])
    direction = target - obs_pos
    direction /= np.linalg.norm(direction)
    screen_convention = getattr(metric, "sachs_screen_convention", "metric")

    g = metric.metric_tensor(obs_4d)
    if screen_convention == "conformal_metric":
        g_init = g / (a_0 * a_0)
        basis_a = 1.0
    else:
        g_init = g
        basis_a = a_0

    k_spatial = direction * c
    spatial_sq = (g_init[1, 1] * k_spatial[0]**2
                + g_init[2, 2] * k_spatial[1]**2
                + g_init[3, 3] * k_spatial[2]**2)
    k0 = -np.sqrt(abs(-spatial_sq / g_init[0, 0]))   # backward tracing
    k_mu = np.array([k0, *k_spatial])

    e1, e2 = init_sachs_basis(-k_mu, g_init, basis_a, convention=screen_convention)

    p = Photon(obs_4d.copy(), k_mu.copy())
    p.e1     = e1.copy()
    p.e2     = e2.copy()
    p.D_flat = np.array([0.0, 0.0, 0.0, 0.0])   # Jacobi IC: D(0) = 0
    p.P_flat = np.array([1.0, 0.0, 0.0, 1.0])   # Jacobi IC: P(0) = I
    return p


# =====================================================================
#  MAIN
# =====================================================================
def main():
    t_total = time.time()

    # =================================================================
    # 1. COSMOLOGY
    # =================================================================
    print("=" * 70)
    print("  NFW LENSING CONE SIMULATION")
    print("=" * 70)
    print("\n1. Cosmology ...")
    H0 = 70.0
    Omega_m, Omega_lambda = 0.3, 0.7
    cosmo = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=0,
                           Omega_lambda=Omega_lambda)

    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0   = cosmo.a_of_eta(eta_0)

    eta_min = 0.5 * eta_0
    eta_arr = np.linspace(eta_min, eta_0, 2000)
    a_arr   = np.array([cosmo.a_of_eta(e) for e in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic",
                                     fill_value="extrapolate")
    print(f"   eta0 = {eta_0:.4e} s,  a(eta0) = {a_0:.6f}")

    # =================================================================
    # 2. GRID + NFW HALO
    # =================================================================
    print("2. Grid + NFW halo ...")
    N         = 512
    box_Mpc   = 10.0
    grid_size = box_Mpc * one_Mpc

    grid = Grid(
        shape   = (N, N, N),
        spacing = (grid_size / N,) * 3,
        origin  = np.array([0.0, 0.0, 0.0]),
    )

    # NFW halo at centre of the box
    M_200 = 1e15 * one_Msun
    c_NFW = 5.0
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    halo = NFWHalo(M_200, c_NFW, center)

    print(f"   {halo}")
    print(f"   R_200 = {halo.R_200/one_Mpc:.3f} Mpc,  r_s = {halo.r_s/one_Mpc*1000:.0f} kpc")

    # Build the potential field on the grid
    dx_Mpc = (grid_size / N) / one_Mpc
    print(f"   Grid {N}^3,  box = {box_Mpc:.0f} Mpc,  dx = {dx_Mpc*1000:.1f} kpc")
    print(f"   r_s / dx = {halo.r_s / (grid_size/N):.1f}")

    t_grid = time.time()
    x1d = np.linspace(0, grid_size, N)
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    phi_field = halo.potential(X, Y, Z)
    grid.add_field("Phi", phi_field)
    del X, Y, Z   # free memory
    print(f"   [ok] Potential field computed in {time.time()-t_grid:.1f}s")

    phi_max = np.max(np.abs(phi_field))
    print(f"   |Phi|_max = {phi_max:.3e}   ->  Phi/c^2 = {phi_max/c**2:.3e}")

    # =================================================================
    # 3. METRIC
    # =================================================================
    print("3. Metric ...")
    interpolator = Interpolator4DFast(grid, boundary="clamp", scheme="tricubic")
    metric = PerturbedFLRWMetricFast(
        a_of_eta       = a_of_eta,
        grid           = grid,
        interpolator   = interpolator,
        adot_of_eta    = cosmo.adot_of_eta,
        cosmology      = cosmo,
        enable_lensing = True,
        slow_roll      = True,
        sachs_screen_convention = "conformal_metric",
    )
    print("   [ok] metric ready")

    # =================================================================
    # 4. PHOTON CONE
    # =================================================================
    print("4. Photon cone ...")

    # Observer near the bottom face of the box
    obs_z_Mpc = 0.5
    obs_pos = np.array([box_Mpc/2, box_Mpc/2, obs_z_Mpc]) * one_Mpc

    # Direction basis
    dir_to_center = center - obs_pos
    dist_to_center = np.linalg.norm(dir_to_center)
    dir_hat = dir_to_center / dist_to_center

    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)
    e_perp2 = np.cross(dir_hat, e_perp1)
    e_perp2 /= np.linalg.norm(e_perp2)

    # Impact parameters: fine sampling from 0 to 4 Mpc (covers 2 R_200)
    # then sparse out to 4.5 Mpc (background reference)
    R200_Mpc = halo.R_200 / one_Mpc
    rs_Mpc = halo.r_s / one_Mpc

    b_values_Mpc = np.unique(np.sort(np.concatenate([
        np.array([0.0]),
        np.linspace(0.02, 0.2, 10),             # inner core (< r_s/2)
        np.linspace(0.25, rs_Mpc, 10),           # up to r_s
        np.linspace(rs_Mpc + 0.05, R200_Mpc, 15),  # r_s to R_200
        np.linspace(R200_Mpc + 0.1, 3.0, 10),   # beyond R_200
        np.linspace(3.5, 4.5, 5),               # far field / background
        np.logspace(np.log10(0.02), np.log10(4.5), 30),  # log fill
    ])))
    b_values = b_values_Mpc * one_Mpc

    # --- 1D radial profile photons ---
    photons_profile = []
    b_profile_Mpc = []
    for b in b_values:
        target = center + b * e_perp1
        p = make_photon(obs_pos, target, metric, eta_0, a_0)
        photons_profile.append(p)
        b_profile_Mpc.append(b / one_Mpc)
    n_profile = len(photons_profile)

    # --- 2D map photons ---
    map_half_Mpc = 3.0   # +/-3 Mpc (covers ~1.5 R_200)
    n_map_1d = 25         # 25x25 = 625 photons
    b1_arr = np.linspace(-map_half_Mpc, map_half_Mpc, n_map_1d) * one_Mpc
    b2_arr = np.linspace(-map_half_Mpc, map_half_Mpc, n_map_1d) * one_Mpc

    photons_map = []
    map_b1_Mpc = []
    map_b2_Mpc = []
    for b1 in b1_arr:
        for b2 in b2_arr:
            target = center + b1 * e_perp1 + b2 * e_perp2
            p = make_photon(obs_pos, target, metric, eta_0, a_0)
            photons_map.append(p)
            map_b1_Mpc.append(b1 / one_Mpc)
            map_b2_Mpc.append(b2 / one_Mpc)

    n_map = len(photons_map)
    n_total = n_profile + n_map
    print(f"   Profile photons : {n_profile}  (b = 0 .. {b_values_Mpc[-1]:.2f} Mpc)")
    print(f"   Map photons     : {n_map}  ({n_map_1d}x{n_map_1d} grid, +/-{map_half_Mpc} Mpc)")
    print(f"   Total photons   : {n_total}")

    # =================================================================
    # 5. INTEGRATOR  --  D_s = 2 x D_l geometry
    # =================================================================
    print("5. Integrator ...")
    dt_init = grid.spacing[0] / (5.0 * c)    # positive affine step

    D_l  = np.linalg.norm(center - obs_pos)           # ~4.5 Mpc
    D_s  = 2.0 * D_l                                  # ~9 Mpc
    max_dist_in_box = grid_size - np.min(obs_pos)
    D_s  = min(D_s, 0.95 * max_dist_in_box)
    step_length = c * dt_init
    n_steps = int(np.ceil(D_s / step_length))
    D_s_Mpc = D_s / one_Mpc

    print(f"   D_l  = {D_l/one_Mpc:.2f} Mpc  (observer  -> halo)")
    print(f"   D_s  = {D_s_Mpc:.2f} Mpc  (source plane)")
    print(f"   step = {step_length/one_Mpc*1000:.1f} kpc   ->  n_steps = {n_steps}")
    print(f"   r_s / step = {halo.r_s/step_length:.1f}")

    integrator = Integrator(
        metric     = metric,
        dt         = dt_init,
        mode       = "sequential",
        integrator = "rk4",
        rtol       = 1e-8,
        atol       = 1e-12,
        dt_min     = 1e-20,
        dt_max     = abs(dt_init) * 50,
    )

    # =================================================================
    # 6. INTEGRATE
    # =================================================================
    all_photons = photons_profile + photons_map
    lambda_S = n_steps * dt_init

    print(f"\n6. Integrating {n_total} photons x {n_steps} steps ...")
    print(f"   lambda_S = {lambda_S:.6e} s")
    t_int = time.time()

    kappas = np.empty(n_total)
    mus    = np.empty(n_total)
    gammas = np.empty(n_total)
    D_flats = np.empty((n_total, 4))
    final_pos = np.empty((n_total, 3))

    for idx, photon in enumerate(all_photons):
        photon.record()
        integrator.integrate_single(
            photon,
            stop_mode  = "steps",
            stop_value = n_steps,
        )
        D_norm = photon.D_flat / lambda_S
        kappa, mu, shear = lensing_from_jacobi(D_norm)
        kappas[idx] = kappa
        mus[idx]    = mu
        gammas[idx] = shear
        D_flats[idx] = D_norm
        final_pos[idx] = photon.x[1:4]

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_int
            rate = (idx + 1) / elapsed
            eta_remaining = (n_total - idx - 1) / rate if rate > 0 else 0
            dk = kappa - kappas[0] if idx > 0 else 0.0
            print(f"   [{idx+1:4d}/{n_total}]  "
                  f"kappa = {kappa:+.6e}  dkappa = {dk:+.3e}  |gamma| = {shear:.3e}  "
                  f"({elapsed:.0f}s, ~{eta_remaining:.0f}s left)")

    dt_elapsed = time.time() - t_int
    print(f"   [ok] Done in {dt_elapsed:.1f} s  ({dt_elapsed/n_total:.2f} s/photon)")

    # =================================================================
    # 7. EXTRACT RESULTS
    # =================================================================
    b_prof  = np.array(b_profile_Mpc)
    k_prof  = kappas[:n_profile]
    g_prof  = gammas[:n_profile]
    m_prof  = mus[:n_profile]

    b1_map = np.array(map_b1_Mpc)
    b2_map = np.array(map_b2_Mpc)
    k_map  = kappas[n_profile:]
    g_map  = gammas[n_profile:]
    m_map  = mus[n_profile:]

    # =================================================================
    # 8. NFW ANALYTIC PREDICTIONS
    # =================================================================
    D_s_actual = np.mean(np.linalg.norm(final_pos - obs_pos, axis=1))
    D_ls_actual = D_s_actual - D_l
    if D_ls_actual <= 0:
        D_ls_actual = D_l

    # Effective Sigma_cr matching excalibur's Sachs conventions (see note in
    # run_lensing_nfw_amr.py): divide comoving Sigma_cr by (1+z_l).
    from scipy.optimize import brentq as _brentq
    z_l = _brentq(lambda z: cosmo.comoving_distance(z) - D_l, 0.0, 5.0)

    Sigma_cr_comoving, Sigma_cr_physical = sigma_cr_conventions(D_l, D_s_actual, D_ls_actual, z_l)
    Sigma_cr = Sigma_cr_comoving
    Sigma_cr_Mpc2 = Sigma_cr * one_Mpc**2 / one_Msun
    Sigma_cr_physical_Mpc2 = Sigma_cr_physical * one_Mpc**2 / one_Msun
    reference_label = lensing_convention_label(DEFAULT_LENSING_REFERENCE_CONVENTION)

    print(f"\n   D_l  = {D_l/one_Mpc:.2f} Mpc")
    print(f"   D_s  = {D_s_actual/one_Mpc:.2f} Mpc  (actual)")
    print(f"   D_ls = {D_ls_actual/one_Mpc:.2f} Mpc")
    print(f"   Sigma_cr_ref ({reference_label}) = {Sigma_cr_Mpc2:.3e} Msun/Mpc^2")
    print(f"   Sigma_cr_physical = {Sigma_cr_physical_Mpc2:.3e} Msun/Mpc^2")

    # Analytic NFW kappa and gamma
    b_m = np.abs(b_prof) * one_Mpc
    b_m = np.maximum(b_m, 1e-3 * one_Mpc)  # avoid b=0 singularity
    k_analytic_comoving = halo.kappa_analytic(b_m, Sigma_cr_comoving)
    g_analytic_comoving = halo.gamma_analytic(b_m, Sigma_cr_comoving)
    k_analytic_physical = halo.kappa_analytic(b_m, Sigma_cr_physical)
    g_analytic_physical = halo.gamma_analytic(b_m, Sigma_cr_physical)
    k_analytic = k_analytic_comoving
    g_analytic = g_analytic_comoving

    # =================================================================
    # 9. SAVE
    # =================================================================
    namer = RunNamer(
        "lensing_nfw",
        integrator="rk4",
        metric="FLRWP1",
        profile="nfw",
        M=M_200 / one_Msun,
        c_NFW=c_NFW,
        Rvir=round(halo.R_200 / one_Mpc, 3),
        rs=round(halo.r_s / one_Mpc, 4),
        zl=round(z_l, 5),
        Dl=round(D_l / one_Mpc, 1),
        Ds=round(D_s_actual / one_Mpc, 1),
        obs=(round(obs_pos[0]/one_Mpc, 1),
             round(obs_pos[1]/one_Mpc, 1),
             round(obs_pos[2]/one_Mpc, 1)),
        box_Mpc=box_Mpc,
        N=N,
        Nph=n_total,
    )
    outfile = namer.npz()

    np.savez(
        outfile,
        # Profile
        b_profile_Mpc=b_prof,
        kappa_profile=k_prof,
        gamma_profile=g_prof,
        mu_profile=m_prof,
        kappa_analytic=k_analytic,
        gamma_analytic=g_analytic,
        kappa_analytic_comoving=k_analytic_comoving,
        gamma_analytic_comoving=g_analytic_comoving,
        kappa_analytic_physical=k_analytic_physical,
        gamma_analytic_physical=g_analytic_physical,
        D_flat_profile=D_flats[:n_profile],
        # Map
        b1_map_Mpc=b1_map,
        b2_map_Mpc=b2_map,
        kappa_map=k_map,
        gamma_map=g_map,
        mu_map=m_map,
        D_flat_map=D_flats[n_profile:],
        n_map_1d=n_map_1d,
        map_half_Mpc=map_half_Mpc,
        # Halo params
        halo_type="NFW",
        N_grid=N,
        box_Mpc=box_Mpc,
        M_Msun=M_200 / one_Msun,
        M_200_Msun=M_200 / one_Msun,
        c_NFW=c_NFW,
        R_200_Mpc=halo.R_200 / one_Mpc,
        r_s_Mpc=halo.r_s / one_Mpc,
        R_vir_Mpc=halo.R_200 / one_Mpc,   # alias for analysis script
        n_steps=n_steps,
        dt_init=dt_init,
        Sigma_cr=Sigma_cr,
        Sigma_cr_comoving=Sigma_cr_comoving,
        Sigma_cr_physical=Sigma_cr_physical,
        lensing_reference_convention=DEFAULT_LENSING_REFERENCE_CONVENTION,
        D_l_Mpc=D_l / one_Mpc,
        D_s_Mpc=D_s_actual / one_Mpc,
        D_ls_Mpc=D_ls_actual / one_Mpc,
        sigma_kms=0.0,   # not used for NFW
        # Cosmology & affine distance (for D_A comparison)
        lambda_S=lambda_S,
        H0_kms_Mpc=H0,
        Omega_m=Omega_m,
        Omega_lambda=Omega_lambda,
    )
    print(f"\n   [ok] Saved to {outfile}")

    # =================================================================
    # 10. SUMMARY
    # =================================================================
    total = time.time() - t_total
    k_bg = k_prof[-1]
    dk_prof = np.abs(k_prof - k_bg)

    # Ratio dkappa_num / kappa_analytic
    mask_inside = (b_prof > 0.05) & (b_prof < R200_Mpc)
    if mask_inside.any() and k_analytic[mask_inside].max() > 0:
        ratio = dk_prof[mask_inside] / k_analytic[mask_inside]
        ratio = ratio[np.isfinite(ratio) & (k_analytic[mask_inside] > 0)]
        if len(ratio) > 0:
            ratio_mean = ratio.mean()
            ratio_std = ratio.std()
        else:
            ratio_mean = ratio_std = float("nan")
    else:
        ratio_mean = ratio_std = float("nan")

    print("\n" + "=" * 70)
    print("  NFW LENSING CONE  --  SUMMARY")
    print("=" * 70)
    print(f"  Grid             : {N}^3, {box_Mpc:.0f} Mpc (dx = {dx_Mpc*1000:.1f} kpc)")
    print(f"  Halo             : {halo}")
    print(f"  Integrator       : RK4 x {n_steps} steps")
    print(f"  D_l  -> D_s        : {D_l/one_Mpc:.2f}  -> {D_s_actual/one_Mpc:.2f} Mpc")
    print(f"  Sigma_cr_ref ({reference_label}) : {Sigma_cr_Mpc2:.3e} Msun/Mpc^2")
    print(f"  Sigma_cr_physical      : {Sigma_cr_physical_Mpc2:.3e} Msun/Mpc^2")
    print(f"  kappa_bg (far field) : {k_bg:.6e}")
    print(f"  dkappa range         : [{dk_prof.min():.3e}, {dk_prof.max():.3e}]")
    print(f"  kappa_analytic max ({reference_label}) : {k_analytic.max():.3e}")
    print(f"  |gamma| range        : [{g_prof.min():.3e}, {g_prof.max():.3e}]")
    print(f"  dkappa/kappa_analytic ({reference_label}) : {ratio_mean:.4f} +/- {ratio_std:.4f}")
    print(f"  Total time       : {total:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
