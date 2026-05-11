#!/usr/bin/env python3
r"""
NFW lensing simulation on a 200 Mpc box with N=1024.

This gives:
  - z_source ~ 0.045  (real cosmological distance)
  - D_A ~ 182 Mpc     (non-trivial angular diameter distance)
  - dx ~ 195 kpc      (r_s / dx ~ 2.1 for M_200=1e15, c=5)

Memory requirement: ~9 GB for the potential field (float64, 1024^3).
Runtime estimate:  ~1-2 hours on 14 cores (sequential photon loop).
"""

import os, sys, time
import numpy as np
from scipy import interpolate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Excalibur imports
from excalibur.core.constants import c, G, one_Mpc, one_Msun, one_Gpc
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_4d_fast import InterpolatorFast as Interpolator4DFast
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.observables.lensing_conventions import (
    DEFAULT_LENSING_REFERENCE_CONVENTION,
    lensing_convention_label,
    sigma_cr_conventions,
)
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from excalibur.objects.nfw_halo import NFWHalo
from excalibur.io.filename_utils import RunNamer


def make_photon(obs_pos, target, metric, eta_0, a_0):
    """Create a Photon with null-condition k^mu and Sachs basis."""
    obs_4d = np.array([eta_0, *obs_pos])
    direction = target - obs_pos
    direction /= np.linalg.norm(direction)
    screen_convention = getattr(metric, "sachs_screen_convention", "metric")

    g = metric.metric_tensor(obs_4d)
    k_spatial = direction * c
    spatial_sq = (g[1, 1] * k_spatial[0]**2
                + g[2, 2] * k_spatial[1]**2
                + g[3, 3] * k_spatial[2]**2)
    k0 = -np.sqrt(abs(-spatial_sq / g[0, 0]))   # backward tracing
    k_mu = np.array([k0, *k_spatial])

    e1, e2 = init_sachs_basis(-k_mu, g, a_0, convention=screen_convention)

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
    print("  NFW LENSING  --  COSMOLOGICAL BOX (200 Mpc, N=1024)")
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
    N         = 1024
    box_Mpc   = 200.0
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

    dx_Mpc = (grid_size / N) / one_Mpc
    R200_Mpc = halo.R_200 / one_Mpc
    rs_Mpc = halo.r_s / one_Mpc

    print(f"   {halo}")
    print(f"   R_200 = {R200_Mpc:.3f} Mpc,  r_s = {rs_Mpc*1000:.0f} kpc")
    print(f"   Grid {N}^3,  box = {box_Mpc:.0f} Mpc,  dx = {dx_Mpc*1000:.1f} kpc")
    print(f"   r_s / dx = {halo.r_s / (grid_size/N):.1f}")
    print(f"   Memory for potential field: {N**3 * 8 / 1e9:.2f} GB")

    # Build the potential field on the grid
    print("   Computing NFW potential on grid (this may take a few minutes) ...")
    t_grid = time.time()
    x1d = np.linspace(0, grid_size, N)
    # Use chunks to avoid allocating three N^3 arrays at once (saves ~25 GB)
    phi_field = np.empty((N, N, N), dtype=np.float64)
    Y1d, Z1d = np.meshgrid(x1d, x1d, indexing="ij")
    for ix in range(N):
        X_slice = np.full_like(Y1d, x1d[ix])
        phi_field[ix, :, :] = halo.potential(X_slice, Y1d, Z1d)
        if (ix + 1) % 128 == 0:
            elapsed = time.time() - t_grid
            frac = (ix + 1) / N
            eta_s = elapsed / frac * (1 - frac)
            print(f"     slice {ix+1}/{N}  ({elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining)")
    del Y1d, Z1d

    grid.add_field("Phi", phi_field)
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
    obs_z_Mpc = 5.0    # 5 Mpc inside the box
    obs_pos = np.array([box_Mpc/2, box_Mpc/2, obs_z_Mpc]) * one_Mpc

    # Halo is at box center  -> D_l ~ 95 Mpc
    dir_to_center = center - obs_pos
    D_l = np.linalg.norm(dir_to_center)
    dir_hat = dir_to_center / D_l

    # Perpendicular basis for impact parameters
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)
    e_perp2 = np.cross(dir_hat, e_perp1)
    e_perp2 /= np.linalg.norm(e_perp2)

    # Impact parameters (Mpc): fine in the core, logarithmic spread outside
    b_values_Mpc = np.unique(np.sort(np.concatenate([
        np.array([0.0]),
        np.linspace(0.05, 0.4, 8),                  # inner core
        np.linspace(0.5, rs_Mpc, 8),                # up to r_s
        np.linspace(rs_Mpc + 0.1, R200_Mpc, 12),   # r_s to R_200
        np.linspace(R200_Mpc + 0.2, 5.0, 8),        # beyond R_200
        np.linspace(6.0, 15.0, 5),                   # far field
        np.logspace(np.log10(0.05), np.log10(15.0), 25),
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
    map_half_Mpc = 5.0      # +/-5 Mpc (covers ~2.4 R_200)
    n_map_1d = 31            # 31x31 = 961 photons
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
    print(f"   D_l (obs  -> halo) = {D_l/one_Mpc:.1f} Mpc")
    print(f"   Profile photons : {n_profile}  (b = 0 .. {b_values_Mpc[-1]:.1f} Mpc)")
    print(f"   Map photons     : {n_map}  ({n_map_1d}x{n_map_1d} grid, +/-{map_half_Mpc} Mpc)")
    print(f"   Total photons   : {n_total}")

    # =================================================================
    # 5. INTEGRATOR
    # =================================================================
    print("5. Integrator ...")
    dt_init = grid.spacing[0] / (5.0 * c)    # positive affine step

    # Source plane: behind halo, near back of box
    D_s  = 2.0 * D_l
    max_dist_in_box = grid_size - np.min(obs_pos)
    D_s  = min(D_s, 0.95 * max_dist_in_box)
    step_length = c * dt_init
    n_steps = int(np.ceil(D_s / step_length))

    D_s_Mpc = D_s / one_Mpc
    D_ls = D_s - D_l

    # Compute source redshift
    from scipy.optimize import brentq
    try:
        z_source = brentq(lambda z: cosmo.comoving_distance(z) - D_s, 0.0, 5.0)
    except ValueError:
        z_source = 0.05
    DA_FLRW = cosmo.angular_diameter_distance(z_source)

    print(f"   D_l  = {D_l/one_Mpc:.1f} Mpc  (observer  -> halo)")
    print(f"   D_s  = {D_s_Mpc:.1f} Mpc  (source plane)")
    print(f"   D_ls = {D_ls/one_Mpc:.1f} Mpc")
    print(f"   z_s  ~ {z_source:.4f}")
    print(f"   D_A^FLRW(z_s) = {DA_FLRW/one_Mpc:.1f} Mpc")
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
    print(f"   lambda_S = {lambda_S:.6e} s   (chi_S = c*lambda_S = {c*lambda_S/one_Mpc:.1f} Mpc)")
    t_int = time.time()

    kappas  = np.empty(n_total)
    mus     = np.empty(n_total)
    gammas  = np.empty(n_total)
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

    b_m = np.abs(b_prof) * one_Mpc
    b_m = np.maximum(b_m, 1e-3 * one_Mpc)
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
        "lensing_nfw_cosmo",
        integrator="rk4",
        metric="FLRWP1",
        profile="nfw",
        M=M_200 / one_Msun,
        c_NFW=c_NFW,
        Rvir=round(R200_Mpc, 3),
        rs=round(rs_Mpc, 4),
        zl=round(z_l, 5),
        zs=round(z_source, 5),
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
        R_200_Mpc=R200_Mpc,
        r_s_Mpc=rs_Mpc,
        R_vir_Mpc=R200_Mpc,
        n_steps=n_steps,
        dt_init=dt_init,
        Sigma_cr=Sigma_cr,
        Sigma_cr_comoving=Sigma_cr_comoving,
        Sigma_cr_physical=Sigma_cr_physical,
        lensing_reference_convention=DEFAULT_LENSING_REFERENCE_CONVENTION,
        D_l_Mpc=D_l / one_Mpc,
        D_s_Mpc=D_s_actual / one_Mpc,
        D_ls_Mpc=D_ls_actual / one_Mpc,
        sigma_kms=0.0,
        # Cosmology & affine distance (for D_A comparison)
        lambda_S=lambda_S,
        H0_kms_Mpc=H0,
        Omega_m=Omega_m,
        Omega_lambda=Omega_lambda,
        z_source=z_source,
    )
    print(f"\n   [ok] Saved to {outfile}")

    # =================================================================
    # 10. SUMMARY
    # =================================================================
    total = time.time() - t_total
    k_bg = k_prof[-1]
    dk_prof = np.abs(k_prof - k_bg)

    print("\n" + "=" * 70)
    print("  NFW LENSING  --  COSMOLOGICAL BOX  --  SUMMARY")
    print("=" * 70)
    print(f"  Halo     : NFW, M_200 = {M_200/one_Msun:.0e} Msun, c = {c_NFW:.0f}")
    print(f"             R_200 = {R200_Mpc:.3f} Mpc,  r_s = {rs_Mpc*1000:.0f} kpc")
    print(f"  Grid     : {N}^3,  {box_Mpc:.0f} Mpc,  dx = {dx_Mpc*1000:.1f} kpc")
    print(f"  Geometry : D_l = {D_l/one_Mpc:.1f},  D_s = {D_s_actual/one_Mpc:.1f},  "
          f"D_ls = {D_ls_actual/one_Mpc:.1f}  Mpc")
    print(f"  z_source ~ {z_source:.4f}")
    print(f"  D_A^FLRW = {DA_FLRW/one_Mpc:.1f} Mpc")
    print(f"  Sigma_cr_ref ({reference_label}) = {Sigma_cr_Mpc2:.3e} Msun/Mpc^2")
    print(f"  Sigma_cr_physical      = {Sigma_cr_physical_Mpc2:.3e} Msun/Mpc^2")
    print(f"  Photons  : {n_profile} profile + {n_map} map = {n_total}")
    print(f"  Steps    : {n_steps}")
    print(f"  kappa_bg     = {k_bg:.6e}")
    print(f"  dkappa_max   = {dk_prof.max():.4e}  (analytic {reference_label}: {k_analytic.max():.4e})")
    print(f"  |gamma|_max  = {g_prof.max():.4e}  (analytic {reference_label}: {g_analytic.max():.4e})")
    print(f"  Total time: {total/60:.1f} min")
    print("=" * 70)

    # Quick ratio stats
    mask = (b_prof > 0.03) & (b_prof < R200_Mpc) & (k_analytic > 0)
    if mask.any():
        ratio = dk_prof[mask] / k_analytic[mask]
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio) > 0:
            print(f"  dkappa_num / kappa_NFW ({reference_label}, b < R_200): {ratio.mean():.3f} +/- {ratio.std():.3f}")

    print(f"\n  Run:  python analyze_lensing_nfw.py {outfile}")


if __name__ == "__main__":
    main()
