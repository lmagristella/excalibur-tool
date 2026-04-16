#!/usr/bin/env python3
"""
NFW lensing simulation with AMR (Adaptive Mesh Refinement).

The AMR patches are generated automatically where |grad Phi| is large,
giving sub-kpc resolution near the halo core at a fraction of the memory
cost of a uniform high-res grid.
"""

import os, sys, time
import numpy as np
from scipy import interpolate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import c, G, one_Mpc, one_Msun, one_Gpc, one_Myr, one_Gyr
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.amr_grid import AMRGrid, AMRInterpolator
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from excalibur.objects.nfw_halo import NFWHalo
from excalibur.io.filename_utils import RunNamer


def make_photon(obs_pos, target, metric, eta_0, a_0):
    """Create a Photon with null-condition k^mu and Sachs basis."""
    obs_4d = np.array([eta_0, *obs_pos])
    direction = target - obs_pos
    direction /= np.linalg.norm(direction)

    g = metric.metric_tensor(obs_4d)
    k_spatial = direction * c
    spatial_sq = (g[1, 1] * k_spatial[0]**2
                + g[2, 2] * k_spatial[1]**2
                + g[3, 3] * k_spatial[2]**2)
    k0 = -np.sqrt(abs(-spatial_sq / g[0, 0]))
    k_mu = np.array([k0, *k_spatial])

    e1, e2 = init_sachs_basis(k_mu, g, a_0)

    p = Photon(obs_4d.copy(), k_mu.copy())
    p.e1     = e1.copy()
    p.e2     = e2.copy()
    p.D_flat = np.array([0.0, 0.0, 0.0, 0.0])
    p.P_flat = np.array([1.0, 0.0, 0.0, 1.0])
    return p


def main():
    t_total = time.time()

    # =================================================================
    # 1. COSMOLOGY
    # =================================================================
    print("=" * 70)
    print("  NFW LENSING  --  AMR SIMULATION")
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
    print(f"   eta_0 = {eta_0:.4e} s,  a(eta_0) = {a_0:.6f}")

    # =================================================================
    # 2. ROOT GRID + NFW HALO
    # =================================================================
    print("2. Root grid + NFW halo ...")
    N_root    = 256
    box_Mpc   = 1000.0
    grid_size = box_Mpc * one_Mpc

    root_grid = Grid(
        shape   = (N_root, N_root, N_root),
        spacing = (grid_size / N_root,) * 3,
        origin  = np.array([0.0, 0.0, 0.0]),
    )

    M_200 = 1e15 * one_Msun
    c_NFW = 5.0
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    halo = NFWHalo(M_200, c_NFW, center)

    dx_root_Mpc = (grid_size / N_root) / one_Mpc
    R200_Mpc = halo.R_200 / one_Mpc
    rs_Mpc = halo.r_s / one_Mpc

    print(f"   {halo}")
    print(f"   R_200 = {R200_Mpc:.3f} Mpc,  r_s = {rs_Mpc*1000:.0f} kpc")
    print(f"   Root grid {N_root}^3,  box = {box_Mpc:.0f} Mpc")
    print(f"   dx_root = {dx_root_Mpc*1000:.1f} kpc,  r_s / dx_root = {halo.r_s / (grid_size/N_root):.1f}")
    print(f"   Root memory: {N_root**3 * 8 / 1e9:.2f} GB")

    # Fill root grid with NFW potential
    print("   Computing root grid potential ...")
    t_grid = time.time()
    x1d = np.linspace(0, grid_size, N_root)
    Y1d, Z1d = np.meshgrid(x1d, x1d, indexing="ij")
    phi_root = np.empty((N_root, N_root, N_root), dtype=np.float64)
    for ix in range(N_root):
        X_slice = np.full_like(Y1d, x1d[ix])
        phi_root[ix, :, :] = halo.potential(X_slice, Y1d, Z1d)
        if (ix + 1) % 64 == 0:
            print(f"     root slice {ix+1}/{N_root}")
    del Y1d, Z1d
    root_grid.add_field("Phi", phi_root)
    print(f"   [ok] Root potential in {time.time()-t_grid:.1f}s")

    phi_max = np.max(np.abs(phi_root))
    print(f"   |Phi|_max = {phi_max:.3e}  =>  Phi/c^2 = {phi_max/c**2:.3e}")

    # =================================================================
    # 3. AMR REFINEMENT
    # =================================================================
    print("\n3. AMR refinement ...")

    def phi_fn(x, y, z):
        """Exact NFW potential at arbitrary points."""
        return halo.potential(x, y, z)

    t_amr = time.time()
    amr = AMRGrid.from_field(
        root_grid, "Phi", phi_fn,
        max_level      = 5,
        ratio          = 4,         # 4x per level 
        refine_threshold = 0.005,   # refine where |grad phi|*dx/|phi_max| > 0.5%
        refine_mode    = "gradient",
        min_patch_cells = 64,
        boundary       = "clamp",
        scheme         = "tricubic",
        verbose        = True,
    )
    print(f"   [ok] AMR hierarchy built in {time.time()-t_amr:.1f}s")
    print(f"   {amr}")

    # Effective resolution at finest level
    if amr.patches:
        finest = max(amr.patches, key=lambda p: p.level)
        dx_finest_kpc = finest.spacing[0] / 3.0857e19
        print(f"   Finest level: dx = {dx_finest_kpc:.1f} kpc,  "
              f"r_s / dx_finest = {halo.r_s / finest.spacing[0]:.1f}")

    # =================================================================
    # 4. METRIC (using AMR interpolator)
    # =================================================================
    print("\n4. Metric ...")
    amr_interp = AMRInterpolator(amr, boundary="clamp", scheme="tricubic")

    metric = PerturbedFLRWMetricFast(
        a_of_eta       = a_of_eta,
        grid           = root_grid,       # root grid for shape/spacing compat
        interpolator   = amr_interp,      # AMR interpolator (drop-in)
        adot_of_eta    = cosmo.adot_of_eta,
        cosmology      = cosmo,
        enable_lensing = True,
        slow_roll      = False,
    )
    print("   [ok] metric ready (AMR interpolator)")

    # =================================================================
    # 5. PHOTON CONE
    # =================================================================
    print("5. Photon cone ...")

    obs_z_Mpc = 5.0
    obs_pos = np.array([box_Mpc/2, box_Mpc/2, obs_z_Mpc]) * one_Mpc

    dir_to_center = center - obs_pos
    D_l = np.linalg.norm(dir_to_center)
    dir_hat = dir_to_center / D_l

    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)
    e_perp2 = np.cross(dir_hat, e_perp1)
    e_perp2 /= np.linalg.norm(e_perp2)

    # Impact parameter grid  --  same as uniform-grid simulation
    b_values_Mpc = np.unique(np.sort(np.concatenate([
        np.array([0.0]),
        np.linspace(0.05, 0.4, 8),
        np.linspace(0.5, rs_Mpc, 8),
        np.linspace(rs_Mpc + 0.1, R200_Mpc, 12),
        np.linspace(R200_Mpc + 0.2, 5.0, 8),
        np.linspace(6.0, 15.0, 5),
        np.logspace(np.log10(0.05), np.log10(15.0), 25),
    ])))
    b_values = b_values_Mpc * one_Mpc

    photons_profile = []
    b_profile_Mpc = []
    for b in b_values:
        target = center + b * e_perp1
        p = make_photon(obs_pos, target, metric, eta_0, a_0)
        photons_profile.append(p)
        b_profile_Mpc.append(b / one_Mpc)
    n_profile = len(photons_profile)

    # 2D map
    map_half_Mpc = 5.0
    n_map_1d = 31
    b1_arr = np.linspace(-map_half_Mpc, map_half_Mpc, n_map_1d) * one_Mpc
    b2_arr = np.linspace(-map_half_Mpc, map_half_Mpc, n_map_1d) * one_Mpc

    photons_map = []
    map_b1_Mpc, map_b2_Mpc = [], []
    for b1 in b1_arr:
        for b2 in b2_arr:
            target = center + b1 * e_perp1 + b2 * e_perp2
            p = make_photon(obs_pos, target, metric, eta_0, a_0)
            photons_map.append(p)
            map_b1_Mpc.append(b1 / one_Mpc)
            map_b2_Mpc.append(b2 / one_Mpc)

    n_map = len(photons_map)
    n_total = n_profile + n_map
    print(f"   D_l (obs -> halo) = {D_l/one_Mpc:.1f} Mpc")
    print(f"   Profile photons : {n_profile}")
    print(f"   Map photons     : {n_map}  ({n_map_1d}x{n_map_1d})")
    print(f"   Total           : {n_total}")

    # =================================================================
    # 6. INTEGRATOR
    # =================================================================
    print("6. Integrator ...")
    # Use the finest AMR spacing for step size
    if amr.patches:
        dx_finest = min(p.spacing[0] for p in amr.patches)
    else:
        dx_finest = root_grid.spacing[0]
    dt_init = dx_finest / (5.0 * c)

    D_s  = 2.0 * D_l
    max_dist_in_box = grid_size - np.min(obs_pos)
    D_s  = min(D_s, 0.95 * max_dist_in_box)
    step_length = c * dt_init
    n_steps = int(np.ceil(D_s / step_length))
    D_s_Mpc = D_s / one_Mpc
    D_ls = D_s - D_l

    from scipy.optimize import brentq
    try:
        z_source = brentq(lambda z: cosmo.comoving_distance(z) - D_s, 0.0, 5.0)
    except ValueError:
        z_source = 0.05
    DA_FLRW = cosmo.angular_diameter_distance(z_source)

    print(f"   D_l  = {D_l/one_Mpc:.1f} Mpc")
    print(f"   D_s  = {D_s_Mpc:.1f} Mpc")
    print(f"   D_ls = {D_ls/one_Mpc:.1f} Mpc")
    print(f"   z_s  ~ {z_source:.4f}")
    print(f"   D_A^FLRW = {DA_FLRW/one_Mpc:.1f} Mpc")
    print(f"   dt   = {dt_init:.3e} s  =>  step = {step_length/one_Mpc*1000:.1f} kpc")
    print(f"   n_steps = {n_steps}")
    print(f"   r_s / step = {halo.r_s/step_length:.1f}")

    # If n_steps is very large due to tiny dt, use adaptive stepping
    # by increasing dt outside refined regions. For now use fixed dt.
    # Cap n_steps at a reasonable maximum.
    if n_steps > 50000:
        print(f"   WARNING: n_steps = {n_steps} is very large (fine AMR dt).")
        print(f"   Increasing dt to root-grid based stepping...")
        dt_init = root_grid.spacing[0] / (5.0 * c)
        step_length = c * dt_init
        n_steps = int(np.ceil(D_s / step_length))
        print(f"   New: dt = {dt_init:.3e} s,  step = {step_length/one_Mpc*1000:.1f} kpc,  n_steps = {n_steps}")
        print(f"dt in Myr: {dt_init/one_Myr:.3e} Myr,  step in kpc: {step_length/one_Mpc*1000:.1f} kpc")

    integrator = Integrator(
        metric     = metric,
        dt         = dt_init,
        mode       = "sequential",
        integrator = "rk4",
        rtol       = 1e-8,
        atol       = 1e-13,
    )

    # =================================================================
    # 7. INTEGRATE
    # =================================================================
    all_photons = photons_profile + photons_map
    lambda_S = n_steps * dt_init   # target affine parameter

    print(f"\n7. Integrating {n_total} photons (adaptive, lambda_target = {lambda_S:.4e} s) ...")
    print(f"   chi_S = c*lambda_S = {c*lambda_S/one_Mpc:.1f} Mpc")
    t_int = time.time()

    kappas  = np.empty(n_total)
    mus     = np.empty(n_total)
    gammas  = np.empty(n_total)
    D_flats = np.empty((n_total, 4))
    final_pos = np.empty((n_total, 3))
    lambda_actuals = np.empty(n_total)

    for idx, photon in enumerate(all_photons):
        integrator.integrate_single(
            photon,
            stop_mode    = "affine",
            stop_value   = lambda_S,
            record_every = 0,          # no intermediate recording (speed!)
        )
        # Normalize D by the actual affine parameter traversed
        lam_actual = photon.lambda_affine
        D_norm = photon.D_flat / lam_actual
        kappa, mu, shear = lensing_from_jacobi(D_norm)
        kappas[idx] = kappa
        mus[idx]    = mu
        gammas[idx] = shear
        D_flats[idx] = D_norm
        final_pos[idx] = photon.x[1:4]
        lambda_actuals[idx] = lam_actual

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_int
            rate = (idx + 1) / elapsed
            eta_remaining = (n_total - idx - 1) / rate if rate > 0 else 0
            dk = kappa - kappas[0] if idx > 0 else 0.0
            print(f"   [{idx+1:4d}/{n_total}]  "
                  f"kappa = {kappa:+.6e}  dk = {dk:+.3e}  |gamma| = {shear:.3e}  "
                  f"({elapsed:.0f}s, ~{eta_remaining:.0f}s left)")

    dt_elapsed = time.time() - t_int
    print(f"  Done in {dt_elapsed:.1f} s  ({dt_elapsed/n_total:.2f} s/photon)")

    # =================================================================
    # 8. EXTRACT RESULTS
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
    # 9. NFW ANALYTIC  (Sigma_cr with effective lensing kernel)
    # =================================================================
    # Sigma_cr = c^2/(4 pi G) * chi_s / (chi_l * chi_ls) / (1+z_l)
    # which is equivalent to  c^2/(4 pi G) * chi_s / (DA_l * chi_ls).
    from scipy.optimize import brentq as _brentq
    z_l = _brentq(lambda z: cosmo.comoving_distance(z) - D_l, 0.0, 5.0)

    Sigma_cr = (c**2 / (4.0 * np.pi * G)) * D_s / (D_l * D_ls) / (1.0 + z_l)
    Sigma_cr_Mpc2 = Sigma_cr * one_Mpc**2 / one_Msun

    # Also keep angular-diameter distances for reference / D_A comparison
    DA_l  = cosmo.angular_diameter_distance(z_l)
    DA_s  = cosmo.angular_diameter_distance(z_source)
    DA_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_source)

    b_m = np.abs(b_prof) * one_Mpc
    b_m = np.maximum(b_m, 1e-3 * one_Mpc)
    k_analytic = halo.kappa_analytic(b_m, Sigma_cr)
    g_analytic = halo.gamma_analytic(b_m, Sigma_cr)

    # =================================================================
    # 10. SAVE
    # =================================================================
    namer = RunNamer(
        "lensing_nfw_amr",
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
        Ds=round(D_s / one_Mpc, 1),
        obs=(round(obs_pos[0]/one_Mpc, 1),
             round(obs_pos[1]/one_Mpc, 1),
             round(obs_pos[2]/one_Mpc, 1)),
        box_Mpc=box_Mpc,
        N=N_root,
        AMR=amr.max_level,
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
        D_flat_profile=D_flats[:n_profile],
        lambda_actual_profile=lambda_actuals[:n_profile],
        # Map
        b1_map_Mpc=b1_map,
        b2_map_Mpc=b2_map,
        kappa_map=k_map,
        gamma_map=g_map,
        mu_map=m_map,
        D_flat_map=D_flats[n_profile:],
        lambda_actual_map=lambda_actuals[n_profile:],
        n_map_1d=n_map_1d,
        map_half_Mpc=map_half_Mpc,
        # Halo params
        halo_type="NFW",
        N_grid=N_root,
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
        D_l_Mpc=D_l / one_Mpc,
        D_s_Mpc=D_s / one_Mpc,
        D_ls_Mpc=D_ls / one_Mpc,
        DA_l_Mpc=DA_l / one_Mpc,
        DA_s_Mpc=DA_s / one_Mpc,
        DA_ls_Mpc=DA_ls / one_Mpc,
        z_l=z_l,
        sigma_kms=0.0,
        # Cosmology & affine
        lambda_S=lambda_S,
        H0_kms_Mpc=H0,
        Omega_m=Omega_m,
        Omega_lambda=Omega_lambda,
        z_source=z_source,
        # AMR info
        amr_max_level=amr.max_level,
        amr_n_patches=len(amr.patches),
        grid_type="AMR",
    )
    print(f"\n   [ok] Saved to {outfile}")

    # =================================================================
    # 11. SUMMARY
    # =================================================================
    total = time.time() - t_total
    k_bg = k_prof[-1]
    dk_prof = np.abs(k_prof - k_bg)

    print("\n" + "=" * 70)
    print("  NFW LENSING  --  AMR  --  SUMMARY")
    print("=" * 70)
    print(f"  Halo     : NFW, M_200 = {M_200/one_Msun:.0e} Msun, c = {c_NFW:.0f}")
    print(f"             R_200 = {R200_Mpc:.3f} Mpc,  r_s = {rs_Mpc*1000:.0f} kpc")
    print(f"  Grid     : {N_root}^3 root + {len(amr.patches)} AMR patches")
    if amr.patches:
        finest = max(amr.patches, key=lambda p: p.level)
        print(f"  Finest dx: {finest.spacing[0]/3.0857e19:.1f} kpc  (r_s/dx = {halo.r_s/finest.spacing[0]:.0f})")
    print(f"  Geometry : chi_l = {D_l/one_Mpc:.1f},  chi_s = {D_s/one_Mpc:.1f},  "
          f"chi_ls = {D_ls/one_Mpc:.1f}  Mpc  (comoving)")
    print(f"           DA_l = {DA_l/one_Mpc:.1f},  DA_s = {DA_s/one_Mpc:.1f},  "
          f"DA_ls = {DA_ls/one_Mpc:.1f}  Mpc  (angular diam.)")
    print(f"  z_l ~ {z_l:.4f},  z_s ~ {z_source:.4f}")
    print(f"  D_A^FLRW = {DA_FLRW/one_Mpc:.1f} Mpc")
    print(f"  Sigma_cr = {Sigma_cr_Mpc2:.3e} Msun/Mpc^2  (comoving / (1+z_l))")
    print(f"  Photons  : {n_profile} profile + {n_map} map = {n_total}")
    print(f"  Steps    : {n_steps}")
    print(f"  kappa_bg = {k_bg:.6e}")
    print(f"  dk_max   = {dk_prof.max():.4e}  (analytic: {k_analytic.max():.4e})")
    print(f"  |gamma|_max  = {g_prof.max():.4e}  (analytic: {g_analytic.max():.4e})")
    print(f"  Total time: {total/60:.1f} min")
    print("=" * 70)

    # Background-subtracted ratio: dk / (kappa_NFW - kappa_NFW_bg)
    ka_bg = k_analytic[-1]   # analytic kappa at outermost photon (still has NFW signal)
    ka_corrected = np.maximum(k_analytic - ka_bg, 0.0)

    mask = (b_prof > 0.03) & (b_prof < R200_Mpc) & (ka_corrected > 0)
    if mask.any():
        ratio_k = dk_prof[mask] / ka_corrected[mask]
        ratio_k = ratio_k[np.isfinite(ratio_k)]
        if len(ratio_k) > 0:
            print(f"  dk_num / (kappa_NFW-kappa_bg) (b < R_200): {ratio_k.mean():.3f} +/- {ratio_k.std():.3f}")
    # Uncorrected for reference
    mask_raw = (b_prof > 0.03) & (b_prof < R200_Mpc) & (k_analytic > 0)
    if mask_raw.any():
        ratio_raw = dk_prof[mask_raw] / k_analytic[mask_raw]
        ratio_raw = ratio_raw[np.isfinite(ratio_raw)]
        if len(ratio_raw) > 0:
            print(f"  (uncorrected dk/kappa_NFW:            {ratio_raw.mean():.3f} +/- {ratio_raw.std():.3f})")

    mask_g = (b_prof > 0.05) & (b_prof < R200_Mpc) & (g_analytic > 0)
    if mask_g.any():
        ratio_g = g_prof[mask_g] / g_analytic[mask_g]
        ratio_g = ratio_g[np.isfinite(ratio_g)]
        if len(ratio_g) > 0:
            print(f"  |gamma|_num / gamma_NFW (b < R_200):       {ratio_g.mean():.3f} +/- {ratio_g.std():.3f})")



if __name__ == "__main__":
    main()
