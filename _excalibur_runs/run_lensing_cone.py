#!/usr/bin/env python3
r"""
Multi-photon weak-lensing simulation with excalibur.

Shoots a cone of photons at different impact parameters through a SIS-like
halo and records the lensing observables kappa(b), |gamma|(b) for every photon.

Improvements over run_lensing_simulation.py:
    - 256^3 grid  (dx ~ 2 Mpc  -> sharper potential)
    - 500 RK4 steps  (photons traverse ~ 2x more of the box)
    - Cone of ~60 photons at varying impact parameters
    - Saves all trajectories + observables to .npz for postprocessing
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
from excalibur.objects.spherical_mass import spherical_mass
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from excalibur.io.filename_utils import RunNamer


# =====================================================================
#  HELPER: build a photon aimed at (halo_center + offset), with Sachs IC
# =====================================================================
def make_photon(obs_pos, target, metric, eta_0, a_0):
    """
    Create a Photon at *obs_pos* aimed toward *target*, with
    null-condition-consistent k^mu and Sachs basis initialised.
    """
    obs_4d = np.array([eta_0, *obs_pos])
    direction = target - obs_pos
    direction /= np.linalg.norm(direction)

    g = metric.metric_tensor(obs_4d)
    k_spatial = direction * c
    spatial_sq = (g[1, 1] * k_spatial[0]**2
                + g[2, 2] * k_spatial[1]**2
                + g[3, 3] * k_spatial[2]**2)
    k0 = -np.sqrt(abs(-spatial_sq / g[0, 0]))   # backward tracing
    k_mu = np.array([k0, *k_spatial])

    e1, e2 = init_sachs_basis(-k_mu, g, a_0)

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
    print("  MULTI-PHOTON LENSING CONE SIMULATION")
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
    # 2. GRID + HALO
    # =================================================================
    print("2. Grid + halo ...")
    N         = 256
    box_Mpc   = 200.0                  # smaller box  -> finer dx
    grid_size = box_Mpc * one_Mpc

    grid = Grid(
        shape   = (N, N, N),
        spacing = (grid_size / N,) * 3,
        origin  = np.array([0.0, 0.0, 0.0]),
    )

    M     = 1e14 * one_Msun
    R_vir = 20.0 * one_Mpc            # ~26 cells across  -> well-resolved
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    halo = spherical_mass(M, R_vir, center)

    x1d = np.linspace(0, grid_size, N)
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    phi_field = halo.potential(X, Y, Z)
    grid.add_field("Phi", phi_field)

    phi_max = np.max(np.abs(phi_field))
    dx_Mpc = (grid_size / N) / one_Mpc
    print(f"   Grid {N}^3,  box = {box_Mpc:.0f} Mpc,  dx = {dx_Mpc:.2f} Mpc")
    print(f"   M = {M/one_Msun:.1e} Msun,  R_vir = {R_vir/one_Mpc:.1f} Mpc")
    print(f"   |Phi|_max = {phi_max:.3e}   ->  Phi/c^2 = {phi_max/c**2:.3e}")

    # =================================================================
    # 3. METRIC (lensing ON, slow-roll ON)
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
    )
    print("   [ok] metric ready")

    # =================================================================
    # 4. PHOTON CONE  --  varying impact parameter
    # =================================================================
    print("4. Photon cone ...")

    # Observer near the edge of the box, at the start of the z-axis
    obs_pos = np.array([100.0, 100.0, 5.0]) * one_Mpc   # near bottom face
    # This puts the observer ~95 Mpc from the halo (center at 100 Mpc)

    # Baseline direction toward halo center
    dir_to_center = center - obs_pos
    dist_to_center = np.linalg.norm(dir_to_center)
    dir_hat = dir_to_center / dist_to_center

    # Build two orthonormal vectors perp to dir_hat (for impact parameter offsets)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)
    e_perp2 = np.cross(dir_hat, e_perp1)
    e_perp2 /= np.linalg.norm(e_perp2)

    # Impact parameters from 0.5 Mpc to 80 Mpc (well beyond R_vir = 20 Mpc)
    b_values_Mpc = np.unique(np.sort(np.concatenate([
        np.array([0.0]),                                  # on-axis
        np.linspace(0.5, 5, 10),                          # inner (well inside R_vir)
        np.linspace(6, 20, 15),                           # near R_vir boundary
        np.linspace(22, 50, 10),                           # 1-2.5 R_vir
        np.linspace(55, 80, 6),                            # far field
        np.logspace(np.log10(0.5), np.log10(80), 25),    # log-spaced fill
    ])))
    b_values = b_values_Mpc * one_Mpc

    # For each b, create photons offset in both perp directions,
    # plus the on-axis photon, and photons in a 2D ring pattern
    # We'll use a 1D radial scan along e_perp1 for the profile,
    # plus a 2D grid for the maps.

    # --- 1D radial profile photons ---
    photons_profile = []
    b_profile_Mpc = []
    for b in b_values:
        target = center + b * e_perp1
        p = make_photon(obs_pos, target, metric, eta_0, a_0)
        photons_profile.append(p)
        b_profile_Mpc.append(b / one_Mpc)

    n_profile = len(photons_profile)

    # --- 2D map photons (grid of impact parameters) ---
    map_half_Mpc = 60.0   # half-size of map in Mpc (covers ~3 R_vir)
    n_map_1d = 25          # 25x25 = 625 photons for the map
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
    print(f"   Profile photons : {n_profile}  (b = 0 .. {b_values_Mpc[-1]:.1f} Mpc)")
    print(f"   Map photons     : {n_map}  ({n_map_1d}x{n_map_1d} grid, +/-{map_half_Mpc} Mpc)")
    print(f"   Total photons   : {n_total}")

    # =================================================================
    # 5. INTEGRATOR  --  source at D_s ~ 2 x D_l (thin-lens geometry)
    # =================================================================
    print("5. Integrator ...")
    # With k^0 < 0 (time-reversed photon), dlambda must be POSITIVE so that
    #   deta = k^0 dlambda < 0  (backward in conformal time)
    #   dx^i = k^i dlambda > 0  (photon propagates toward the halo)
    dt_init = grid.spacing[0] / (5.0 * c)    # positive affine step

    # -- Source distance from thin-lens geometry ----------------------
    # D_l = distance observer  -> lens centre
    # D_s = 2 x D_l   (source plane behind the halo at equal distance)
    # Also cap so photon stays inside the box (grid_size along z-axis).
    D_l  = np.linalg.norm(center - obs_pos)           # ~95 Mpc
    D_s  = 2.0 * D_l                                  # ~190 Mpc
    # Safety: the photon must not exit the box.  The observer is at
    # z = 5 Mpc, so the max usable distance along z is ~195 Mpc.
    max_dist_in_box = grid_size - np.min(obs_pos)      # conservative
    D_s  = min(D_s, 0.95 * max_dist_in_box)            # 5 % margin
    step_length = c * dt_init                           # ~0.156 Mpc per step
    n_steps = int(np.ceil(D_s / step_length))
    D_s_Mpc = D_s / one_Mpc

    print(f"   D_l  = {D_l/one_Mpc:.1f} Mpc  (observer  -> halo)")
    print(f"   D_s  = {D_s_Mpc:.1f} Mpc  (source plane, 2xD_l capped to box)")
    print(f"   step = {step_length/one_Mpc:.3f} Mpc   ->  n_steps = {n_steps}")

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
    print(f"   dt = {dt_init:.3e} s,  n_steps = {n_steps}")

    # =================================================================
    # 6. INTEGRATE   --   profile photons first, then map photons
    # =================================================================
    all_photons = photons_profile + photons_map
    labels = (["profile"] * n_profile) + (["map"] * n_map)

    # Affine parameter at the source: lambda_S = n_steps x dt
    # With Jacobi IC D(0)=0, P(0)=I, the unlensed beam gives D(lambda)=lambda*I.
    # We normalise D by lambda_S so that D_norm = D/lambda_S  -> I in flat space,
    # and then  kappa = 1 - 1/2 tr(D_norm)  matches the standard thin-lens Sigma/Sigma_cr.
    lambda_S = n_steps * dt_init

    print(f"\n6. Integrating {n_total} photons x {n_steps} steps ...")
    print(f"   lambda_S = {lambda_S:.6e} s  (affine distance to source)")
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
        # Normalise D by lambda_S so unlensed beam  -> identity
        D_norm = photon.D_flat / lambda_S
        # Read lensing observables from normalised Jacobi map
        kappa, mu, shear = lensing_from_jacobi(D_norm)
        kappas[idx] = kappa
        mus[idx]    = mu
        gammas[idx] = shear
        D_flats[idx] = D_norm
        final_pos[idx] = photon.x[1:4]

        # Progress report every 50 photons
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_int
            rate = (idx + 1) / elapsed
            eta_remaining = (n_total - idx - 1) / rate if rate > 0 else 0
            dk = kappa - kappas[0] if idx > 0 else 0.0
            print(f"   [{idx+1:4d}/{n_total}]  "
                  f"kappa = {kappa:+.6e}  dkappa = {dk:+.3e}  |gamma| = {shear:.3e}  "
                  f"({elapsed:.0f}s elapsed, ~{eta_remaining:.0f}s remaining)")

    dt_elapsed = time.time() - t_int
    print(f"   [ok] All {n_total} photons done in {dt_elapsed:.1f} s  "
          f"({dt_elapsed/n_total:.2f} s/photon)")

    # =================================================================
    # 7. EXTRACT RESULTS
    # =================================================================
    # Profile arrays
    b_prof  = np.array(b_profile_Mpc)
    k_prof  = kappas[:n_profile]
    g_prof  = gammas[:n_profile]
    m_prof  = mus[:n_profile]

    # Map arrays
    b1_map = np.array(map_b1_Mpc)
    b2_map = np.array(map_b2_Mpc)
    k_map  = kappas[n_profile:]
    g_map  = gammas[n_profile:]
    m_map  = mus[n_profile:]

    # =================================================================
    # 8. SIS ANALYTIC PREDICTIONS
    # =================================================================
    # For a uniform-density sphere (not strictly SIS, but the potential
    # is -GM/r outside R_vir), the projected convergence at impact
    # parameter b is related to the surface mass density Sigma(b):
    #
    #   kappa_analytic(b) = Sigma(b) / Sigma_cr
    #
    # For a uniform sphere seen in projection:
    #   Sigma(b) = 2 rho0 sqrt(R^2 - b^2)  for b < R,   0 for b >= R
    #
    # We use the thin-lens geometry set in Section 5:
    #   D_l  = observer  -> halo centre  (already computed)
    #   D_s  = 2 x D_l  (capped to box)  --  also already computed
    #   D_ls = D_s - D_l

    # Velocity dispersion for SIS-equivalent:
    # sigma^2 = GM / (2 R_vir)  (virial theorem rough estimate)
    sigma2 = G * M / (2.0 * R_vir)
    sigma = np.sqrt(sigma2)
    sigma_kms = sigma / 1e3
    print(f"\n   Effective sigma ~ {sigma_kms:.0f} km/s")

    # Use the actual mean final z-position of the profile photons to
    # derive D_s self-consistently (in case the integrator overshoots
    # or undershoots slightly compared to the n_steps estimate).
    D_s_actual = np.mean(np.linalg.norm(final_pos - obs_pos, axis=1))
    D_ls_actual = D_s_actual - D_l
    if D_ls_actual <= 0:
        D_ls_actual = D_l   # fallback: symmetric lens

    print(f"   D_l  = {D_l/one_Mpc:.1f} Mpc")
    print(f"   D_s  = {D_s_actual/one_Mpc:.1f} Mpc  (actual mean final distance)")
    print(f"   D_ls = {D_ls_actual/one_Mpc:.1f} Mpc")

    # Critical surface density:
    # Effective Sigma_cr matching excalibur's Sachs conventions (see note in
    # run_lensing_nfw_amr.py): divide comoving Sigma_cr by (1+z_l).
    from scipy.optimize import brentq as _brentq
    z_l = _brentq(lambda z: cosmo.comoving_distance(z) - D_l, 0.0, 5.0)

    Sigma_cr = (c**2 / (4.0 * np.pi * G)) * D_s_actual / (D_l * D_ls_actual) / (1.0 + z_l)
    Sigma_cr_Mpc2 = Sigma_cr * one_Mpc**2 / one_Msun  # in Msun/Mpc^2
    print(f"   Sigma_cr ~ {Sigma_cr_Mpc2:.3e} Msun/Mpc^2")

    # Analytic kappa for the uniform sphere:
    def kappa_sphere(b_Mpc_arr):
        """Projected convergence of a uniform sphere of mass M, radius R."""
        R = R_vir
        rho0 = M / (4.0/3.0 * np.pi * R**3)
        kappa_arr = np.zeros_like(b_Mpc_arr)
        for i, bm in enumerate(b_Mpc_arr):
            b_si = abs(bm) * one_Mpc
            if b_si < R and b_si > 0:
                Sigma_b = 2.0 * rho0 * np.sqrt(R**2 - b_si**2)
                kappa_arr[i] = Sigma_b / Sigma_cr
            elif b_si == 0:
                Sigma_b = 2.0 * rho0 * R
                kappa_arr[i] = Sigma_b / Sigma_cr
        return kappa_arr

    k_analytic = kappa_sphere(b_prof)

    # =================================================================
    # 9. SAVE ALL RESULTS
    # =================================================================
    namer = RunNamer(
        "lensing_cone",
        integrator="rk4",
        metric="FLRWP1",
        profile="sphere",
        M=M / one_Msun,
        Rvir=round(R_vir / one_Mpc, 1),
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
        # Metadata
        N_grid=N,
        box_Mpc=box_Mpc,
        M_Msun=M / one_Msun,
        R_vir_Mpc=R_vir / one_Mpc,
        n_steps=n_steps,
        sigma_kms=sigma_kms,
        Sigma_cr=Sigma_cr,
        D_l_Mpc=D_l / one_Mpc,
        D_s_Mpc=D_s_actual / one_Mpc,
        D_ls_Mpc=D_ls_actual / one_Mpc,
        # Cosmology & affine distance (for D_A comparison)
        lambda_S=lambda_S,
        H0_kms_Mpc=H0,
        Omega_m=Omega_m,
        Omega_lambda=Omega_lambda,
    )
    print(f"\n   [ok] Results saved to {outfile}")

    # =================================================================
    # 10. QUICK SUMMARY
    # =================================================================
    total = time.time() - t_total

    # Background-subtracted convergence (subtract far-field value)
    k_bg = k_prof[-1]   # outermost photon ~ background
    dk_prof = k_prof - k_bg

    print("\n" + "=" * 70)
    print("  LENSING CONE SIMULATION  --  SUMMARY")
    print("=" * 70)
    print(f"  Grid             : {N}^3, {box_Mpc:.0f} Mpc (dx = {dx_Mpc:.2f} Mpc)")
    print(f"  Halo             : {M/one_Msun:.1e} Msun, R_vir = {R_vir/one_Mpc:.1f} Mpc")
    print(f"  Integrator       : RK4 x {n_steps} steps, slow_roll=True")
    print(f"  D_l  -> D_s        : {D_l/one_Mpc:.1f}  -> {D_s_actual/one_Mpc:.1f} Mpc  (2xD_l geometry)")
    print(f"  Profile photons  : {n_profile}")
    print(f"  Map photons      : {n_map}")
    print(f"  kappa_bg (far field) : {k_bg:.6e}")
    print(f"  dkappa range (halo)  : [{dk_prof.min():.3e}, {dk_prof.max():.3e}]")
    print(f"  |gamma| range        : [{g_prof.min():.3e}, {g_prof.max():.3e}]")
    print(f"  Total time       : {total:.1f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
