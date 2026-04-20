"""
End-to-end sanity test for the analytical bypass in a lensing context.

Builds a small NFW AMR grid, wraps the AMR interpolator with the
AnalyticalBypassInterpolator, and traces a handful of profile photons
at impact parameters that span:
    - inside the bypass sphere  (where we expect kappa > 1)
    - crossing the bypass boundary
    - outside the bypass (AMR-only, should reproduce prior behaviour)

Run directly:
    python _tests/test_nfw_bypass_lensing.py
"""

import os, sys, time
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import c, G, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.amr_grid import AMRGrid, AMRInterpolator
from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from excalibur.objects.nfw_halo import NFWHalo


def make_photon(obs_pos, target, metric, eta_0, a_0):
    obs_4d = np.array([eta_0, *obs_pos])
    direction = target - obs_pos
    direction /= np.linalg.norm(direction)
    g = metric.metric_tensor(obs_4d)
    k_spatial = direction * c
    spatial_sq = (g[1, 1] * k_spatial[0] ** 2
                  + g[2, 2] * k_spatial[1] ** 2
                  + g[3, 3] * k_spatial[2] ** 2)
    k0 = -np.sqrt(abs(-spatial_sq / g[0, 0]))
    k_mu = np.array([k0, *k_spatial])
    e1, e2 = init_sachs_basis(k_mu, g, a_0)
    p = Photon(obs_4d.copy(), k_mu.copy())
    p.e1 = e1.copy()
    p.e2 = e2.copy()
    p.D_flat = np.zeros(4)
    p.P_flat = np.array([1.0, 0.0, 0.0, 1.0])
    return p


def run_once(use_bypass, halo, bypass_mult, amr, root_grid, metric_builder,
             photons_factory):
    """Build metric (+optional bypass) and integrate a batch of photons."""
    amr_interp = AMRInterpolator(amr, boundary="clamp", scheme="tricubic")

    finest_spacing = min((p.spacing[0] for p in amr.patches),
                         default=root_grid.spacing[0])

    if use_bypass:
        interp = AnalyticalBypassInterpolator(
            base_interp=amr_interp,
            analytical_source=halo,
            bypass_radius=bypass_mult * finest_spacing,
        )
        label = f"bypass={bypass_mult}*dx_finest"
    else:
        interp = amr_interp
        label = "no-bypass"

    metric = metric_builder(interp)
    photons, b_list = photons_factory(metric)

    dt_fine = halo.r_s / (8.0 * c)
    integrator = Integrator(
        metric=metric, dt=dt_fine, mode="sequential",
        integrator="rk45", rtol=1e-8, atol=1e-13,
    )

    n = len(photons)
    kappas = np.empty(n)
    gammas = np.empty(n)
    mus = np.empty(n)
    lam_S = photons[0]._lam_S
    for i, p in enumerate(photons):
        integrator.integrate_single(
            p, stop_mode="affine", stop_value=lam_S, record_every=0,
        )
        D = p.D_flat / p.lambda_affine
        k, mu, sh = lensing_from_jacobi(D)
        kappas[i], gammas[i], mus[i] = k, sh, mu

    return label, np.array(b_list), kappas, gammas, mus, finest_spacing


def main():
    t0 = time.time()
    print("=" * 70)
    print("  NFW analytical-bypass lensing sanity test")
    print("=" * 70)

    # ---------- Cosmology ----------
    cosmo = LCDM_Cosmology(70.0, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0 = cosmo.a_of_eta(eta_0)
    eta_arr = np.linspace(0.5 * eta_0, eta_0, 2000)
    a_arr = np.array([cosmo.a_of_eta(e) for e in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic",
                                    fill_value="extrapolate")

    # ---------- Halo + root grid ----------
    # Sanity test, not production: small box concentrated around the halo
    # so we get usable resolution with a modest memory budget.
    # 96^3 root x 8 bytes = 7 MB; AMR adds ~a few small patches.
    N_root = 96
    box_Mpc = 400.0
    grid_size = box_Mpc * one_Mpc
    root_grid = Grid(
        shape=(N_root, N_root, N_root),
        spacing=(grid_size / N_root,) * 3,
        origin=np.array([0.0, 0.0, 0.0]),
    )

    M_200 = 2e15 * one_Msun
    c_NFW = 7.0
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    halo = NFWHalo(M_200, c_NFW, center)
    rs_Mpc = halo.r_s / one_Mpc
    R200_Mpc = halo.R_200 / one_Mpc
    print(f"  Halo: M={M_200/one_Msun:.1e} Msun, c={c_NFW}, "
          f"R_200={R200_Mpc:.2f} Mpc, r_s={rs_Mpc*1000:.0f} kpc")
    print(f"  Root grid: {N_root}^3, box={box_Mpc} Mpc, "
          f"dx_root={grid_size/N_root/one_Mpc*1000:.0f} kpc")

    # Fill potential
    x1d = np.linspace(0, grid_size, N_root)
    Y1d, Z1d = np.meshgrid(x1d, x1d, indexing="ij")
    phi_root = np.empty((N_root, N_root, N_root))
    for ix in range(N_root):
        X_slice = np.full_like(Y1d, x1d[ix])
        phi_root[ix, :, :] = halo.potential(X_slice, Y1d, Z1d)
    del Y1d, Z1d
    root_grid.add_field("Phi", phi_root)

    # AMR
    print("  Building AMR ...")
    # Modest AMR: the analytical bypass covers the cusp, so the AMR only
    # needs to give a reasonable baseline outside the bypass region.
    amr = AMRGrid.from_field(
        root_grid, "Phi",
        lambda x, y, z: halo.potential(x, y, z),
        max_level=3, ratio=2, refine_threshold=0.05,
        refine_mode="gradient", min_patch_cells=32,
        boundary="clamp", scheme="tricubic", verbose=False,
    )
    finest_spacing = min(p.spacing[0] for p in amr.patches)
    print(f"  AMR: {len(amr.patches)} patches, "
          f"dx_finest={finest_spacing/one_Mpc*1000:.1f} kpc "
          f"({finest_spacing/halo.r_s:.2f} r_s)")

    # ---------- Observer + source geometry ----------
    obs_pos = np.array([box_Mpc/2, box_Mpc/2, 5.0]) * one_Mpc
    D_l = float(np.linalg.norm(center - obs_pos))
    dir_hat = (center - obs_pos) / D_l
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)

    z_s = 1.0
    D_s = cosmo.comoving_distance(z_s)
    max_d = grid_size - np.min(obs_pos)
    D_s = min(D_s, 0.95 * max_d)
    D_ls = D_s - D_l
    z_l = brentq(lambda z: cosmo.comoving_distance(z) - D_l, 0.0, 5.0)
    Sigma_cr = (c ** 2 / (4.0 * np.pi * G)) * D_s / (D_l * D_ls) / (1.0 + z_l)
    print(f"  D_l={D_l/one_Mpc:.0f} Mpc, D_s={D_s/one_Mpc:.0f} Mpc, "
          f"z_l={z_l:.3f}")

    # Impact parameters crossing the bypass boundary (bypass=3*dx_finest here)
    bypass_mult = 3.0
    bypass_r_Mpc = bypass_mult * finest_spacing / one_Mpc
    print(f"  Bypass radius = {bypass_r_Mpc*1000:.0f} kpc")
    b_values_Mpc = np.array([
        0.02,                # deep inside cusp (bypass active)
        0.10,
        bypass_r_Mpc * 0.5,  # inside
        bypass_r_Mpc * 1.5,  # outside
        0.6,
        2.0,                 # comfortably outside
        rs_Mpc,
    ])
    b_values_Mpc = np.unique(np.round(b_values_Mpc, 4))

    def photons_factory(metric):
        photons = []
        for b_Mpc in b_values_Mpc:
            target = center + (b_Mpc * one_Mpc) * e_perp1
            p = make_photon(obs_pos, target, metric, eta_0, a_0)
            p._lam_S = D_s / c
            photons.append(p)
        return photons, list(b_values_Mpc)

    def metric_builder(interp):
        return PerturbedFLRWMetricFast(
            a_of_eta=a_of_eta, grid=root_grid, interpolator=interp,
            adot_of_eta=cosmo.adot_of_eta, cosmology=cosmo,
            enable_lensing=True, slow_roll=True,
        )

    # ---------- Analytic reference ----------
    b_m = b_values_Mpc * one_Mpc
    k_an = halo.kappa_analytic(b_m, Sigma_cr)
    g_an = halo.gamma_analytic(b_m, Sigma_cr)

    # ---------- Run both cases ----------
    print("\n  Integrating WITHOUT bypass ...")
    t = time.time()
    _, b_no, k_no, g_no, m_no, _ = run_once(
        False, halo, bypass_mult, amr, root_grid, metric_builder,
        photons_factory,
    )
    print(f"    done in {time.time()-t:.0f}s")

    print("  Integrating WITH bypass ...")
    t = time.time()
    _, b_yes, k_yes, g_yes, m_yes, _ = run_once(
        True, halo, bypass_mult, amr, root_grid, metric_builder,
        photons_factory,
    )
    print(f"    done in {time.time()-t:.0f}s")

    # ---------- Report ----------
    k_bg_no = k_no[-1]
    k_bg_yes = k_yes[-1]
    print("\n" + "=" * 72)
    print(f"  {'b [Mpc]':>8}  {'kappa_an':>10}  {'k_no':>10}  {'k_yes':>10}  "
          f"{'mu_no':>8}  {'mu_yes':>8}")
    print("-" * 72)
    for i, b in enumerate(b_values_Mpc):
        mark = "*" if b < bypass_r_Mpc else " "
        print(f" {mark}{b:>8.3f}  {k_an[i]:>10.3e}  "
              f"{k_no[i]-k_bg_no:>10.3e}  {k_yes[i]-k_bg_yes:>10.3e}  "
              f"{m_no[i]:>+8.2f}  {m_yes[i]:>+8.2f}")
    print("=" * 72)
    print(f"  * = impact parameter inside bypass sphere "
          f"({bypass_r_Mpc*1000:.0f} kpc)")
    print(f"  kappa_an_max = {k_an.max():.3f}   "
          f"(k_sim_max no-bypass = {(k_no-k_bg_no).max():.3f}, "
          f"with-bypass = {(k_yes-k_bg_yes).max():.3f})")
    print(f"  mu_max: no-bypass = {m_no.max():.1f}, "
          f"with-bypass = {m_yes.max():.1f}")

    # Critical curve captured?
    got_crit_no = np.any((k_no - k_bg_no) > 1.0)
    got_crit_yes = np.any((k_yes - k_bg_yes) > 1.0)
    print(f"\n  kappa > 1 captured?  no-bypass: {got_crit_no}, "
          f"with-bypass: {got_crit_yes}")
    print(f"\n  Total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
