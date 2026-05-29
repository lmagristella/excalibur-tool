#!/usr/bin/env python3
"""
TRIAXIAL NFW lensing simulation  --  FULLY ANALYTICAL variant.

Companion to ``run_lensing_nfw_analytic.py`` (which uses a *spherical*
:class:`NFWHalo`). This variant swaps in a :class:`TriaxialNFWHalo` so the
projected lens is *elliptical* on the sky -- which opens the tangential
caustic into the characteristic 4-cusp astroid (the "diamond") and produces
4- and 5-image configurations.

Geometry note: the optical axis is +z, so the sky plane is (x, y). With the
default identity orientation, the on-sky projected axis ratio is ``b/a``
(``--axis-ratio-ba``); make it < 1 to get a visibly elliptical lens.

Output schema matches the spherical run (np.savez keys identical), so every
post-processing / viewer script applies unchanged. Files are named
``lensing_nfw_triaxial_...`` so they sort apart from the spherical runs.
"""

import argparse
import os, sys, time
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import c, G, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.amr_grid import AMRGrid
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator
from excalibur.integration.integrator_numba import (
    NumbaAMRBackend as StandardNumbaAMRBackend,
    integrate_photon_numba as integrate_photon_numba_standard,
)
from excalibur.observables.lensing_conventions import (
    DEFAULT_LENSING_REFERENCE_CONVENTION,
    lensing_convention_label,
    sigma_cr_conventions,
)
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from excalibur.objects.nfw_halo import NFWHalo, TriaxialNFWHalo
from excalibur.io.filename_utils import RunNamer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fully analytical TRIAXIAL NFW lensing simulation."
    )
    parser.add_argument("--backend", choices=("python", "numba"), default="numba")
    parser.add_argument("--numba-kernel", choices=("standard", "lowalloc", "specialized"), default="standard")
    parser.add_argument("--n-root", type=int, default=16)
    parser.add_argument("--box-mpc", type=float, default=1950.0)
    parser.add_argument("--mass-200-msun", type=float, default=2e15)
    parser.add_argument("--c-nfw", type=float, default=7.0)
    parser.add_argument("--axis-ratio-ba", type=float, default=0.6,
                        help="Intermediate/major axis ratio b/a (on-sky projected "
                             "ellipticity for the default orientation). 1.0 = spherical.")
    parser.add_argument("--axis-ratio-ca", type=float, default=0.6,
                        help="Minor/major axis ratio c/a (line-of-sight axis). "
                             "Must satisfy 0 < c/a <= b/a <= 1.")
    parser.add_argument("--orientation-euler", type=float, nargs=3, default=None,
                        metavar=("RX", "RY", "RZ"),
                        help="Optional intrinsic XYZ Euler angles (deg) rotating the "
                             "halo principal axes. Default: aligned with (x, y, z).")
    parser.add_argument("--label", type=str, default=None,
                        help="Explicit human label saved in the .npz (display_label), "
                             "used verbatim by the interactive viewer's profile switcher. "
                             "If omitted, the viewer infers a label from the geometry.")
    parser.add_argument("--z-lens", type=float, default=None)
    parser.add_argument("--z-source", type=float, default=1.0)
    parser.add_argument("--obs-z-mpc", type=float, default=5.0)
    parser.add_argument("--source-margin-mpc", type=float, default=20.0)
    parser.add_argument("--map-n1d", type=int, default=31)
    parser.add_argument("--map-half-mpc", type=float, default=1.5)
    parser.add_argument("--profile-b-max-mpc", type=float, default=None)
    parser.add_argument("--n-fine-per-rs", type=int, default=8)
    parser.add_argument("--integrator", choices=("rk4", "dopri5"), default="rk4",
                        help="rk4 = fixed-step; dopri5 = adaptive Dormand-Prince 5(4) "
                             "(requires --numba-kernel=specialized).")
    parser.add_argument("--rtol", type=float, default=1e-6,
                        help="Relative tolerance for the adaptive integrator (Jacobi block).")
    parser.add_argument("--atol", type=float, default=1e-9,
                        help="Absolute tolerance for the adaptive integrator (Jacobi block).")
    parser.add_argument("--parallel-batch", action="store_true",
                        help="Integrate all photons in a single numba.prange batch "
                             "(requires --backend=numba --numba-kernel=specialized "
                             "--integrator=dopri5).")
    parser.add_argument("--map-chunk", type=int, default=0,
                        help="Chunk size for the map batch (0 = single batch). "
                             "Useful for very large screens to bound peak memory.")
    parser.add_argument("--no-history", action="store_true",
                        help="Skip writing photon.history during parallel-batch. "
                             "Drops trajectory output for the map but enables "
                             "very large screens (>1 M photons) without OOM.")
    return parser.parse_args()


def _select_numba_backend(kernel_name):
    if kernel_name == "standard":
        return StandardNumbaAMRBackend, integrate_photon_numba_standard

    if kernel_name == "specialized":
        from excalibur.integration.integrator_numba_specialized import (
            NumbaAMRBackend as SpecializedNumbaAMRBackend,
            integrate_photon_numba as integrate_photon_numba_specialized,
        )
        return SpecializedNumbaAMRBackend, integrate_photon_numba_specialized

    from excalibur.integration.integrator_numba_lowalloc import (
        NumbaAMRBackend as LowAllocNumbaAMRBackend,
        integrate_photon_numba as integrate_photon_numba_lowalloc,
    )

    return LowAllocNumbaAMRBackend, integrate_photon_numba_lowalloc


def _integrate_one_photon(photon, backend_name, *, integrator_fine,
                          numba_backend, integrate_photon_impl, dt_fine, lambda_total,
                          record_every, adaptive_kwargs=None):
    if backend_name == "numba":
        if adaptive_kwargs is not None:
            integrate_photon_impl(
                photon,
                numba_backend,
                dt_init=dt_fine,
                lambda_stop=lambda_total,
                record_every=record_every,
                **adaptive_kwargs,
            )
            return photon.lambda_affine

        _, lam, _ = integrate_photon_impl(
            photon,
            numba_backend,
            dt=dt_fine,
            n_steps=int(np.ceil(lambda_total / dt_fine)) + 4,
            lambda_stop=lambda_total,
            record_every=record_every,
        )
        return lam

    integrator_fine.integrate_single(
        photon,
        stop_mode="affine",
        stop_value=lambda_total,
        record_every=record_every,
    )
    return photon.lambda_affine


def make_photon(obs_pos, target, metric, eta_0, a_0):
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
    spatial_sq = (g_init[1, 1] * k_spatial[0] ** 2
                  + g_init[2, 2] * k_spatial[1] ** 2
                  + g_init[3, 3] * k_spatial[2] ** 2)
    k0 = -np.sqrt(abs(-spatial_sq / g_init[0, 0]))
    k_mu = np.array([k0, *k_spatial])

    e1, e2 = init_sachs_basis(k_mu, g_init, basis_a, convention=screen_convention)

    p = Photon(obs_4d.copy(), k_mu.copy(), record_lensing=True)
    p.e1 = e1.copy()
    p.e2 = e2.copy()
    return p


def main():
    args = parse_args()
    t_total = time.time()
    numba_backend_cls, integrate_photon_impl = _select_numba_backend(args.numba_kernel)

    adaptive_kwargs = None
    integrate_photons_batch_impl = None
    if args.integrator == "dopri5":
        if args.backend != "numba" or args.numba_kernel != "specialized":
            raise ValueError("--integrator=dopri5 requires --backend=numba --numba-kernel=specialized.")
        from excalibur.integration.integrator_numba_specialized import (
            integrate_photon_numba_dopri5,
            integrate_photons_batch_dopri5,
        )
        integrate_photon_impl = integrate_photon_numba_dopri5
        integrate_photons_batch_impl = integrate_photons_batch_dopri5
        adaptive_kwargs = {"rtol": float(args.rtol), "atol": float(args.atol)}

    if args.parallel_batch and args.integrator != "dopri5":
        raise ValueError("--parallel-batch requires --integrator=dopri5.")

    # =================================================================
    # 1. COSMOLOGY
    # =================================================================
    print("=" * 70)
    print("  TRIAXIAL NFW LENSING  --  FULLY ANALYTICAL")
    print("=" * 70)
    print("\n1. Cosmology ...")
    print(f"   backend = {args.backend}")
    if args.backend == "numba":
        print(f"   numba kernel = {args.numba_kernel}")
    if args.integrator != "rk4":
        print(f"   integrator = {args.integrator}  (rtol={args.rtol:.1e}, atol={args.atol:.1e})")
    if args.parallel_batch:
        print(f"   parallel-batch = on (numba.prange over photons)")
    if args.z_lens is not None:
        print(f"   requested z_l = {args.z_lens:.4f}, requested z_s = {args.z_source:.4f}")
    H0 = 70.0
    Omega_m, Omega_lambda = 0.3, 0.7
    cosmo = LCDM_Cosmology(H0, Omega_m=Omega_m, Omega_r=0,
                           Omega_lambda=Omega_lambda)

    explicit_redshift_geometry = args.z_lens is not None
    if explicit_redshift_geometry:
        if args.z_source <= args.z_lens:
            raise ValueError("--z-source must be larger than --z-lens for a lensing configuration")
        D_l_target = cosmo.comoving_distance(args.z_lens)
        D_s_target = cosmo.comoving_distance(args.z_source)
    else:
        D_l_target = None
        D_s_target = None

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
    # 2. HALO + GRID
    # =================================================================
    # The grid is a placeholder, we set the analytical bypass to have a very large
    # radius, so Phi is never read from it. We keep a tiny root grid purely to satisfy the metric/interpolator API (shape/spacing).
    print("2. Triaxial NFW halo ...")
    box_Mpc = args.box_mpc
    if explicit_redshift_geometry:
        required_box_mpc = args.obs_z_mpc + D_s_target / one_Mpc + args.source_margin_mpc
        if box_Mpc < required_box_mpc:
            box_Mpc = required_box_mpc
            print(f"   auto-expanded box to {box_Mpc:.1f} Mpc for requested z_s")
    grid_size = box_Mpc * one_Mpc
    N_root = args.n_root
    root_grid = Grid(
        shape   = (N_root, N_root, N_root),
        spacing = (grid_size / N_root,) * 3,
        origin  = np.array([0.0, 0.0, 0.0]),
    )
    root_grid.add_field("Phi", np.zeros((N_root,) * 3))

    M_200 = args.mass_200_msun * one_Msun
    c_NFW = args.c_nfw
    if explicit_redshift_geometry:
        center = np.array([
            0.5 * box_Mpc,
            0.5 * box_Mpc,
            args.obs_z_mpc + D_l_target / one_Mpc,
        ]) * one_Mpc
    else:
        center = np.array([0.5, 0.5, 0.5]) * grid_size
    axis_ratios = (args.axis_ratio_ba, args.axis_ratio_ca)
    halo   = TriaxialNFWHalo(
        M_200, c_NFW, center,
        axis_ratios=axis_ratios,
        orientation_euler_deg=args.orientation_euler,
    )

    R200_Mpc = halo.R_200 / one_Mpc
    rs_Mpc   = halo.r_s   / one_Mpc

    print(f"   {halo}")
    print(f"   axis ratios (b/a, c/a) = ({halo.q_intermediate:.3f}, {halo.q_minor:.3f})"
          f"   on-sky projected b/a = {halo.q_intermediate:.3f}")
    print(f"   R_200 = {R200_Mpc:.3f} Mpc,  r_s = {rs_Mpc*1000:.0f} kpc")
    print(f"   Dummy grid: {N_root}^3, box = {box_Mpc:.0f} Mpc "
          f"(placeholder, never queried)")

    # =================================================================
    # 3. ANALYTICAL BYPASS INTERPOLATOR  (infinite radius)
    # =================================================================
    print("\n3. Analytical bypass interpolator ...")
    base_interp = InterpolatorFast(root_grid, boundary="clamp")
    # Any radius > box diagonal covers the whole simulated region.
    bypass_radius = 1e3 * grid_size
    interp = AnalyticalBypassInterpolator(
        base_interp      = base_interp,
        analytical_source= halo,
        bypass_radius    = bypass_radius,
        bypass_fields    = ("Phi",),
        time_derivative  = 0.0,
    )
    print(f"   [ok] bypass_radius = infty (every query returns analytical NFW)")

    # =================================================================
    # 4. METRIC
    # =================================================================
    print("4. Metric ...")
    metric = PerturbedFLRWMetricFast(
        a_of_eta       = a_of_eta,
        grid           = root_grid,
        interpolator   = interp,
        adot_of_eta    = cosmo.adot_of_eta,
        cosmology      = cosmo,
        enable_lensing = True,
        slow_roll      = True,
        analytical_geodesics = True,
        sachs_screen_convention = "conformal_metric",
    )
    print("   [ok] metric ready (analytical NFW potential)")

    # =================================================================
    # 5. PHOTON CONE
    # =================================================================
    print("5. Photon cone ...")

    obs_z_Mpc = args.obs_z_mpc
    obs_pos = np.array([box_Mpc / 2, box_Mpc / 2, obs_z_Mpc]) * one_Mpc

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

    # Impact-parameter profile  --  dense sampling near the Einstein ring
    b_E_approx = 0.46  # Mpc  (approximate Einstein radius for this geometry)
    b_values_Mpc = np.unique(np.sort(np.concatenate([
        np.array([0.0]),
        np.linspace(0.005, 0.05, 12),                       # deep cusp
        np.linspace(0.05, 0.10, 6),
        np.linspace(0.10, b_E_approx * 1.5, 24),            # Einstein ring region
        np.linspace(b_E_approx * 1.5, rs_Mpc, 8),
        np.linspace(rs_Mpc + 0.1, R200_Mpc, 12),
        np.linspace(R200_Mpc + 0.2, 5.0, 8),
        np.linspace(6.0, 15.0, 5),
        np.logspace(np.log10(0.005), np.log10(15.0), 30),
    ])))
    if args.profile_b_max_mpc is not None:
        b_values_Mpc = b_values_Mpc[b_values_Mpc <= args.profile_b_max_mpc]
        if b_values_Mpc.size == 0:
            raise ValueError("--profile-b-max-mpc leaves no profile photons; choose a value >= 0")
    b_values = b_values_Mpc * one_Mpc

    photons_profile = []
    b_profile_Mpc = []
    target_profile_Mpc = []
    for b in b_values:
        target = center + b * e_perp1
        p = make_photon(obs_pos, target, metric, eta_0, a_0)
        photons_profile.append(p)
        b_profile_Mpc.append(b / one_Mpc)
        target_profile_Mpc.append(target / one_Mpc)
    n_profile = len(photons_profile)

    # 2D map  --  focus on strong-lensing region
    map_half_Mpc = args.map_half_mpc
    n_map_1d = args.map_n1d
    b1_arr = np.linspace(-map_half_Mpc, map_half_Mpc, n_map_1d) * one_Mpc
    b2_arr = np.linspace(-map_half_Mpc, map_half_Mpc, n_map_1d) * one_Mpc

    photons_map = []
    map_b1_Mpc, map_b2_Mpc = [], []
    target_map_Mpc = []
    for b1 in b1_arr:
        for b2 in b2_arr:
            target = center + b1 * e_perp1 + b2 * e_perp2
            p = make_photon(obs_pos, target, metric, eta_0, a_0)
            photons_map.append(p)
            map_b1_Mpc.append(b1 / one_Mpc)
            map_b2_Mpc.append(b2 / one_Mpc)
            target_map_Mpc.append(target / one_Mpc)

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
    n_fine_per_rs = args.n_fine_per_rs
    dt_fine = halo.r_s / (n_fine_per_rs * c)
    step_fine = c * dt_fine

    if explicit_redshift_geometry:
        D_s = D_s_target
        z_source = args.z_source
    else:
        D_s = cosmo.comoving_distance(args.z_source)
    max_dist_in_box = grid_size - np.min(obs_pos)
    if explicit_redshift_geometry:
        if D_s > max_dist_in_box:
            raise ValueError(
                "Requested z_source lies outside the simulation box even after expansion; "
                "increase --source-margin-mpc or --box-mpc"
            )
    else:
        D_s = min(D_s, 0.95 * max_dist_in_box)
    D_ls = D_s - D_l
    if not explicit_redshift_geometry:
        try:
            z_source = brentq(lambda z: cosmo.comoving_distance(z) - D_s,
                              0.0, 5.0)
        except ValueError:
            z_source = 0.05
    DA_FLRW = cosmo.angular_diameter_distance(z_source)

    lambda_total = D_s / c
    n_steps_total = int(np.ceil(D_s / step_fine))
    traj_stride = max(1, n_steps_total // 500)   # ~500 pts/trajectory

    print(f"   D_l  = {D_l/one_Mpc:.1f} Mpc")
    print(f"   D_s  = {D_s/one_Mpc:.1f} Mpc")
    print(f"   D_ls = {D_ls/one_Mpc:.1f} Mpc")
    print(f"   z_s  ~ {z_source:.4f}")
    print(f"   Uniform fine step: {step_fine/one_Mpc*1000:.0f} kpc, "
          f"n_steps = {n_steps_total}, traj_stride = {traj_stride}")

    integrator_fine = None
    numba_backend = None
    if args.backend == "python":
        integrator_fine = Integrator(
            metric     = metric,
            dt         = dt_fine,
            mode       = "sequential",
            integrator = "rk4",
            rtol       = 1e-8,
            atol       = 1e-13,
        )
    else:
        analytical_amr = AMRGrid(root_grid)
        numba_backend = numba_backend_cls(
            analytical_amr,
            cosmo,
            c_val=c,
            slow_roll=True,
            lensing=True,
            sachs_screen_convention=getattr(metric, "sachs_screen_convention", "metric"),
            analytical_source=halo,
            bypass_radius=bypass_radius,
            eta_range=(eta_min, eta_0),
        )
        print("   Warming up numba kernels ...")
        t_warm = time.time()
        numba_backend.warmup()
        print(f"   [ok] numba warmup in {time.time()-t_warm:.1f}s")

    # =================================================================
    # 7. INTEGRATE
    # =================================================================
    all_photons = photons_profile + photons_map
    lambda_S = lambda_total

    print(f"\n7. Integrating {n_total} photons ({args.backend}) ...")
    print(f"   chi_S = c*lambda_S = {c*lambda_S/one_Mpc:.1f} Mpc")
    t_int = time.time()

    kappas  = np.empty(n_total)
    mus     = np.empty(n_total)
    gammas  = np.empty(n_total)
    D_flats = np.empty((n_total, 4))
    final_pos = np.empty((n_total, 3))
    lambda_actuals = np.empty(n_total)

    def _store_photon_result(i, lam):
        photon = all_photons[i]
        D_norm = photon.D_flat / lam
        kappa, mu, shear = lensing_from_jacobi(D_norm)
        kappas[i]         = kappa
        mus[i]            = mu
        gammas[i]         = shear
        D_flats[i]        = D_norm
        final_pos[i]      = photon.x[1:4]
        lambda_actuals[i] = lam
        return kappa, shear

    if args.parallel_batch:
        # Bound trajectory buffer by a generous multiple of the RK4 step count
        # (DOPRI5 takes far fewer accepted steps in practice).
        max_steps_batch = max(4 * n_steps_total, 1024)
        traj_capacity_profile = 2
        # DOPRI5 takes far fewer accepted steps than the RK4 bound, and the
        # raytrace map only needs a handful of points to locate the source-plane
        # crossing (the 41^3 run was valid with 2-3 pts/photon). Cap the per-photon
        # buffer so large maps don't OOM: n_photons x cap x 24 x 8 bytes.
        traj_capacity_map = min(2 + max_steps_batch // max(traj_stride, 1) + 2, 64)

        # Profile photons: no trajectory recording.
        if n_profile > 0:
            integrate_photons_batch_impl(
                photons_profile,
                numba_backend,
                dt_init=dt_fine,
                lambda_stop=lambda_total,
                rtol=adaptive_kwargs["rtol"],
                atol=adaptive_kwargs["atol"],
                max_steps=max_steps_batch,
                record_every=0,
                traj_capacity=traj_capacity_profile,
            )
        # Map photons: record every traj_stride accepted steps. Process in
        # chunks (--map-chunk) so the per-batch trajectory buffer stays bounded:
        # peak memory ~ chunk x traj_capacity_map x 24 x 8 bytes. Essential for
        # million-photon maps (e.g. chunk=50000 -> ~0.6 GiB/chunk vs 12 GiB).
        if n_map > 0:
            chunk = args.map_chunk if args.map_chunk and args.map_chunk > 0 else n_map
            for start in range(0, n_map, chunk):
                sub = photons_map[start:start + chunk]
                integrate_photons_batch_impl(
                    sub,
                    numba_backend,
                    dt_init=dt_fine,
                    lambda_stop=lambda_total,
                    rtol=adaptive_kwargs["rtol"],
                    atol=adaptive_kwargs["atol"],
                    max_steps=max_steps_batch,
                    record_every=traj_stride,
                    traj_capacity=traj_capacity_map,
                )
                if chunk < n_map:
                    done = min(start + chunk, n_map)
                    print(f"   map chunk {done}/{n_map} "
                          f"({time.time()-t_int:.1f}s)", flush=True)

        for i, photon in enumerate(all_photons):
            kappa, shear = _store_photon_result(i, float(photon.lambda_affine))

        elapsed = time.time() - t_int
        print(f"   [{n_total:4d}/{n_total}] batch done in {elapsed:.2f}s "
              f"({elapsed/n_total*1000:.2f} ms/photon)")
    else:
        for i, photon in enumerate(all_photons):
            record_every = 0 if i < n_profile else traj_stride
            lam = _integrate_one_photon(
                photon,
                args.backend,
                integrator_fine=integrator_fine,
                numba_backend=numba_backend,
                integrate_photon_impl=integrate_photon_impl,
                dt_fine=dt_fine,
                lambda_total=lambda_total,
                record_every=record_every,
                adaptive_kwargs=adaptive_kwargs,
            )

            kappa, shear = _store_photon_result(i, lam)

            if i % 50 == 0 or i == 0:
                elapsed = time.time() - t_int
                rate = (i + 1) / elapsed
                eta = (n_total - i - 1) / rate if rate > 0 else 0
                print(f"   [{i+1:4d}/{n_total}]  "
                      f"kappa = {kappa:+.6e}  |gamma| = {shear:.3e}  "
                      f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    dt_elapsed = time.time() - t_int
    print(f"  Done in {dt_elapsed:.1f} s  ({dt_elapsed/n_total:.2f} s/photon)")

    # Collect trajectories from photon history.
    # With record_lensing=True the recorded state layout is:
    #   [0:4]   x  (eta, x, y, z)
    #   [4:8]   u  (k^mu)
    #   [8:12]  D_flat  (Jacobi matrix, 4 components)
    #   [12:16] P_flat  (Jacobi velocity, 4 components)
    #   [16:]   metric quantities (a, phi, ...)
    raw_trajs = [
        np.array([[s[0], *s[1:4]/one_Mpc] for s in p.history.states])
        for p in all_photons
    ]
    raw_D = [
        np.array([s[8:12] for s in p.history.states])
        for p in all_photons
    ]
    raw_P = [
        np.array([s[12:16] for s in p.history.states])
        for p in all_photons
    ]
    traj_n_pts = np.array([t.shape[0] for t in raw_trajs], dtype=np.int32)
    max_pts = int(traj_n_pts.max())
    traj_x4_Mpc   = np.full((n_total, max_pts, 4), np.nan, dtype=np.float32)
    traj_D_flat   = np.full((n_total, max_pts, 4), np.nan, dtype=np.float32)
    traj_P_flat   = np.full((n_total, max_pts, 4), np.nan, dtype=np.float32)
    for i in range(n_total):
        n = traj_n_pts[i]
        traj_x4_Mpc[i, :n] = raw_trajs[i]
        traj_D_flat[i,  :n] = raw_D[i]
        traj_P_flat[i,  :n] = raw_P[i]
    total_mb = (traj_x4_Mpc.nbytes + traj_D_flat.nbytes + traj_P_flat.nbytes) / 1e6
    print(f"   Trajectories: {n_total} × {max_pts} pts "
          f"(stride={traj_stride}) = {total_mb:.0f} MB")

    # =================================================================
    # 8. EXTRACT
    # =================================================================
    b_prof  = np.array(b_profile_Mpc)
    target_profile_Mpc = np.array(target_profile_Mpc, dtype=np.float64)
    k_prof  = kappas[:n_profile]
    g_prof  = gammas[:n_profile]
    m_prof  = mus[:n_profile]

    b1_map = np.array(map_b1_Mpc)
    b2_map = np.array(map_b2_Mpc)
    target_map_Mpc = np.array(target_map_Mpc, dtype=np.float64)
    k_map  = kappas[n_profile:]
    g_map  = gammas[n_profile:]
    m_map  = mus[n_profile:]

    # =================================================================
    # 9. NFW ANALYTIC REFERENCE  (Sigma_cr with lensing kernel)
    # =================================================================
    if explicit_redshift_geometry:
        z_l = args.z_lens
    else:
        z_l = brentq(lambda z: cosmo.comoving_distance(z) - D_l, 0.0, 5.0)
    Sigma_cr_comoving, Sigma_cr_physical = sigma_cr_conventions(D_l, D_s, D_ls, z_l)
    Sigma_cr = Sigma_cr_comoving
    Sigma_cr_Mpc2 = Sigma_cr * one_Mpc ** 2 / one_Msun
    Sigma_cr_physical_Mpc2 = Sigma_cr_physical * one_Mpc ** 2 / one_Msun
    reference_label = lensing_convention_label(DEFAULT_LENSING_REFERENCE_CONVENTION)

    DA_l  = cosmo.angular_diameter_distance(z_l)
    DA_s  = cosmo.angular_diameter_distance(z_source)
    DA_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_source)

    b_m = np.abs(b_prof) * one_Mpc
    b_m = np.maximum(b_m, 1e-3 * one_Mpc)
    # TriaxialNFWHalo has no closed-form circular kappa/gamma profile (the lens
    # is elliptical), so the 1D analytic reference is undefined. We keep the
    # schema by filling NaN; the raytracing viewer never reads these fields.
    nan_ref = np.full_like(b_m, np.nan, dtype=float)
    k_analytic_comoving = nan_ref.copy()
    g_analytic_comoving = nan_ref.copy()
    k_analytic_physical = nan_ref.copy()
    g_analytic_physical = nan_ref.copy()
    k_analytic = k_analytic_comoving
    g_analytic = g_analytic_comoving

    # =================================================================
    # 10. SAVE
    # =================================================================
    # Orientation tag so runs that differ only by orientation don't share a
    # filename (e.g. broadside vs 45-deg-inclined prolate have the same b/a,c/a).
    if args.orientation_euler is None:
        orient_tag = "aligned"
    else:
        orient_tag = "e" + "_".join(f"{a:g}" for a in args.orientation_euler)

    namer = RunNamer(
        "lensing_nfw_triaxial",
        integrator="rk4",
        metric="FLRWP1",
        profile="triaxnfw",
        M=M_200 / one_Msun,
        c_NFW=c_NFW,
        ba=round(halo.q_intermediate, 3),
        ca=round(halo.q_minor, 3),
        orient=orient_tag,
        Rvir=round(R200_Mpc, 3),
        rs=round(rs_Mpc, 4),
        zl=round(z_l, 5),
        zs=round(z_source, 5),
        Dl=round(D_l / one_Mpc, 1),
        Ds=round(D_s / one_Mpc, 1),
        obs=(round(obs_pos[0] / one_Mpc, 1),
             round(obs_pos[1] / one_Mpc, 1),
             round(obs_pos[2] / one_Mpc, 1)),
        box_Mpc=box_Mpc,
        Nph=n_total,
    )
    outfile = namer.npz()

    source_plane_center_Mpc = obs_pos / one_Mpc + (D_s / one_Mpc) * dir_hat

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
        final_pos_map_Mpc=final_pos[n_profile:] / one_Mpc,
        target_map_Mpc=target_map_Mpc,
        # Halo params
        halo_type="TriaxialNFW",
        axis_ratio_ba=halo.q_intermediate,
        axis_ratio_ca=halo.q_minor,
        halo_rotation_matrix=halo.rotation_matrix,
        display_label=(args.label if args.label else ""),
        N_grid=N_root,
        box_Mpc=box_Mpc,
        M_Msun=M_200 / one_Msun,
        M_200_Msun=M_200 / one_Msun,
        c_NFW=c_NFW,
        R_200_Mpc=R200_Mpc,
        r_s_Mpc=rs_Mpc,
        R_vir_Mpc=R200_Mpc,
        n_steps=n_steps_total,
        dt_fine=dt_fine,
        Sigma_cr=Sigma_cr,
        Sigma_cr_comoving=Sigma_cr_comoving,
        Sigma_cr_physical=Sigma_cr_physical,
        lensing_reference_convention=DEFAULT_LENSING_REFERENCE_CONVENTION,
        D_l_Mpc=D_l / one_Mpc,
        D_s_Mpc=D_s / one_Mpc,
        D_ls_Mpc=D_ls / one_Mpc,
        DA_l_Mpc=DA_l / one_Mpc,
        DA_s_Mpc=DA_s / one_Mpc,
        DA_ls_Mpc=DA_ls / one_Mpc,
        z_l=z_l,
        sigma_kms=0.0,
        geometry_mode="explicit_redshift" if explicit_redshift_geometry else "box_centered",
        requested_z_l=args.z_lens if explicit_redshift_geometry else -1.0,
        requested_z_source=args.z_source,
        # Cosmology & affine
        lambda_S=lambda_S,
        H0_kms_Mpc=H0,
        Omega_m=Omega_m,
        Omega_lambda=Omega_lambda,
        z_source=z_source,
        # Grid info
        grid_type="ANALYTIC",
        lens_center_Mpc=center / one_Mpc,
        optical_axis_hat=dir_hat,
        screen_e1=e_perp1,
        screen_e2=e_perp2,
        source_plane_center_Mpc=source_plane_center_Mpc,
        final_pos_profile_Mpc=final_pos[:n_profile] / one_Mpc,
        target_profile_Mpc=target_profile_Mpc,
        # Trajectories for raytracing
        # state layout: [x(4), u(4), D_flat(4), P_flat(4), quantities...]
        traj_x4_Mpc=traj_x4_Mpc,
        traj_D_flat=traj_D_flat,
        traj_P_flat=traj_P_flat,
        traj_n_pts=traj_n_pts,
        traj_stride=traj_stride,
        obs_pos_Mpc=obs_pos / one_Mpc,
    )
    print(f"\n   [ok] Saved to {outfile}")

    # =================================================================
    # 11. SUMMARY
    # =================================================================
    total = time.time() - t_total
    k_bg = k_prof[-1]
    dk_prof = np.abs(k_prof - k_bg)

    print("\n" + "=" * 70)
    print("  NFW LENSING  --  ANALYTICAL  --  SUMMARY")
    print("=" * 70)
    print(f"  Halo     : Triaxial NFW, M_200 = {M_200/one_Msun:.0e} Msun, c = {c_NFW:.0f}, "
          f"b/a = {halo.q_intermediate:.2f}, c/a = {halo.q_minor:.2f}")
    print(f"             R_200 = {R200_Mpc:.3f} Mpc,  r_s = {rs_Mpc*1000:.0f} kpc")
    print(f"  Source   : analytical (no grid reconstruction)")
    print(f"  Geometry : chi_l = {D_l/one_Mpc:.1f},  chi_s = {D_s/one_Mpc:.1f},  "
          f"chi_ls = {D_ls/one_Mpc:.1f}  Mpc")
    print(f"           DA_l = {DA_l/one_Mpc:.1f},  DA_s = {DA_s/one_Mpc:.1f},  "
          f"DA_ls = {DA_ls/one_Mpc:.1f}  Mpc")
    print(f"  z_l ~ {z_l:.4f},  z_s ~ {z_source:.4f}")
    print(f"  Sigma_cr_ref ({reference_label}) = {Sigma_cr_Mpc2:.3e} Msun/Mpc^2")
    print(f"  Sigma_cr_physical      = {Sigma_cr_physical_Mpc2:.3e} Msun/Mpc^2")
    print(f"  Photons  : {n_profile} profile + {n_map} map = {n_total}")
    print(f"  Steps    : {n_steps_total}")
    print(f"  kappa_bg    = {k_bg:.6e}")
    print(f"  dk_max      = {dk_prof.max():.4e}  "
          f"(analytic {reference_label}: {k_analytic.max():.4e})")
    print(f"  |gamma|_max = {g_prof.max():.4e}  "
          f"(analytic {reference_label}: {g_analytic.max():.4e})")
    print(f"  mu_max      = {m_prof.max():+.2f}")
    print(f"  kappa > 1 hit : {(k_prof - k_bg > 1.0).any()}")
    print(f"  Total time  : {total/60:.1f} min")
    print("=" * 70)

    # Background-subtracted ratio
    ka_bg = k_analytic[-1]
    ka_corrected = np.maximum(k_analytic - ka_bg, 0.0)

    mask = (b_prof > 0.03) & (b_prof < R200_Mpc) & (ka_corrected > 0)
    if mask.any():
        ratio_k = dk_prof[mask] / ka_corrected[mask]
        ratio_k = ratio_k[np.isfinite(ratio_k)]
        if len(ratio_k) > 0:
            print(f"  dk_num / (kappa_NFW - kappa_bg) (b < R_200): "
                  f"{ratio_k.mean():.3f} +/- {ratio_k.std():.3f}")

    mask_g = (b_prof > 0.05) & (b_prof < R200_Mpc) & (g_analytic > 0)
    if mask_g.any():
        ratio_g = g_prof[mask_g] / g_analytic[mask_g]
        ratio_g = ratio_g[np.isfinite(ratio_g)]
        if len(ratio_g) > 0:
            print(f"  |gamma|_num / gamma_NFW ({reference_label}, b < R_200): "
                  f"{ratio_g.mean():.3f} +/- {ratio_g.std():.3f}")


if __name__ == "__main__":
    main()
