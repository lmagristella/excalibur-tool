import math

import numpy as np
import pytest
from scipy import interpolate
from scipy.optimize import brentq

pytest.importorskip("numba")

from excalibur.core.constants import c, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.amr_grid import AMRGrid, AMRInterpolator
from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
from excalibur.grid.grid import Grid
from excalibur.integration.integrator import Integrator
from excalibur.integration.integrator_numba import NumbaAMRBackend, integrate_photon_numba
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.objects.nfw_halo import NFWHalo
from excalibur.observables.lensing_conventions import sigma_cr_conventions
from excalibur.observables.optical_tidal_matrix import lensing_from_jacobi
from excalibur.observables.sachs_basis import init_sachs_basis
from excalibur.photon.photon import Photon


def _build_setup():
    cosmo = LCDM_Cosmology(70.0, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmo.a_of_eta(1e18)
    eta_0 = cosmo._eta_at_a1
    a_0 = cosmo.a_of_eta(eta_0)

    eta_min = 0.5 * eta_0
    eta_arr = np.linspace(eta_min, eta_0, 2000)
    a_arr = np.array([cosmo.a_of_eta(eta) for eta in eta_arr])
    a_of_eta = interpolate.interp1d(eta_arr, a_arr, kind="cubic", fill_value="extrapolate")

    box_mpc = 400.0
    n_root = 64
    grid_size = box_mpc * one_Mpc
    root_grid = Grid(
        shape=(n_root, n_root, n_root),
        spacing=(grid_size / n_root,) * 3,
        origin=np.zeros(3),
    )

    halo = NFWHalo(2e15 * one_Msun, 7.0, np.array([0.5, 0.5, 0.5]) * grid_size)
    coords = np.linspace(0.0, grid_size, n_root)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    root_grid.add_field("Phi", halo.potential(x, y, z))

    amr = AMRGrid.from_field(
        root_grid,
        "Phi",
        lambda x_val, y_val, z_val: halo.potential(x_val, y_val, z_val),
        max_level=3,
        ratio=4,
        refine_threshold=0.005,
        refine_mode="gradient",
        min_patch_cells=32,
        boundary="clamp",
        scheme="tricubic",
        verbose=False,
    )
    amr_interp_base = AMRInterpolator(amr, boundary="clamp", scheme="tricubic")
    amr_interp = AnalyticalBypassInterpolator(
        base_interp=amr_interp_base,
        analytical_source=halo,
        bypass_radius=np.inf,
        bypass_fields=("Phi",),
        time_derivative=0.0,
    )

    obs_pos = np.array([box_mpc / 2.0, box_mpc / 2.0, 5.0]) * one_Mpc
    center = halo.center
    d_l = float(np.linalg.norm(center - obs_pos))
    d_s = min(cosmo.comoving_distance(0.12), 0.95 * (grid_size - np.min(obs_pos)))
    d_ls = d_s - d_l
    z_l = brentq(lambda redshift: cosmo.comoving_distance(redshift) - d_l, 0.0, 5.0)
    z_s = brentq(lambda redshift: cosmo.comoving_distance(redshift) - d_s, 0.0, 5.0)
    sigma_cr_comoving, sigma_cr_physical = sigma_cr_conventions(d_l, d_s, d_ls, z_l)

    dir_hat = (center - obs_pos) / d_l
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, dir_hat)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e_perp1 = seed - np.dot(seed, dir_hat) * dir_hat
    e_perp1 /= np.linalg.norm(e_perp1)

    impact_parameter = 1.0 * one_Mpc
    target = center + impact_parameter * e_perp1
    dt = halo.r_s / (8.0 * c)
    lambda_total = d_s / c
    n_steps = int(math.ceil(lambda_total / dt)) + 4

    return {
        "cosmo": cosmo,
        "eta_0": eta_0,
        "eta_min": eta_min,
        "a_0": a_0,
        "a_of_eta": a_of_eta,
        "root_grid": root_grid,
        "amr": amr,
        "amr_interp": amr_interp,
        "halo": halo,
        "obs_pos": obs_pos,
        "target": target,
        "dt": dt,
        "lambda_total": lambda_total,
        "n_steps": n_steps,
        "impact_parameter": impact_parameter,
        "z_s": z_s,
        "sigma_cr_comoving": sigma_cr_comoving,
        "sigma_cr_physical": sigma_cr_physical,
        "bypass_radius": np.inf,
    }


def _make_metric(setup, convention):
    return PerturbedFLRWMetricFast(
        a_of_eta=setup["a_of_eta"],
        grid=setup["root_grid"],
        interpolator=setup["amr_interp"],
        adot_of_eta=setup["cosmo"].adot_of_eta,
        cosmology=setup["cosmo"],
        enable_lensing=True,
        slow_roll=True,
        sachs_screen_convention=convention,
    )


def _make_photon(obs_pos, target, metric, eta_0, a_0):
    obs_4d = np.array([eta_0, *obs_pos])
    direction = target - obs_pos
    direction /= np.linalg.norm(direction)
    screen_convention = getattr(metric, "sachs_screen_convention", "metric")
    g_mu_nu = metric.metric_tensor(obs_4d)
    if screen_convention == "conformal_metric":
        g_init = g_mu_nu / (a_0 * a_0)
        basis_a = 1.0
    else:
        g_init = g_mu_nu
        basis_a = a_0

    k_spatial = direction * c
    spatial_sq = (
        g_init[1, 1] * k_spatial[0] ** 2
        + g_init[2, 2] * k_spatial[1] ** 2
        + g_init[3, 3] * k_spatial[2] ** 2
    )
    k0 = -np.sqrt(abs(-spatial_sq / g_init[0, 0]))
    k_mu = np.array([k0, *k_spatial])
    e1_mu, e2_mu = init_sachs_basis(k_mu, g_init, basis_a, convention=screen_convention)

    photon = Photon(obs_4d.copy(), k_mu.copy())
    photon.e1 = e1_mu.copy()
    photon.e2 = e2_mu.copy()
    photon.D_flat = np.zeros(4)
    photon.P_flat = np.array([1.0, 0.0, 0.0, 1.0])
    return photon


def _integrate_python(metric, setup):
    photon = _make_photon(setup["obs_pos"], setup["target"], metric, setup["eta_0"], setup["a_0"])
    integrator = Integrator(
        metric=metric,
        dt=setup["dt"],
        mode="sequential",
        integrator="rk4",
        rtol=1e-8,
        atol=1e-13,
    )
    integrator.integrate_single(
        photon,
        stop_mode="affine",
        stop_value=setup["lambda_total"],
        record_every=0,
    )
    return photon


def _integrate_numba(setup, convention):
    metric = _make_metric(setup, convention)
    photon = _make_photon(setup["obs_pos"], setup["target"], metric, setup["eta_0"], setup["a_0"])
    backend = NumbaAMRBackend(
        setup["amr"],
        setup["cosmo"],
        c_val=c,
        slow_roll=True,
        lensing=True,
        sachs_screen_convention=convention,
        analytical_source=setup["halo"],
        bypass_radius=setup["bypass_radius"],
        eta_range=(setup["eta_min"], setup["eta_0"]),
    )
    backend.warmup()
    integrate_photon_numba(
        photon,
        backend,
        dt=setup["dt"],
        n_steps=setup["n_steps"],
        lambda_stop=setup["lambda_total"],
        record_every=0,
    )
    return photon


def _lensing_triplet(photon):
    kappa, _, gamma = lensing_from_jacobi(photon.D_flat / photon.lambda_affine)
    return kappa, gamma, photon.lambda_affine


def test_numba_screen_convention_matches_python_and_reference():
    setup = _build_setup()

    metric_py = _make_metric(setup, "metric")
    conformal_py = _make_metric(setup, "conformal_metric")

    photon_py_metric = _integrate_python(metric_py, setup)
    photon_py_conf = _integrate_python(conformal_py, setup)
    photon_nb_metric = _integrate_numba(setup, "metric")
    photon_nb_conf = _integrate_numba(setup, "conformal_metric")

    kappa_py_metric, gamma_py_metric, lambda_py_metric = _lensing_triplet(photon_py_metric)
    kappa_py_conf, gamma_py_conf, lambda_py_conf = _lensing_triplet(photon_py_conf)
    kappa_nb_metric, gamma_nb_metric, lambda_nb_metric = _lensing_triplet(photon_nb_metric)
    kappa_nb_conf, gamma_nb_conf, lambda_nb_conf = _lensing_triplet(photon_nb_conf)
    z_end_py_conf = 1.0 / setup["cosmo"].a_of_eta(photon_py_conf.x[0]) - 1.0
    z_end_nb_conf = 1.0 / setup["cosmo"].a_of_eta(photon_nb_conf.x[0]) - 1.0

    halo = setup["halo"]
    impact_parameter = np.array([setup["impact_parameter"]])
    kappa_an_phys = float(halo.kappa_analytic(impact_parameter, setup["sigma_cr_physical"])[0])
    gamma_an_phys = float(halo.gamma_analytic(impact_parameter, setup["sigma_cr_physical"])[0])
    kappa_an_conf = float(halo.kappa_analytic(impact_parameter, setup["sigma_cr_comoving"])[0])
    gamma_an_conf = float(halo.gamma_analytic(impact_parameter, setup["sigma_cr_comoving"])[0])

    assert abs(kappa_nb_conf / kappa_an_conf - 1.0) < 0.08
    assert abs(gamma_nb_conf / gamma_an_conf - 1.0) < 0.08
    # The "metric" screen branch is known to carry a setup-dependent bias
    # v_S / (a_S * v_tilde_S) on top of the (1+z_l) factor of the conformal
    # branch (see _audits/2026-05-29_conformal_screen_audit.md). With the
    # corrected Sigma_cr_physical formula (= comoving * (1+z_l)), the bias
    # is no longer accidentally absorbed and a ~15% residual is expected.
    assert abs(kappa_nb_metric / kappa_an_phys - 1.0) < 0.25
    assert abs(gamma_nb_metric / gamma_an_phys - 1.0) < 0.25

    assert abs(kappa_nb_conf - kappa_py_conf) / abs(kappa_py_conf) < 0.03
    assert abs(gamma_nb_conf - gamma_py_conf) / abs(gamma_py_conf) < 0.03
    assert abs(kappa_nb_metric - kappa_py_metric) / abs(kappa_py_metric) < 0.04
    assert abs(gamma_nb_metric - gamma_py_metric) / abs(gamma_py_metric) < 0.04

    assert kappa_nb_metric > 1.05 * kappa_nb_conf
    assert gamma_nb_metric > 1.03 * gamma_nb_conf

    assert abs(lambda_nb_metric - lambda_py_metric) / lambda_py_metric < 5e-4
    assert abs(lambda_nb_conf - lambda_py_conf) / lambda_py_conf < 5e-4
    # The "python + conformal_metric" branch is mathematically mixed: it
    # integrates the geodesic with full FLRW Christoffels (physical affine v)
    # but interprets lambda_total = d_s/c as a CONFORMAL affine. The resulting
    # eta_end overshoots z_s by O(<a^2>-1), which can reach ~0.01 even for the
    # small z_s = 0.089 used here. See _audits/diagnose_z_end.py and
    # _audits/2026-05-29_conformal_screen_audit.md (branch B analysis).
    # The numba specialized branch (a=1 in geodesic) is unaffected and stays
    # at the tighter tolerance.
    assert abs(z_end_py_conf - setup["z_s"]) < 1.5e-2
    assert abs(z_end_nb_conf - setup["z_s"]) < 5e-3