import numpy as np

from excalibur.core.constants import c, one_Mpc
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator


def _make_metrics_phi0(N=16, grid_size=20 * one_Mpc):
    cosmology = LCDM_Cosmology(70, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmology.a_of_eta(1e18)
    eta = cosmology._eta_at_a1

    spacing = (grid_size / N, grid_size / N, grid_size / N)
    origin3 = np.array([0.0, 0.0, 0.0])

    grid = Grid((N, N, N), spacing, origin3)
    grid.add_field("Phi", np.zeros((N, N, N), dtype=float))
    interp = InterpolatorFast(grid)

    slow = PerturbedFLRWMetric(cosmology=cosmology, grid=grid, interpolator=interp)
    fast = PerturbedFLRWMetricFast(
        a_of_eta=cosmology.a_of_eta,
        adot_of_eta=cosmology.adot_of_eta,
        grid=grid,
        interpolator=interp,
        analytical_geodesics=False,
    )

    return cosmology, eta, grid_size, slow, fast


def test_flrw_fast_matches_slow_short_integration_phi0():
    """End-to-end: integrate the same photon with slow+fast metrics and compare (x,u).

    We keep Phi=0 so the only differences can come from algebra/caching/numba.
    """

    cosmology, eta, grid_size, metric_slow, metric_fast = _make_metrics_phi0()

    # Initial state well inside grid.
    x0 = np.array([eta, 0.2 * grid_size, 0.3 * grid_size, 0.4 * grid_size], dtype=float)

    # Pick an arbitrary but consistent null initial direction.
    # Spatial components in m/s
    u_spatial = np.array([0.12 * c, -0.07 * c, 0.05 * c], dtype=float)

    # Solve null for u0 separately for each metric (should match closely)
    g_s = metric_slow.metric_tensor(x0)
    g_f = metric_fast.metric_tensor(x0)

    def solve_u0(g):
        g00 = g[0, 0]
        spatial_term = g[1, 1] * u_spatial[0] ** 2 + g[2, 2] * u_spatial[1] ** 2 + g[3, 3] * u_spatial[2] ** 2
        return -np.sqrt(abs(-spatial_term / g00))

    u0_s = solve_u0(g_s)
    u0_f = solve_u0(g_f)

    p_slow = Photon(x0.copy(), np.array([u0_s, *u_spatial], dtype=float))
    p_fast = Photon(x0.copy(), np.array([u0_f, *u_spatial], dtype=float))

    # Integrate a small number of steps.
    # dt here is d(lambda). Choose small enough to avoid divergence.
    num_steps = 200
    dlambda = 1e-3

    int_slow = Integrator(metric=metric_slow, dt=dlambda, integrator="rk4")
    int_fast = Integrator(metric=metric_fast, dt=dlambda, integrator="rk4")

    int_slow.integrate_single(p_slow, stop_mode="steps", stop_value=num_steps)
    int_fast.integrate_single(p_fast, stop_mode="steps", stop_value=num_steps)

    # Compare final state. Tolerances are intentionally not ultra-tight because
    # RK4 + tiny differences can accumulate.
    dx = np.max(np.abs(p_slow.x - p_fast.x))
    du = np.max(np.abs(p_slow.u - p_fast.u))

    assert dx < 1e-6 * grid_size
    assert du < 1e-6 * c
