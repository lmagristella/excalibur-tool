import numpy as np

from excalibur.core.constants import c, G, one_Mpc, one_Msun
from excalibur.core.cosmology import LCDM_Cosmology
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator_fast import InterpolatorFast
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.metrics.perturbed_flrw_metric_fast import PerturbedFLRWMetricFast
from excalibur.objects.spherical_mass import spherical_mass
from excalibur.photon.photon import Photon
from excalibur.integration.integrator import Integrator


def test_flrw_fast_matches_slow_short_integration_with_spherical_mass_potential():
    """End-to-end: slow vs fast with a non-trivial Phi from spherical_mass.

    We build a static potential Phi(x) (in SI: m^2/s^2) from a uniform spherical mass.
    Then we integrate the same photon with both metrics and compare final (x,u).

    This is intentionally a SMALL test (cheap grid + few steps) to keep CI/runtime sane.
    """

    cosmology = LCDM_Cosmology(70, Omega_m=0.3, Omega_r=0.0, Omega_lambda=0.7)
    _ = cosmology.a_of_eta(1e18)
    eta = cosmology._eta_at_a1

    # Grid
    N = 24
    grid_size = 40 * one_Mpc
    spacing = (grid_size / N, grid_size / N, grid_size / N)
    origin3 = np.array([0.0, 0.0, 0.0])
    grid = Grid((N, N, N), spacing, origin3)

    # Build spherical potential centered in the box
    center = np.array([0.5, 0.5, 0.5]) * grid_size
    M = 1e15 * one_Msun
    R = 5 * one_Mpc
    halo = spherical_mass(mass=M, radius=R, center=center)

    # Sample Phi on grid nodes
    xs = origin3[0] + np.arange(N) * spacing[0]
    ys = origin3[1] + np.arange(N) * spacing[1]
    zs = origin3[2] + np.arange(N) * spacing[2]
    Phi = halo.potential(xs, ys, zs)

    # Sanity: Phi should be negative and finite
    assert np.isfinite(Phi).all()
    assert np.min(Phi) < 0.0

    grid.add_field("Phi", Phi.astype(float))
    interp = InterpolatorFast(grid)

    metric_slow = PerturbedFLRWMetric(cosmology=cosmology, grid=grid, interpolator=interp)
    metric_fast = PerturbedFLRWMetricFast(
        a_of_eta=cosmology.a_of_eta,
        adot_of_eta=cosmology.adot_of_eta,
        grid=grid,
        interpolator=interp,
        analytical_geodesics=False,
    )

    # Initial point away from center but well inside grid (avoid stencil boundary)
    x0 = np.array([eta, 0.25 * grid_size, 0.35 * grid_size, 0.45 * grid_size], dtype=float)

    # Give the photon a direction roughly towards the mass center
    dir3 = center - x0[1:]
    dir3 = dir3 / np.linalg.norm(dir3)
    u_spatial = dir3 * (0.2 * c)

    # Solve u0 from each metric's null condition
    def solve_u0(metric):
        g = metric.metric_tensor(x0)
        g00 = g[0, 0]
        spatial_term = g[1, 1] * u_spatial[0] ** 2 + g[2, 2] * u_spatial[1] ** 2 + g[3, 3] * u_spatial[2] ** 2
        return -np.sqrt(abs(-spatial_term / g00))

    p_slow = Photon(x0.copy(), np.array([solve_u0(metric_slow), *u_spatial], dtype=float))
    p_fast = Photon(x0.copy(), np.array([solve_u0(metric_fast), *u_spatial], dtype=float))

    # Integrate
    num_steps = 300
    dlambda = 5e-4
    int_slow = Integrator(metric=metric_slow, dt=dlambda, integrator="rk4")
    int_fast = Integrator(metric=metric_fast, dt=dlambda, integrator="rk4")

    int_slow.integrate_single(p_slow, stop_mode="steps", stop_value=num_steps)
    int_fast.integrate_single(p_fast, stop_mode="steps", stop_value=num_steps)

    # Compare final states
    dx = np.max(np.abs(p_slow.x - p_fast.x))
    du = np.max(np.abs(p_slow.u - p_fast.u))

    # Tolerances: allow some drift, but it should stay small.
    assert dx < 1e-4 * grid_size
    assert du < 1e-4 * c

    # Bonus: potential should be weak-field (|Phi|/c^2 << 1) in this setup.
    # This is not a physics assertion for the whole code, just a guard that the test
    # isn't pushing into an ultra-relativistic regime.
    Phi_here, _, _ = interp.value_gradient_and_time_derivative(x0[1:], "Phi", eta)
    assert abs(Phi_here) / (c**2) < 1e-3
