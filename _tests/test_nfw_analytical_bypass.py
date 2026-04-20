"""
Tests for NFW analytical potential derivatives and the
AnalyticalBypassInterpolator wrapper.
"""

import numpy as np
import pytest

from excalibur.core.constants import G, one_Mpc, one_Msun
from excalibur.objects.nfw_halo import NFWHalo
from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator


# ---------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def halo():
    center = np.array([0.0, 0.0, 0.0])
    return NFWHalo(M_200=2.0e15 * one_Msun, c_NFW=7.0, center=center)


# ---------------------------------------------------------------
#  1) Gradient matches finite-difference of potential
# ---------------------------------------------------------------
def test_gradient_matches_finite_difference(halo):
    rs = halo.r_s
    # Sample at a few radii and directions
    pts = [
        np.array([0.3 * rs, 0.0, 0.0]),
        np.array([0.0, 1.5 * rs, 0.4 * rs]),
        np.array([2.0 * rs, -1.0 * rs, 0.7 * rs]),
    ]
    h = 1e-3 * rs
    for p in pts:
        gx, gy, gz = halo.potential_gradient(p[0], p[1], p[2])
        fd = np.empty(3)
        for i in range(3):
            pp = p.copy(); pp[i] += h
            pm = p.copy(); pm[i] -= h
            fd[i] = (float(halo.potential(pp[0], pp[1], pp[2]))
                     - float(halo.potential(pm[0], pm[1], pm[2]))) / (2.0 * h)
        scale = np.max(np.abs(fd)) + 1e-30
        np.testing.assert_allclose([gx, gy, gz], fd, rtol=0.0, atol=1e-4 * scale)


# ---------------------------------------------------------------
#  2) Hessian matches finite-difference of gradient
# ---------------------------------------------------------------
def test_hessian_matches_finite_difference(halo):
    rs = halo.r_s
    p = np.array([1.2 * rs, -0.5 * rs, 0.3 * rs])
    H = halo.potential_hessian(p[0], p[1], p[2])
    h = 1e-3 * rs
    H_fd = np.empty((3, 3))
    for j in range(3):
        pp = p.copy(); pp[j] += h
        pm = p.copy(); pm[j] -= h
        gp = np.array(halo.potential_gradient(pp[0], pp[1], pp[2]))
        gm = np.array(halo.potential_gradient(pm[0], pm[1], pm[2]))
        H_fd[:, j] = (gp - gm) / (2.0 * h)
    np.testing.assert_allclose(H, H_fd, rtol=1e-4, atol=0.0)


# ---------------------------------------------------------------
#  3) Poisson equation: trace(H) = 4 pi G rho(r)
# ---------------------------------------------------------------
def test_hessian_satisfies_poisson(halo):
    rs = halo.r_s
    for radius in [0.2 * rs, 0.8 * rs, 2.0 * rs, 5.0 * rs]:
        p = np.array([radius, 0.0, 0.0])
        H = halo.potential_hessian(p[0], p[1], p[2])
        trace = H[0, 0] + H[1, 1] + H[2, 2]
        rho = float(halo.density(p[0], p[1], p[2]))
        expected = 4.0 * np.pi * G * rho
        np.testing.assert_allclose(trace, expected, rtol=1e-10)


# ---------------------------------------------------------------
#  4) Shell theorem cross-check: |grad| = GM(<r)/r^2
# ---------------------------------------------------------------
def test_gradient_magnitude_matches_shell_theorem(halo):
    rs = halo.r_s
    for radius in [0.3 * rs, 1.0 * rs, 4.0 * rs]:
        p = np.array([0.7 * radius, 0.5 * radius, 0.5 * radius])
        p *= radius / np.linalg.norm(p)
        g = np.array(halo.potential_gradient(p[0], p[1], p[2]))
        r = np.linalg.norm(p)
        expected = G * halo.mass_enclosed(r) / r ** 2
        np.testing.assert_allclose(np.linalg.norm(g), expected, rtol=1e-10)


# ---------------------------------------------------------------
#  5) Bypass switching: outside -> delegates, inside -> analytical
# ---------------------------------------------------------------
class _DummyBase:
    """Minimal stand-in for a grid interpolator. Returns constants."""

    def __init__(self):
        self.grid = None
        self.boundary = "clamp"

    def value_gradient_hessian_and_time_derivative(self, x, field, t=None):
        return (-999.0, (1.0, 2.0, 3.0),
                (11.0, 22.0, 33.0, 12.0, 13.0, 23.0), 0.0)

    def value_gradient_and_time_derivative(self, x, field, t=None):
        return -999.0, (1.0, 2.0, 3.0), 0.0

    def interpolate(self, x, field, t=None):
        return -999.0

    def gradient(self, x, field, t=None):
        return (1.0, 2.0, 3.0)

    def value_and_gradient(self, x, field, t=None):
        return -999.0, (1.0, 2.0, 3.0)

    def hessian(self, x, field, t=None):
        return np.array([[11., 12., 13.], [12., 22., 23.], [13., 23., 33.]])

    def laplacian(self, x, field, t=None):
        return 66.0


def test_bypass_switches_at_boundary(halo):
    rs = halo.r_s
    bypass_r = 0.5 * rs
    base = _DummyBase()
    wrap = AnalyticalBypassInterpolator(
        base_interp=base,
        analytical_source=halo,
        bypass_radius=bypass_r,
    )

    inside = np.array([0.2 * rs, 0.0, 0.0])
    outside = np.array([1.5 * rs, 0.0, 0.0])

    # Outside -> dummy values
    v_out, g_out, h_out, dt_out = wrap.value_gradient_hessian_and_time_derivative(
        outside, "Phi"
    )
    assert v_out == -999.0
    assert g_out == (1.0, 2.0, 3.0)

    # Inside -> analytical values (should NOT be the dummy)
    v_in, g_in, h_in, dt_in = wrap.value_gradient_hessian_and_time_derivative(
        inside, "Phi"
    )
    expected_phi = float(halo.potential(inside[0], inside[1], inside[2]))
    expected_grad = halo.potential_gradient(inside[0], inside[1], inside[2])
    np.testing.assert_allclose(v_in, expected_phi, rtol=1e-12)
    np.testing.assert_allclose(g_in, expected_grad, rtol=1e-12)
    assert v_in != -999.0


def test_bypass_ignores_non_bypass_fields(halo):
    rs = halo.r_s
    base = _DummyBase()
    wrap = AnalyticalBypassInterpolator(
        base_interp=base,
        analytical_source=halo,
        bypass_radius=0.5 * rs,
        bypass_fields=("Phi",),
    )
    inside = np.array([0.2 * rs, 0.0, 0.0])
    # Request a different field -> should pass through unchanged
    v, g, h, dt = wrap.value_gradient_hessian_and_time_derivative(inside, "Psi")
    assert v == -999.0
