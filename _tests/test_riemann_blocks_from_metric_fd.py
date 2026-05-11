import numpy as np

from excalibur.core.constants import c
from excalibur.observables.optical_tidal_matrix import optical_tidal_matrix_optimized
from excalibur.observables.riemann_perturbed_flrw import riemann_blocks_kernel
from excalibur.observables.sachs_basis import init_sachs_basis


def quadratic_phi(xyz, hess_phi):
    return 0.5 * float(xyz @ hess_phi @ xyz)


def metric_tensor_static_quadratic(x_mu, a, hess_phi):
    xyz = np.asarray(x_mu[1:4], dtype=float)
    phi = quadratic_phi(xyz, hess_phi)
    g = np.zeros((4, 4), dtype=float)
    g[0, 0] = -a * a * (1.0 + 2.0 * phi / (c * c)) * c * c
    g[1, 1] = a * a * (1.0 - 2.0 * phi / (c * c))
    g[2, 2] = a * a * (1.0 - 2.0 * phi / (c * c))
    g[3, 3] = a * a * (1.0 - 2.0 * phi / (c * c))
    return g


def metric_gradient_fd_from_metric(metric_fn, x_mu, h):
    grad = np.zeros((4, 4, 4), dtype=float)
    for alpha in range(4):
        dx = np.zeros(4)
        dx[alpha] = h
        gp = metric_fn(x_mu + dx)
        gm = metric_fn(x_mu - dx)
        grad[alpha] = (gp - gm) / (2.0 * h)
    return grad


def metric_gradient_fd(x_mu, a, hess_phi, h):
    return metric_gradient_fd_from_metric(
        lambda y: metric_tensor_static_quadratic(y, a, hess_phi), x_mu, h
    )


def christoffel_from_metric_fd_from_metric(metric_fn, x_mu, h):
    g = metric_fn(x_mu)
    g_inv = np.linalg.inv(g)
    dg = metric_gradient_fd_from_metric(metric_fn, x_mu, h)
    gamma = np.zeros((4, 4, 4), dtype=float)
    for rho in range(4):
        for mu in range(4):
            for nu in range(4):
                s = 0.0
                for sigma in range(4):
                    s += g_inv[rho, sigma] * (
                        dg[mu, sigma, nu] + dg[nu, sigma, mu] - dg[sigma, mu, nu]
                    )
                gamma[rho, mu, nu] = 0.5 * s
    return gamma


def christoffel_from_metric_fd(x_mu, a, hess_phi, h):
    return christoffel_from_metric_fd_from_metric(
        lambda y: metric_tensor_static_quadratic(y, a, hess_phi), x_mu, h
    )


def christoffel_gradient_fd_from_metric(metric_fn, x_mu, h):
    out = np.zeros((4, 4, 4, 4), dtype=float)
    for alpha in range(4):
        dx = np.zeros(4)
        dx[alpha] = h
        gp = christoffel_from_metric_fd_from_metric(metric_fn, x_mu + dx, h)
        gm = christoffel_from_metric_fd_from_metric(metric_fn, x_mu - dx, h)
        out[alpha] = (gp - gm) / (2.0 * h)
    return out


def christoffel_gradient_fd(x_mu, a, hess_phi, h):
    return christoffel_gradient_fd_from_metric(
        lambda y: metric_tensor_static_quadratic(y, a, hess_phi), x_mu, h
    )


def riemann_all_down_from_metric_fd_from_metric(metric_fn, x_mu, h):
    g = metric_fn(x_mu)
    gamma = christoffel_from_metric_fd_from_metric(metric_fn, x_mu, h)
    dgamma = christoffel_gradient_fd_from_metric(metric_fn, x_mu, h)

    R_up = np.zeros((4, 4, 4, 4), dtype=float)
    for rho in range(4):
        for sigma in range(4):
            for mu in range(4):
                for nu in range(4):
                    val = dgamma[mu, rho, sigma, nu] - dgamma[nu, rho, sigma, mu]
                    for lam in range(4):
                        val += gamma[rho, lam, mu] * gamma[lam, sigma, nu]
                        val -= gamma[rho, lam, nu] * gamma[lam, sigma, mu]
                    R_up[rho, sigma, mu, nu] = val

    R_down = np.einsum("ar,rbmn->abmn", g, R_up)
    return R_down


def riemann_all_down_from_metric_fd(x_mu, a, hess_phi, h):
    return riemann_all_down_from_metric_fd_from_metric(
        lambda y: metric_tensor_static_quadratic(y, a, hess_phi), x_mu, h
    )


def contract_full_riemann_to_optical_matrix(R_down, k_mu, e1_mu, e2_mu):
    out = np.zeros((2, 2), dtype=float)
    e = np.vstack([e1_mu, e2_mu])
    for A in range(2):
        for B in range(2):
            val = 0.0
            for mu in range(4):
                for alpha in range(4):
                    for nu in range(4):
                        for beta in range(4):
                            val += (
                                R_down[mu, alpha, nu, beta]
                                * k_mu[alpha] * k_mu[beta]
                                * e[A, mu] * e[B, nu]
                            )
            out[A, B] = val
    return out


def metric_tensor_homogeneous_time_only(x_mu, a0, a1, a2, phi0, phi1, phi2):
    eta = float(x_mu[0])
    a = a0 + a1 * eta + 0.5 * a2 * eta * eta
    phi = phi0 + phi1 * eta + 0.5 * phi2 * eta * eta
    g = np.zeros((4, 4), dtype=float)
    g[0, 0] = -a * a * (1.0 + 2.0 * phi / (c * c)) * c * c
    g[1, 1] = a * a * (1.0 - 2.0 * phi / (c * c))
    g[2, 2] = g[1, 1]
    g[3, 3] = g[1, 1]
    return g


def test_hessian_blocks_match_metric_finite_difference_derivation():
    """Independent check of the Hessian blocks derived directly from the metric.

    We use a static metric with constant scale factor and a quadratic potential,
    evaluated at the quadratic center so that grad(Phi)=0 and the exact Riemann
    at that point is purely Hessian-driven.  This gives a metric-based reference
    independent from ``riemann_blocks_kernel``.
    """
    a = 1.07
    hess_phi = np.array([
        [2.0e-8, 0.4e-8, -0.3e-8],
        [0.4e-8, -1.6e-8, 0.7e-8],
        [-0.3e-8, 0.7e-8, 0.9e-8],
    ])

    x_mu = np.array([0.0, 0.0, 0.0, 0.0])
    h = 1.0e6

    R_down_fd = riemann_all_down_from_metric_fd(x_mu, a, hess_phi, h)
    Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
        a,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        np.zeros(3),
        np.zeros(3),
        hess_phi,
        c,
    )

    ref_k00l = R_down_fd[1:4, 0, 0, 1:4]
    ref_0lki = R_down_fd[0, 1:4, 1:4, 1:4]
    ref_kijl = R_down_fd[1:4, 1:4, 1:4, 1:4]

    scale1 = np.max(np.abs(Rd_k00l))
    scale3 = np.max(np.abs(Rd_kijl))
    np.testing.assert_allclose(Rd_k00l, ref_k00l, rtol=5e-4, atol=1e-12 * scale1)
    np.testing.assert_allclose(Rd_0lki, ref_0lki, rtol=0.0, atol=1e-12 * scale1)
    active = np.abs(Rd_kijl) > 1e-30
    np.testing.assert_allclose(Rd_kijl[active], ref_kijl[active], rtol=2e-3, atol=1e-12 * scale3)
    assert np.max(np.abs(ref_kijl[~active])) < 1e-10 * scale3


def test_block1_scalar_matches_homogeneous_time_metric_fd_derivation():
    """Independent check of the homogeneous time-dependent block1 scalar term.

    With no spatial gradients and no spatial Hessian, the only non-zero content
    in R_{k00l} is the diagonal scalar piece.  We reconstruct the full Riemann
    directly from the time-dependent metric and compare that extracted block to
    the closed-form expression used in ``riemann_blocks_kernel``.
    """
    a0, a1, a2 = 1.1, 2.0e-2, -3.0e-4
    phi0, phi1, phi2 = -2.0e8, 3.0e5, -4.0e2
    x_mu = np.array([0.0, 0.0, 0.0, 0.0])
    h = 1.0e-2

    metric_fn = lambda y: metric_tensor_homogeneous_time_only(y, a0, a1, a2, phi0, phi1, phi2)
    R_down_fd = riemann_all_down_from_metric_fd_from_metric(metric_fn, x_mu, h)

    H = a1 / a0
    Hprime = a2 / a0 - H * H
    Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
        a0,
        H,
        Hprime,
        phi0,
        phi1,
        phi2,
        np.zeros(3),
        np.zeros(3),
        np.zeros((3, 3)),
        c,
    )

    ref_k00l = R_down_fd[1:4, 0, 0, 1:4]
    ref_0lki = R_down_fd[0, 1:4, 1:4, 1:4]
    scale1 = np.max(np.abs(Rd_k00l))
    np.testing.assert_allclose(Rd_k00l, ref_k00l, rtol=5e-5, atol=1e-12 * scale1)
    np.testing.assert_allclose(Rd_0lki, ref_0lki, rtol=0.0, atol=1e-12 * scale1)


def test_homogeneous_metric_fd_and_code_differ_at_operator_level_by_construction():
    """The homogeneous FLRW sector uses a background-subtracted optical operator.

    In a purely homogeneous time-dependent metric, a full Riemann tensor rebuilt
    directly from the metric carries the usual isotropic FLRW focusing.  The
    production optical pipeline instead cancels that homogeneous background so
    that the Jacobi map free-streams as D = lambda I and the background distance
    is reintroduced through the affine normalisation.  This test makes that
    convention explicit: the mismatch is real at the raw metric-Riemann level,
    but it should not be interpreted as a block1 validation failure.
    """
    a0, a1, a2 = 1.1, 2.0e-2, -3.0e-4
    phi0, phi1, phi2 = -2.0e8, 3.0e5, -4.0e2
    x_mu = np.array([0.0, 0.0, 0.0, 0.0])
    h = 1.0e-2

    metric_fn = lambda y: metric_tensor_homogeneous_time_only(y, a0, a1, a2, phi0, phi1, phi2)
    R_down_fd = riemann_all_down_from_metric_fd_from_metric(metric_fn, x_mu, h)

    H = a1 / a0
    Hprime = a2 / a0 - H * H
    Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
        a0,
        H,
        Hprime,
        phi0,
        phi1,
        phi2,
        np.zeros(3),
        np.zeros(3),
        np.zeros((3, 3)),
        c,
    )

    direction = np.array([0.37, -0.48, 0.8])
    direction /= np.linalg.norm(direction)
    g_mu_nu = metric_tensor_homogeneous_time_only(x_mu, a0, a1, a2, phi0, phi1, phi2)
    k_spatial = direction * c
    spatial_sq = np.dot(np.diag(g_mu_nu)[1:4], k_spatial * k_spatial)
    k0 = -np.sqrt(abs(-spatial_sq / g_mu_nu[0, 0]))
    k_mu = np.array([k0, *k_spatial])
    e1_mu, e2_mu = init_sachs_basis(k_mu, g_mu_nu, a0)

    R_code = optical_tidal_matrix_optimized(Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu, g_mu_nu)
    R_fd_full = contract_full_riemann_to_optical_matrix(R_down_fd, k_mu, e1_mu, e2_mu)

    scale_fd = np.max(np.abs(R_fd_full))
    assert scale_fd > 0.0
    np.testing.assert_allclose(R_fd_full[0, 0], R_fd_full[1, 1], rtol=5e-5)
    np.testing.assert_allclose(R_fd_full[0, 1], 0.0, atol=1e-12 * scale_fd)
    np.testing.assert_allclose(R_fd_full[1, 0], 0.0, atol=1e-12 * scale_fd)
    assert np.max(np.abs(R_code)) < 1e-6 * scale_fd