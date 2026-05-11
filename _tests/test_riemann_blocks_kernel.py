import numpy as np

from excalibur.core.constants import c
from excalibur.observables.optical_tidal_matrix import optical_tidal_matrix_optimized
from excalibur.observables.riemann_perturbed_flrw import riemann_blocks_kernel
from excalibur.observables.sachs_basis import init_sachs_basis


def _expected_block3_hessian(a, hess_phi):
    out = np.zeros((3, 3, 3, 3))
    fac = a * a / (c * c)
    for k_idx in range(3):
        for i_idx in range(3):
            for j_idx in range(3):
                for l_idx in range(3):
                    val = 0.0
                    if k_idx == j_idx:
                        val += hess_phi[i_idx, l_idx]
                    if k_idx == l_idx:
                        val -= hess_phi[i_idx, j_idx]
                    if i_idx == j_idx:
                        val -= hess_phi[k_idx, l_idx]
                    if i_idx == l_idx:
                        val += hess_phi[k_idx, j_idx]
                    out[k_idx, i_idx, j_idx, l_idx] = fac * val
    return out


def _derived_block1_hessian_from_linearized_lapse(a, hess_phi):
    """Derive the Hessian part of R_{k00l} from the perturbed lapse.

    For fixed scale factor and static perturbation,

        g_{00} = -a^2 (1 + 2 Psi / c^2) c^2,

    so with the first index lowered,

        Gamma_{k00} = -1/2 d_k g_{00} = a^2 d_k Psi.

    At linear order the only Hessian contribution to R_{k00l} is then

        R_{k00l}^{(H)} = - d_l Gamma_{k00} = -a^2 d_k d_l Psi.
    """
    return -(a * a) * hess_phi


def _derived_block3_hessian_from_linearized_spatial_metric(a, hess_phi):
    """Derive the Hessian part of R_{kijl} from the linearized spatial metric.

    For a fixed scale factor and static perturbation,

        g_{ij} = a^2 (1 - 2 Phi / c^2) delta_{ij},

    the linearized spatial Christoffel with the first index lowered is

        Gamma_{kij} = (a^2/c^2)
            [ -delta_{kj} d_i Phi - delta_{ki} d_j Phi + delta_{ij} d_k Phi ].

    Keeping only terms linear in Phi, the Hessian contribution to the all-down
    Riemann block is

        R_{kijl}^{(H)} = d_j Gamma_{kil} - d_l Gamma_{kij},

    and the two terms proportional to delta_{ki} cancel because the Hessian is
    symmetric: d_l d_j Phi = d_j d_l Phi.
    """
    fac = a * a / (c * c)
    term_a = np.zeros((3, 3, 3, 3))
    term_b = np.zeros((3, 3, 3, 3))
    term_c = np.zeros((3, 3, 3, 3))
    term_d = np.zeros((3, 3, 3, 3))
    canceled_ki = np.zeros((3, 3, 3, 3))

    for k_idx in range(3):
        for i_idx in range(3):
            for j_idx in range(3):
                for l_idx in range(3):
                    if k_idx == j_idx:
                        term_a[k_idx, i_idx, j_idx, l_idx] = fac * hess_phi[i_idx, l_idx]
                    if k_idx == l_idx:
                        term_b[k_idx, i_idx, j_idx, l_idx] = -fac * hess_phi[i_idx, j_idx]
                    if i_idx == j_idx:
                        term_c[k_idx, i_idx, j_idx, l_idx] = -fac * hess_phi[k_idx, l_idx]
                    if i_idx == l_idx:
                        term_d[k_idx, i_idx, j_idx, l_idx] = fac * hess_phi[k_idx, j_idx]
                    if k_idx == i_idx:
                        canceled_ki[k_idx, i_idx, j_idx, l_idx] = fac * (
                            hess_phi[j_idx, l_idx] - hess_phi[l_idx, j_idx]
                        )

    return term_a, term_b, term_c, term_d, canceled_ki


def _expected_block3_scalar(a, H, Hprime, phi, phi_dot):
    out = np.zeros((3, 3, 3, 3))
    fac = a * a / (c * c)
    second_scalar = Hprime - (2.0 * H * phi_dot + 6.0 * H * H * phi) / (c * c)
    for k_idx in range(3):
        for i_idx in range(3):
            for j_idx in range(3):
                for l_idx in range(3):
                    kron = 0.0
                    if l_idx == i_idx and k_idx == j_idx:
                        kron += 1.0
                    if l_idx == k_idx and i_idx == j_idx:
                        kron -= 1.0
                    out[k_idx, i_idx, j_idx, l_idx] = fac * second_scalar * kron
    return out


def _assert_block3_symmetries(block):
    for k_idx in range(3):
        for i_idx in range(3):
            for j_idx in range(3):
                for l_idx in range(3):
                    np.testing.assert_allclose(block[k_idx, i_idx, j_idx, l_idx], -block[i_idx, k_idx, j_idx, l_idx])
                    np.testing.assert_allclose(block[k_idx, i_idx, j_idx, l_idx], -block[k_idx, i_idx, l_idx, j_idx])
                    np.testing.assert_allclose(block[k_idx, i_idx, j_idx, l_idx], block[j_idx, l_idx, k_idx, i_idx])


def test_block1_and_block3_match_static_newtonian_limit():
    a = 1.13
    H = 0.0
    Hprime = 0.0
    phi = -2.0e10
    phi_dot = 0.0
    phi_ddot = 0.0
    grad_phi = np.array([1.0e-8, -2.0e-8, 0.5e-8])
    grad_phi_dot = np.zeros(3)
    hess_phi = np.array([
        [4.0e-19, 0.5e-19, -0.2e-19],
        [0.5e-19, -3.0e-19, 0.8e-19],
        [-0.2e-19, 0.8e-19, 1.5e-19],
    ])

    Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
        a, H, Hprime, phi, phi_dot, phi_ddot, grad_phi, grad_phi_dot, hess_phi, c
    )

    np.testing.assert_allclose(Rd_k00l, -(a * a) * hess_phi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(Rd_0lki, np.zeros((3, 3, 3)), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(Rd_kijl, _expected_block3_hessian(a, hess_phi), rtol=0.0, atol=0.0)
    _assert_block3_symmetries(Rd_kijl)


def test_block1_hessian_matches_linearized_lapse_derivation():
    a = 1.04
    hess_phi = np.array([
        [3.0e-19, -0.7e-19, 0.1e-19],
        [-0.7e-19, 1.5e-19, 0.5e-19],
        [0.1e-19, 0.5e-19, -2.4e-19],
    ])

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

    np.testing.assert_allclose(Rd_k00l, _derived_block1_hessian_from_linearized_lapse(a, hess_phi), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(Rd_0lki, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(Rd_kijl, _expected_block3_hessian(a, hess_phi), rtol=0.0, atol=0.0)


def test_block1_scalar_piece_is_pure_diagonal():
    a = 0.91
    H = 2.1e-18
    Hprime = -1.4e-36
    phi = -7.0e12
    phi_dot = 3.0e-4
    phi_ddot = -9.0e-22
    grad_phi = np.zeros(3)
    grad_phi_dot = np.zeros(3)
    hess_phi = np.zeros((3, 3))

    Rd_k00l, Rd_0lki, _ = riemann_blocks_kernel(
        a, H, Hprime, phi, phi_dot, phi_ddot, grad_phi, grad_phi_dot, hess_phi, c
    )

    diag_scalar = Hprime * (1.0 - 2.0 * phi / (c * c)) + phi_ddot / (c * c) + 2.0 * H * phi_dot / (c * c)
    expected = (a * a) * diag_scalar * np.eye(3)

    np.testing.assert_allclose(Rd_k00l, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(Rd_0lki, np.zeros((3, 3, 3)), rtol=0.0, atol=0.0)


def test_block3_scalar_piece_matches_expected_kron_structure():
    a = 0.84
    H = 1.8e-18
    Hprime = -8.0e-37
    phi = -5.0e11
    phi_dot = 2.5e-4
    phi_ddot = 0.0
    grad_phi = np.zeros(3)
    grad_phi_dot = np.zeros(3)
    hess_phi = np.zeros((3, 3))

    _, _, Rd_kijl = riemann_blocks_kernel(
        a, H, Hprime, phi, phi_dot, phi_ddot, grad_phi, grad_phi_dot, hess_phi, c
    )

    expected = _expected_block3_scalar(a, H, Hprime, phi, phi_dot)
    np.testing.assert_allclose(Rd_kijl, expected, rtol=0.0, atol=0.0)
    _assert_block3_symmetries(Rd_kijl)


def test_block3_hessian_matches_linearized_spatial_metric_derivation():
    a = 0.93
    hess_phi = np.array([
        [1.4e-18, 0.2e-18, -0.5e-18],
        [0.2e-18, -0.9e-18, 0.6e-18],
        [-0.5e-18, 0.6e-18, 0.3e-18],
    ])

    _, _, Rd_kijl = riemann_blocks_kernel(
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

    term_a, term_b, term_c, term_d, canceled_ki = _derived_block3_hessian_from_linearized_spatial_metric(a, hess_phi)
    derived = term_a + term_b + term_c + term_d + canceled_ki

    scale = np.max(np.abs(Rd_kijl))
    np.testing.assert_allclose(canceled_ki, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(Rd_kijl, derived, rtol=0.0, atol=1e-14 * scale)

    expected = _expected_block3_hessian(a, hess_phi)
    np.testing.assert_allclose(term_a + term_b + term_c + term_d, expected, rtol=0.0, atol=1e-14 * scale)


def test_pure_flrw_blocks_cancel_in_optical_matrix():
    a = 0.87
    H = 2.0e-18
    Hprime = -9.5e-37
    phi = 0.0
    phi_dot = 0.0
    phi_ddot = 0.0
    grad_phi = np.zeros(3)
    grad_phi_dot = np.zeros(3)
    hess_phi = np.zeros((3, 3))

    Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
        a, H, Hprime, phi, phi_dot, phi_ddot, grad_phi, grad_phi_dot, hess_phi, c
    )

    direction = np.array([0.37, -0.48, 0.8])
    direction /= np.linalg.norm(direction)
    k_mu = np.array([1.0, *(c * direction)])

    g_mu_nu = np.diag([-a * a * c * c, a * a, a * a, a * a])
    e1_mu, e2_mu = init_sachs_basis(k_mu, g_mu_nu, a)
    R_AB = optical_tidal_matrix_optimized(Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu, g_mu_nu)

    scale = abs(a * a * Hprime)
    assert np.max(np.abs(R_AB)) < 1e-12 * scale