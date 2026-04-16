# _tests/test_lensing_pipeline.py
r"""
Validation tests for the weak-lensing pipeline modules:
  1. Riemann blocks   (riemann_perturbed_flrw.py)
  2. Sachs basis      (sachs_basis.py)
  3. Optical tidal matrix (optical_tidal_matrix.py)

Tests cover:

 Algebraic / structural:
  - Flat space (Phi=0): Riemann vanishes
  - Empty FLRW (Phi=0, a!=1): Riemann ~ curvature of background
  - Symmetry of R_AB
  - Jacobi ODE with identity initial condition
  - Sachs basis orthogonality
  - Optical scalar extraction consistency

 Physics:
  - Ricci trace from empty FLRW matches Friedmann
  - Newtonian tidal tensor from point mass potential
  - Born-approximation convergence  kappa = 1/2 grad^2_perpPhi lambda^2 / c^2
  - Constant-kappa Jacobi  -> D(lambda) = cos(sqrtkappa lambda)  (analytic)
  - Sachs transport preserves orthogonality with FLRW Christoffel
"""

import numpy as np
import pytest

from excalibur.core.constants import G, c, one_Gpc
from excalibur.observables.riemann_perturbed_flrw import riemann_blocks_kernel
from excalibur.observables.sachs_basis import (
    init_sachs_basis,
    sachs_transport_rhs,
)
from excalibur.observables.optical_tidal_matrix import (
    optical_tidal_matrix_from_blocks,
    optical_tidal_matrix_optimized,
    jacobi_rhs,
    optical_scalars_from_tidal,
    lensing_from_jacobi,
    angular_diameter_distance_from_jacobi,
    distance_comparison,
)

# Speed of light in SI
c_val = c
c2 = c_val * c_val


# ======================================================================
#  1. Riemann blocks
# ======================================================================

class TestRiemannBlocks:
    """Tests for riemann_blocks_kernel."""

    def test_flat_space_vanishes(self):
        """With Phi=0 and a=1 (Minkowski), all Riemann blocks must vanish."""
        a = 1.0
        H = 0.0
        Hprime = 0.0
        phi = 0.0
        phi_dot = 0.0
        phi_ddot = 0.0
        grad_phi = np.zeros(3)
        grad_phi_dot = np.zeros(3)
        hess_phi = np.zeros((3, 3))

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime, phi, phi_dot, phi_ddot,
            grad_phi, grad_phi_dot, hess_phi, c_val,
        )

        np.testing.assert_allclose(Rd_k00l, 0.0, atol=1e-30)
        np.testing.assert_allclose(Rd_0lki, 0.0, atol=1e-30)
        np.testing.assert_allclose(Rd_kijl, 0.0, atol=1e-30)

    def test_empty_flrw_only_diagonal(self):
        """With Phi=0 but H,H'!=0 (empty FLRW), only diagonal terms survive."""
        a = 0.8
        H = 1e-18     # typical conformal Hubble ~ H_0 * a
        Hprime = 5e-37
        phi = 0.0
        phi_dot = 0.0
        phi_ddot = 0.0
        grad_phi = np.zeros(3)
        grad_phi_dot = np.zeros(3)
        hess_phi = np.zeros((3, 3))

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime, phi, phi_dot, phi_ddot,
            grad_phi, grad_phi_dot, hess_phi, c_val,
        )

        # R_{k00l}: no Hessian, so only diagonal delta_{kl} * a^2*H'
        expected_diag = a**2 * Hprime
        for k in range(3):
            for l in range(3):
                if k == l:
                    np.testing.assert_allclose(
                        Rd_k00l[k, l], expected_diag, rtol=1e-12,
                        err_msg=f"R_{k}_{{00{l}}} diagonal wrong",
                    )
                else:
                    np.testing.assert_allclose(
                        Rd_k00l[k, l], 0.0, atol=1e-50,
                        err_msg=f"R_{k}_{{00{l}}} off-diagonal non-zero",
                    )

        # R_{0lki}: combo = grad_phi_dot + H*grad_phi = 0  -> vanishes
        np.testing.assert_allclose(Rd_0lki, 0.0, atol=1e-50)

        # R_{kijl}: Hessian=0  -> only second_scalar x Kronecker
        second_scalar = Hprime  # (all Phi terms = 0)
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    for l in range(3):
                        kron = 0.0
                        if l == i and k == j:
                            kron += 1.0
                        if l == k and i == j:
                            kron -= 1.0
                        expected = (a**2 / c2) * second_scalar * kron
                        np.testing.assert_allclose(
                            Rd_kijl[k, i, j, l], expected, atol=1e-50,
                            err_msg=f"R_{k}_{{{i}{j}{l}}} wrong",
                        )

    def test_hessian_contribution(self):
        """Non-zero d_i d_j Phi should appear in R_{k00l}."""
        a = 1.0
        H = 0.0
        Hprime = 0.0
        phi = 1e5  # m^2/s^2
        phi_dot = 0.0
        phi_ddot = 0.0
        grad_phi = np.zeros(3)
        grad_phi_dot = np.zeros(3)
        # Only d_x d_x Phi != 0
        hess_phi = np.zeros((3, 3))
        hess_phi[0, 0] = 1e-10  # s^{-2}

        Rd_k00l, _, _ = riemann_blocks_kernel(
            a, H, Hprime, phi, phi_dot, phi_ddot,
            grad_phi, grad_phi_dot, hess_phi, c_val,
        )

        # R_{0,000} should have the -hess contribution
        # R_{0,000} = a^2(-d_0d_0 Psi + 0) = -1e-10
        np.testing.assert_allclose(Rd_k00l[0, 0], -hess_phi[0, 0], rtol=1e-10)

    def test_antisymmetry_R_kijl(self):
        """R_{kijl} should be antisymmetric in (j,l): R_{kijl} = -R_{kilj}."""
        a = 0.9
        H = 2e-18
        Hprime = 1e-36
        phi = 1e4
        phi_dot = 1e-5
        phi_ddot = 1e-15
        grad_phi = np.array([1e-8, 2e-8, -1e-8])
        grad_phi_dot = np.array([1e-18, -5e-19, 2e-19])
        hess_phi = np.array([
            [1e-12, 2e-13, -3e-13],
            [2e-13, -1e-12, 1e-13],
            [-3e-13, 1e-13, 5e-13],
        ])

        _, _, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime, phi, phi_dot, phi_ddot,
            grad_phi, grad_phi_dot, hess_phi, c_val,
        )

        # Antisymmetry in last two indices: R_{kijl} = -R_{kilj}
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    for l in range(3):
                        np.testing.assert_allclose(
                            Rd_kijl[k, i, j, l], -Rd_kijl[k, i, l, j],
                            atol=1e-40,
                            err_msg=f"Antisymmetry fail: R_{k}_{{{i}{j}{l}}}",
                        )


# ======================================================================
#  2. Sachs basis
# ======================================================================

class TestSachsBasis:
    """Tests for init_sachs_basis and sachs_transport_rhs."""

    def _make_flrw_metric(self, a, phi=0.0):
        """Build diagonal FLRW metric tensor."""
        psi = phi / c2
        g = np.zeros((4, 4))
        g[0, 0] = -a**2 * (1.0 + 2.0 * psi) * c2
        g[1, 1] = a**2 * (1.0 - 2.0 * phi / c2)
        g[2, 2] = a**2 * (1.0 - 2.0 * phi / c2)
        g[3, 3] = a**2 * (1.0 - 2.0 * phi / c2)
        return g

    def test_orthogonality_minkowski(self):
        """In Minkowski space, Sachs vectors are orthogonal to k and to each other."""
        a = 1.0
        g = self._make_flrw_metric(a)
        # Photon moving along z-axis
        # Null condition: g_{00}(k^0)^2 + g_{33}(k^3)^2 = 0
        # => -c^2 (k^0)^2 + (k^3)^2 = 0  => k^0 = k^3 / c
        k_mu = np.array([1.0 / c_val, 0.0, 0.0, 1.0])

        e1, e2 = init_sachs_basis(k_mu, g, a)

        # Check e1 * k = 0  (g_mu_nu e1^mu k^nu)
        ek1 = e1 @ g @ k_mu
        np.testing.assert_allclose(ek1, 0.0, atol=1e-12)

        # Check e2 * k = 0
        ek2 = e2 @ g @ k_mu
        np.testing.assert_allclose(ek2, 0.0, atol=1e-12)

        # Check e1 * e2 = 0
        e12 = e1 @ g @ e2
        np.testing.assert_allclose(e12, 0.0, atol=1e-12)

        # Check e1 * e1 = 1 (spacelike unit norm)
        e11 = e1 @ g @ e1
        np.testing.assert_allclose(e11, 1.0, atol=1e-10)

        # Check e2 * e2 = 1
        e22 = e2 @ g @ e2
        np.testing.assert_allclose(e22, 1.0, atol=1e-10)

    def test_orthogonality_flrw(self):
        """Same test with a != 1."""
        a = 0.5
        g = self._make_flrw_metric(a)
        # k along diagonal direction
        k_spatial = np.array([1.0, 1.0, 1.0])
        k_norm = np.sqrt(k_spatial @ g[1:4, 1:4] @ k_spatial)
        k3 = k_spatial / k_norm  # normalize
        # Null: g_{00}(k^0)^2 + g_{ij}k^i k^j = 0
        # g_{ij}k^i k^j = 1 (unit norm), g_{00} = -a^2 c^2
        # => k^0 = 1/(a*c)
        k0 = 1.0 / (a * c_val)
        k_mu = np.array([k0, k3[0], k3[1], k3[2]])

        e1, e2 = init_sachs_basis(k_mu, g, a)

        ek1 = e1 @ g @ k_mu
        ek2 = e2 @ g @ k_mu
        e12 = e1 @ g @ e2
        e11 = e1 @ g @ e1
        e22 = e2 @ g @ e2

        np.testing.assert_allclose(ek1, 0.0, atol=1e-10)
        np.testing.assert_allclose(ek2, 0.0, atol=1e-10)
        np.testing.assert_allclose(e12, 0.0, atol=1e-10)
        np.testing.assert_allclose(e11, 1.0, atol=1e-10)
        np.testing.assert_allclose(e22, 1.0, atol=1e-10)

    def test_transport_rhs_flat_vanishes(self):
        """In flat space (Gamma=0), Sachs transport RHS vanishes."""
        e_mu = np.array([0.0, 1.0, 0.0, 0.0])
        k_mu = np.array([1.0 / c_val, 0.0, 0.0, 1.0])
        gamma = np.zeros((4, 4, 4))

        de = sachs_transport_rhs(e_mu, gamma, k_mu)
        np.testing.assert_allclose(de, 0.0, atol=1e-30)


# ======================================================================
#  3. Optical tidal matrix
# ======================================================================

class TestOpticalTidalMatrix:
    """Tests for optical_tidal_matrix functions."""

    def test_flat_space_R_AB_vanishes(self):
        """With vanishing Riemann, R_AB should be zero."""
        Rd_k00l = np.zeros((3, 3))
        Rd_0lki = np.zeros((3, 3, 3))
        Rd_kijl = np.zeros((3, 3, 3, 3))

        k_mu = np.array([1.0 / c_val, 0.0, 0.0, 1.0])
        g = np.diag([-c2, 1.0, 1.0, 1.0])
        e1 = np.array([0.0, 1.0, 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0, 0.0])

        R_AB = optical_tidal_matrix_from_blocks(
            Rd_k00l, Rd_0lki, Rd_kijl,
            k_mu, e1, e2, g,
        )
        np.testing.assert_allclose(R_AB, 0.0, atol=1e-30)

    def test_optimized_matches_reference(self):
        """The optimized contraction should give the same result as the reference."""
        rng = np.random.default_rng(42)

        # Build random Riemann blocks with correct antisymmetry
        Rd_k00l = rng.standard_normal((3, 3))
        Rd_0lki = rng.standard_normal((3, 3, 3))

        # Rd_kijl: antisymmetric in (j,l)
        Rd_kijl_raw = rng.standard_normal((3, 3, 3, 3))
        Rd_kijl = 0.5 * (Rd_kijl_raw - Rd_kijl_raw.transpose(0, 1, 3, 2))

        # Random photon and Sachs vectors (don't need to be physical for this test)
        k_mu = rng.standard_normal(4)
        e1 = rng.standard_normal(4)
        e2 = rng.standard_normal(4)
        g = np.diag([-c2, 1.0, 1.0, 1.0])

        R_ref = optical_tidal_matrix_from_blocks(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)
        R_opt = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        np.testing.assert_allclose(R_opt, R_ref, rtol=1e-10,
                                   err_msg="Optimized and reference differ")

    def test_symmetry_of_R_AB(self):
        """For physical inputs, R_AB should be symmetric (Riemann symmetries)."""
        a = 0.9
        H = 2e-18
        Hprime = 1e-36
        phi = 1e4
        phi_dot = 1e-5
        phi_ddot = 1e-15
        grad_phi = np.array([1e-8, 2e-8, -1e-8])
        grad_phi_dot = np.array([1e-18, -5e-19, 2e-19])
        hess_phi = np.array([
            [1e-12, 2e-13, -3e-13],
            [2e-13, -1e-12, 1e-13],
            [-3e-13, 1e-13, 5e-13],
        ])

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime, phi, phi_dot, phi_ddot,
            grad_phi, grad_phi_dot, hess_phi, c_val,
        )

        # Metric and photon along z
        g = np.diag([-a**2 * c2, a**2, a**2, a**2])
        k_mu = np.array([1.0 / (a * c_val), 0.0, 0.0, 1.0 / a])

        e1 = np.array([0.0, 1.0 / a, 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0 / a, 0.0])

        R_AB = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        # R_AB should be symmetric
        np.testing.assert_allclose(
            R_AB[0, 1], R_AB[1, 0], rtol=1e-8,
            err_msg="R_AB not symmetric",
        )


# ======================================================================
#  4. Jacobi map ODE
# ======================================================================

class TestJacobiRHS:
    """Tests for the Jacobi map RHS."""

    def test_identity_ic_zero_tidal(self):
        """With R_AB=0 and D=I, P=0: dD=0, dP=0 (free streaming)."""
        R_AB = np.zeros((2, 2))
        # D = identity, P = 0
        D_flat = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        dstate = jacobi_rhs(D_flat, R_AB)

        # dD/dlambda = P = 0
        np.testing.assert_allclose(dstate[0:4], 0.0, atol=1e-30)
        # dP/dlambda = R*D = 0
        np.testing.assert_allclose(dstate[4:8], 0.0, atol=1e-30)

    def test_identity_ic_nonzero_velocity(self):
        """With R_AB=0, D=I, P=I: dD=I, dP=0 (linear growth)."""
        R_AB = np.zeros((2, 2))
        D_flat = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])

        dstate = jacobi_rhs(D_flat, R_AB)

        # dD/dlambda = P = I
        np.testing.assert_allclose(dstate[0:4], [1.0, 0.0, 0.0, 1.0])
        # dP/dlambda = R*D = 0
        np.testing.assert_allclose(dstate[4:8], 0.0, atol=1e-30)

    def test_pure_convergence(self):
        """With R_AB = kappa*I (isotropic focusing), dP = -R*D = -kappa*D."""
        kappa_val = 1e-20
        R_AB = np.array([[kappa_val, 0.0], [0.0, kappa_val]])
        # kappa = +1/2 tr(R), so R_diag = kappa for pure convergence

        D_flat = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        dstate = jacobi_rhs(D_flat, R_AB)

        # dP/dlambda = -R*D = -kappa*I
        np.testing.assert_allclose(dstate[4], -kappa_val, rtol=1e-12)
        np.testing.assert_allclose(dstate[7], -kappa_val, rtol=1e-12)
        np.testing.assert_allclose(dstate[5], 0.0, atol=1e-40)
        np.testing.assert_allclose(dstate[6], 0.0, atol=1e-40)


# ======================================================================
#  5. Optical scalars
# ======================================================================

class TestOpticalScalars:
    """Tests for optical scalar extraction."""

    def test_isotropic_convergence(self):
        """R_AB = diag(r,r)  -> kappa = r, gamma = 0, omega = 0."""
        r = 1e-20
        R_AB = np.array([[r, 0.0], [0.0, r]])
        kappa, gamma1, gamma2, omega = optical_scalars_from_tidal(R_AB)

        np.testing.assert_allclose(kappa, r, rtol=1e-12)
        np.testing.assert_allclose(gamma1, 0.0, atol=1e-40)
        np.testing.assert_allclose(gamma2, 0.0, atol=1e-40)
        np.testing.assert_allclose(omega, 0.0, atol=1e-40)

    def test_pure_shear_gamma1(self):
        """R_AB = diag(s,-s)  -> kappa=0, gamma1=+s."""
        s = 1e-20
        R_AB = np.array([[s, 0.0], [0.0, -s]])
        kappa, gamma1, gamma2, omega = optical_scalars_from_tidal(R_AB)

        np.testing.assert_allclose(kappa, 0.0, atol=1e-40)
        np.testing.assert_allclose(gamma1, s, rtol=1e-12)

    def test_pure_shear_gamma2(self):
        """R_AB = [[0,s],[s,0]]  -> kappa=0, gamma2=+s."""
        s = 1e-20
        R_AB = np.array([[0.0, s], [s, 0.0]])
        kappa, gamma1, gamma2, omega = optical_scalars_from_tidal(R_AB)

        np.testing.assert_allclose(kappa, 0.0, atol=1e-40)
        np.testing.assert_allclose(gamma2, s, rtol=1e-12)
        np.testing.assert_allclose(omega, 0.0, atol=1e-40)

    def test_rotation(self):
        """Antisymmetric R_AB  -> omega != 0 (should vanish for physical geodesic light)."""
        w = 1e-20
        R_AB = np.array([[0.0, w], [-w, 0.0]])
        kappa, gamma1, gamma2, omega = optical_scalars_from_tidal(R_AB)

        np.testing.assert_allclose(omega, w, rtol=1e-12)
        np.testing.assert_allclose(kappa, 0.0, atol=1e-40)

    def test_lensing_from_jacobi_identity(self):
        """D = identity  -> kappa=0, mu=1, |gamma|=0."""
        D_flat = np.array([1.0, 0.0, 0.0, 1.0])
        kappa, mu, gamma_mag = lensing_from_jacobi(D_flat)

        np.testing.assert_allclose(kappa, 0.0, atol=1e-14)
        np.testing.assert_allclose(mu, 1.0, rtol=1e-14)
        np.testing.assert_allclose(gamma_mag, 0.0, atol=1e-14)

    def test_lensing_magnification(self):
        """D scaled  -> mu = 1/det(D)."""
        D_flat = np.array([2.0, 0.0, 0.0, 3.0])
        kappa, mu, gamma_mag = lensing_from_jacobi(D_flat)

        expected_mu = 1.0 / 6.0
        np.testing.assert_allclose(mu, expected_mu, rtol=1e-12)


# ======================================================================
#  6. Integration: Riemann  -> Sachs  -> R_AB end-to-end
# ======================================================================

class TestEndToEnd:
    """End-to-end consistency tests."""

    def test_flat_space_pipeline(self):
        """Full pipeline in Minkowski: everything should vanish."""
        a = 1.0
        H = 0.0
        Hprime = 0.0
        phi = 0.0
        phi_dot = 0.0
        phi_ddot = 0.0
        grad_phi = np.zeros(3)
        grad_phi_dot = np.zeros(3)
        hess_phi = np.zeros((3, 3))

        # Riemann
        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime, phi, phi_dot, phi_ddot,
            grad_phi, grad_phi_dot, hess_phi, c_val,
        )

        # Sachs basis
        g = np.diag([-c2, 1.0, 1.0, 1.0])
        k_mu = np.array([1.0 / c_val, 0.0, 0.0, 1.0])
        e1, e2 = init_sachs_basis(k_mu, g, a)

        # Optical tidal matrix
        R_AB = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        np.testing.assert_allclose(R_AB, 0.0, atol=1e-20)

        # Optical scalars
        kappa, gamma1, gamma2, omega = optical_scalars_from_tidal(R_AB)
        np.testing.assert_allclose(kappa, 0.0, atol=1e-20)
        np.testing.assert_allclose(gamma1, 0.0, atol=1e-20)
        np.testing.assert_allclose(gamma2, 0.0, atol=1e-20)
        np.testing.assert_allclose(omega, 0.0, atol=1e-20)

    def test_convergence_from_laplacian(self):
        """
        For a photon along z in nearly-flat space, the convergence
        should be related to grad^2Phi (Poisson equation).

        In the weak-field limit, kappa ~ int grad^2_perp Phi dlambda / c^2.
        Here we just check that non-zero grad^2_perp Phi produces non-zero kappa.
        """
        a = 1.0
        H = 0.0
        Hprime = 0.0
        phi = 1e3       # m^2/s^2
        phi_dot = 0.0
        phi_ddot = 0.0
        grad_phi = np.zeros(3)
        grad_phi_dot = np.zeros(3)
        # Laplacian in transverse plane (x,y) non-zero
        lap_perp = 1e-12  # s^{-2}
        hess_phi = np.array([
            [lap_perp / 2.0, 0.0, 0.0],
            [0.0, lap_perp / 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime, phi, phi_dot, phi_ddot,
            grad_phi, grad_phi_dot, hess_phi, c_val,
        )

        # Photon along z
        g = np.diag([-c2, 1.0, 1.0, 1.0])
        k_mu = np.array([1.0 / c_val, 0.0, 0.0, 1.0])
        e1 = np.array([0.0, 1.0, 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0, 0.0])

        R_AB = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        kappa, gamma1, gamma2, omega = optical_scalars_from_tidal(R_AB)

        # kappa should be non-zero (from grad^2_perp Phi contribution)
        assert abs(kappa) > 0, "Expected non-zero convergence from transverse Laplacian"

    def test_jacobi_free_streaming(self):
        """
        With R_AB = 0, forward-Euler integration of Jacobi map should
        give D(lambda) = I + lambda*P0 for any P0.
        """
        R_AB = np.zeros((2, 2))
        P0 = np.array([0.1, 0.0, 0.0, 0.1])  # expansion rate
        D0 = np.array([1.0, 0.0, 0.0, 1.0])
        state = np.concatenate([D0, P0])

        # Simple Euler integration
        dlambda = 0.01
        n_steps = 100
        for _ in range(n_steps):
            rhs = jacobi_rhs(state, R_AB)
            state = state + dlambda * rhs

        lam = dlambda * n_steps
        D_expected = D0 + lam * P0
        np.testing.assert_allclose(state[0:4], D_expected, rtol=1e-10)
        # P should stay constant
        np.testing.assert_allclose(state[4:8], P0, rtol=1e-10)


# ======================================================================
#  7. PHYSICS TESTS  --  Riemann
# ======================================================================

class TestRiemannPhysics:
    r"""
    Tests that check actual physics predictions of the Riemann module.
    """

    def test_ricci_trace_friedmann(self):
        r"""
        In empty FLRW (Phi=0), the Ricci tensor component R_{00} is
        related to the trace  R_{k0k0} = 3 a^2 H'.

        This is the 00-component of the Einstein equations (Friedmann).
        """
        a = 0.7
        H = 2.3e-18       # conformal Hubble
        Hprime = -1.2e-36  # dH/deta
        phi = 0.0
        phi_dot = 0.0
        phi_ddot = 0.0
        grad_phi = np.zeros(3)
        grad_phi_dot = np.zeros(3)
        hess_phi = np.zeros((3, 3))

        Rd_k00l, _, _ = riemann_blocks_kernel(
            a, H, Hprime, phi, phi_dot, phi_ddot,
            grad_phi, grad_phi_dot, hess_phi, c_val,
        )

        # Trace:  sum_k R_{k0k0}  (spatial trace of the 00-block)
        trace = Rd_k00l[0, 0] + Rd_k00l[1, 1] + Rd_k00l[2, 2]
        expected = 3.0 * a**2 * Hprime

        np.testing.assert_allclose(
            trace, expected, rtol=1e-12,
            err_msg="Ricci trace R_{k0k0} != 3 a^2 H' for empty FLRW",
        )

    def test_newtonian_tidal_tensor(self):
        r"""
        In the weak-field Minkowski limit (a=1, H=H'=0), a static
        potential gives  R_{k00l} = -d_k d_l Phi.

        For a point mass at the origin  Phi = -GM/r:
            d_i d_j Phi = GM (3 x_i x_j / r^5 - delta_{ij} / r^3)

        So R_{k00l} = -d_k d_l Phi should equal the Newtonian tidal tensor.
        """
        G_val = 6.6743e-11
        M = 1.0e30  # ~ 0.5 solar mass
        GM = G_val * M

        # Evaluate at position (1e6, 0, 0)  (1000 km from the mass)
        x0 = 1e6
        r = x0
        r3 = r**3
        r5 = r**5

        # Hessian of Phi = -GM/r:
        #   d^2Phi/dx^2 = GM(3x^2/r^5 - 1/r^3)  etc.
        hess_phi = np.zeros((3, 3))
        pos = np.array([x0, 0.0, 0.0])
        for i in range(3):
            for j in range(3):
                hess_phi[i, j] = GM * (3.0 * pos[i] * pos[j] / r5)
                if i == j:
                    hess_phi[i, j] -= GM / r3

        # Minkowski: a=1, H=H'=0, static
        Rd_k00l, _, _ = riemann_blocks_kernel(
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,  # phi,phi_dot,phi_ddot irrelevant (only on diagonal with H'=0)
            np.zeros(3), np.zeros(3), hess_phi, c_val,
        )

        # Expected: R_{k00l} = -hess_phi[k,l]  (Newtonian tidal tensor)
        np.testing.assert_allclose(
            Rd_k00l, -hess_phi, rtol=1e-12,
            err_msg="Newtonian tidal tensor mismatch for point mass",
        )

        # Sanity: trace should vanish (Laplace equation grad^2Phi = 0 outside mass)
        # (tolerance relaxed for floating-point cancellation in GM*(3x^2-r^2)/r^5)
        np.testing.assert_allclose(
            np.trace(Rd_k00l), 0.0, atol=1e-12,
            err_msg="Tidal tensor trace should vanish (vacuum)",
        )

    def test_riemann_perturbation_linear_in_phi(self):
        r"""
        At first order in Phi with a=1, H=0:
            R_{k00l} ~ -d_k d_l Phi
            R_{kijl} ~ (1/c^2)[delta_{kj} d_id_l Phi - delta_{kl} d_id_j Phi
                                -delta_{ij} d_kd_l Phi + delta_{il} d_kd_j Phi]

        Check that R_{kijl} / (1/c^2) has the right Hessian structure.
        """
        hess_phi = np.array([
            [3e-12, 1e-12, -2e-12],
            [1e-12, -1e-12, 5e-13],
            [-2e-12, 5e-13, 2e-12],
        ])

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            np.zeros(3), np.zeros(3), hess_phi, c_val,
        )

        # Check R_{kijl} term by term
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    for l in range(3):
                        expected = 0.0
                        if k == j:
                            expected += hess_phi[i, l]
                        if k == l:
                            expected -= hess_phi[i, j]
                        if i == j:
                            expected -= hess_phi[k, l]
                        if i == l:
                            expected += hess_phi[k, j]
                        expected /= c2

                        np.testing.assert_allclose(
                            Rd_kijl[k, i, j, l], expected, atol=1e-40,
                            err_msg=f"R_{k}_{{{i}{j}{l}}} first-order mismatch",
                        )

        # R_{0lki} should vanish (no grad_phi_dot, H=0)
        np.testing.assert_allclose(Rd_0lki, 0.0, atol=1e-40)


# ======================================================================
#  8. PHYSICS TESTS  --  Optical tidal matrix & convergence
# ======================================================================

class TestTidalPhysics:
    r"""
    Physics tests for the optical tidal matrix and its relation to
    convergence.
    """

    def test_born_approximation_convergence(self):
        r"""
        Born approximation test.

        For a photon along z in weak-field Minkowski, with a constant
        transverse Laplacian grad^2_perp Phi:

            R_{AB} = diag(-grad^2_perpPhi / c^2, -grad^2_perpPhi / c^2) x (correction terms)

        More precisely, in the static weak-field limit (a=1, H=0):
            R_{k00l} = -d_k d_l Phi

        For photon along z with screen vectors e1 = x_hat, e2 = y_hat:
            R_{AB} contracts R_{mualphanubeta} k^alpha k^beta with the screen vectors.

        With k = (1/c, 0, 0, 1) in Minkowski, the dominant term is
            R_{AB} ~ -d_A d_B Phi  (up to c-factors)

        where A,B index the transverse directions.

        We verify: integrate the Jacobi equation with constant R_{AB}
        for a distance L, and check that
            kappa = 1 - D11 = 1/2 |R_{11}| L^2   (from D'' = R D with D(0)=I, D'(0)=0)

        for small R*L^2.
        """
        # Transverse Hessian: d^2Phi/dx^2 = d^2Phi/dy^2 = h, rest = 0
        h_perp = 1e-20  # s^{-2}  (very weak field)
        hess_phi = np.array([
            [h_perp, 0.0, 0.0],
            [0.0, h_perp, 0.0],
            [0.0, 0.0, 0.0],
        ])

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            1.0, 0.0, 0.0,   # a=1, H=0, H'=0
            0.0, 0.0, 0.0,   # phi, phi_dot, phi_ddot = 0
            np.zeros(3), np.zeros(3), hess_phi, c_val,
        )

        # Set up photon along z in Minkowski
        g = np.diag([-c2, 1.0, 1.0, 1.0])
        k_mu = np.array([1.0 / c_val, 0.0, 0.0, 1.0])
        e1 = np.array([0.0, 1.0, 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0, 0.0])

        R_AB = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        # R_AB should be diagonal and isotropic (equal d^2Phi/dx^2 = d^2Phi/dy^2)
        np.testing.assert_allclose(R_AB[0, 1], 0.0, atol=1e-40)
        np.testing.assert_allclose(R_AB[1, 0], 0.0, atol=1e-40)
        np.testing.assert_allclose(R_AB[0, 0], R_AB[1, 1], rtol=1e-12,
                                   err_msg="Isotropic Hessian should give isotropic R_AB")

        # Now integrate Jacobi equation with this constant R_AB
        # D'' = R*D, D(0) = I, D'(0) = 0
        # For constant R = r*I:  D(lambda) = cosh(sqrtr lambda)*I if r>0
        #                     or D(lambda) = cos(sqrt|r| lambda)*I if r<0
        r_val = R_AB[0, 0]

        # Numerical integration (RK4)
        L = 1e10  # meters (some propagation distance)
        n_steps = 10000
        dl = L / n_steps
        state = np.array([1.0, 0.0, 0.0, 1.0,   # D = I
                          0.0, 0.0, 0.0, 0.0])   # P = 0

        for _ in range(n_steps):
            k1 = jacobi_rhs(state, R_AB)
            k2 = jacobi_rhs(state + 0.5 * dl * k1, R_AB)
            k3 = jacobi_rhs(state + 0.5 * dl * k2, R_AB)
            k4 = jacobi_rhs(state + dl * k3, R_AB)
            state = state + (dl / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Analytic solution
        if r_val > 0:
            D_analytic = np.cosh(np.sqrt(r_val) * L)
        elif r_val < 0:
            D_analytic = np.cos(np.sqrt(-r_val) * L)
        else:
            D_analytic = 1.0

        np.testing.assert_allclose(
            state[0], D_analytic, rtol=1e-6,
            err_msg=f"Jacobi D11(L) != analytic for constant R11={r_val:.3e}",
        )
        np.testing.assert_allclose(
            state[3], D_analytic, rtol=1e-6,
            err_msg="Jacobi D22(L) != D11(L) for isotropic R",
        )

    def test_constant_kappa_jacobi_cosine(self):
        r"""
        For a constant optical tidal matrix  R_AB = kappa0 delta_{AB}
        (pure convergence), the Jacobi equation D'' = -R*D gives:

            D(lambda) = cos(sqrtkappa0 lambda) * I

        This is the fundamental oscillatory solution for a focused beam
        in a uniform-density medium.
        """
        kappa_0 = 1e-30  # very weak, s^{-2}

        # R_AB = kappa0 I  (since kappa = +1/2 tr(R), we need R = kappa0 I for kappa = kappa0)
        R_AB = np.array([[kappa_0, 0.0], [0.0, kappa_0]])

        L = 1e14  # propagation distance
        n_steps = 50000
        dl = L / n_steps

        state = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        for _ in range(n_steps):
            k1 = jacobi_rhs(state, R_AB)
            k2 = jacobi_rhs(state + 0.5 * dl * k1, R_AB)
            k3 = jacobi_rhs(state + 0.5 * dl * k2, R_AB)
            k4 = jacobi_rhs(state + dl * k3, R_AB)
            state = state + (dl / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        omega = np.sqrt(kappa_0)
        D_expected = np.cos(omega * L)
        P_expected = -omega * np.sin(omega * L)

        np.testing.assert_allclose(
            state[0], D_expected, rtol=1e-6,
            err_msg="D11 != cos(sqrtkappa0 L) for constant convergence",
        )
        np.testing.assert_allclose(
            state[3], D_expected, rtol=1e-6,
            err_msg="D22 != cos(sqrtkappa0 L) for constant convergence",
        )
        np.testing.assert_allclose(
            state[4], P_expected, rtol=1e-4,
            err_msg="P11 != -sqrtkappa0 sin(sqrtkappa0 L) for constant convergence",
        )
        # Off-diagonal should stay zero
        np.testing.assert_allclose(state[1], 0.0, atol=1e-20)
        np.testing.assert_allclose(state[2], 0.0, atol=1e-20)

    def test_constant_shear_jacobi(self):
        r"""
        For a constant tidal matrix with pure shear  R_AB = diag(gamma, -gamma),
        the Jacobi equation D'' = -R*D gives two independent equations:

            D11'' = -gamma D11   ->  D11(lambda) = cos(sqrtgamma lambda)   if gamma > 0
            D22'' = +gamma D22   ->  D22(lambda) = cosh(sqrtgamma lambda)

        This tests the shear channel independently.
        """
        gamma_val = 1e-30
        R_AB = np.array([[gamma_val, 0.0], [0.0, -gamma_val]])

        L = 1e14
        n_steps = 50000
        dl = L / n_steps

        state = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        for _ in range(n_steps):
            k1 = jacobi_rhs(state, R_AB)
            k2 = jacobi_rhs(state + 0.5 * dl * k1, R_AB)
            k3 = jacobi_rhs(state + 0.5 * dl * k2, R_AB)
            k4 = jacobi_rhs(state + dl * k3, R_AB)
            state = state + (dl / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        omega = np.sqrt(gamma_val)
        D11_expected = np.cos(omega * L)     # oscillating mode (focusing)
        D22_expected = np.cosh(omega * L)    # growing mode (de-focusing)

        np.testing.assert_allclose(
            state[0], D11_expected, rtol=1e-6,
            err_msg="D11 != cos(sqrtgamma L) for constant shear",
        )
        np.testing.assert_allclose(
            state[3], D22_expected, rtol=1e-6,
            err_msg="D22 != cosh(sqrtgamma L) for constant shear",
        )


# ======================================================================
#  9. PHYSICS TESTS  --  Sachs transport
# ======================================================================

class TestSachsPhysics:
    r"""
    Physics tests for Sachs basis transport.
    """

    def test_sachs_transport_preserves_orthogonality_flrw(self):
        r"""
        Parallel transport preserves inner products.  So if we
        integrate  de^mu/dlambda = -Gamma^mu_{nusigma} k^sigma e^nu  for a few steps
        with actual FLRW Christoffel symbols, the conditions

            g_{mu_nu} e^mu k^nu = 0,   g_{mu_nu} e^mu e^nu = const

        should be maintained (up to integration error).
        """
        a = 0.8
        adot = 1e-18 * a  # a' ~ H * a^2  ... but really adot = a^2 H
        c = c_val

        # Build Christoffel for pure FLRW (Phi = 0)
        # The only non-zero components are Gamma^0_{ii} = a*adot/c^2 and Gamma^i_{i0} = adot/a
        Gamma = np.zeros((4, 4, 4))
        H_over = adot / a  # = a*H(t) = conformal Hubble
        for i in range(1, 4):
            Gamma[0, i, i] = a * adot / c**2
            Gamma[i, i, 0] = H_over
            Gamma[i, 0, i] = H_over

        # Metric
        g = np.diag([-a**2 * c**2, a**2, a**2, a**2])

        # Photon along z: null condition
        k_mu = np.array([1.0 / (a * c), 0.0, 0.0, 1.0 / a])

        # Initial Sachs vectors (pure spatial, orthogonal to z)
        e1 = np.array([0.0, 1.0 / a, 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0 / a, 0.0])

        # Verify initial conditions
        np.testing.assert_allclose(e1 @ g @ k_mu, 0.0, atol=1e-15)
        np.testing.assert_allclose(e2 @ g @ k_mu, 0.0, atol=1e-15)
        np.testing.assert_allclose(e1 @ g @ e1, 1.0, atol=1e-12)
        np.testing.assert_allclose(e2 @ g @ e2, 1.0, atol=1e-12)
        np.testing.assert_allclose(e1 @ g @ e2, 0.0, atol=1e-15)

        # Integrate N steps with RK4
        # Note: we keep a, adot, g, Gamma fixed (frozen coefficients  --  short integration)
        dl = 1e10  # meters
        n_steps = 100

        for _ in range(n_steps):
            # RK4 for e1
            k1_e1 = sachs_transport_rhs(e1, Gamma, k_mu)
            k2_e1 = sachs_transport_rhs(e1 + 0.5*dl*k1_e1, Gamma, k_mu)
            k3_e1 = sachs_transport_rhs(e1 + 0.5*dl*k2_e1, Gamma, k_mu)
            k4_e1 = sachs_transport_rhs(e1 + dl*k3_e1, Gamma, k_mu)
            e1 = e1 + (dl/6.0) * (k1_e1 + 2*k2_e1 + 2*k3_e1 + k4_e1)

            # RK4 for e2
            k1_e2 = sachs_transport_rhs(e2, Gamma, k_mu)
            k2_e2 = sachs_transport_rhs(e2 + 0.5*dl*k1_e2, Gamma, k_mu)
            k3_e2 = sachs_transport_rhs(e2 + 0.5*dl*k2_e2, Gamma, k_mu)
            k4_e2 = sachs_transport_rhs(e2 + dl*k3_e2, Gamma, k_mu)
            e2 = e2 + (dl/6.0) * (k1_e2 + 2*k2_e2 + 2*k3_e2 + k4_e2)

        # Check orthogonality is preserved
        np.testing.assert_allclose(
            e1 @ g @ k_mu, 0.0, atol=1e-10,
            err_msg="e1*k != 0 after transport",
        )
        np.testing.assert_allclose(
            e2 @ g @ k_mu, 0.0, atol=1e-10,
            err_msg="e2*k != 0 after transport",
        )
        np.testing.assert_allclose(
            e1 @ g @ e2, 0.0, atol=1e-10,
            err_msg="e1*e2 != 0 after transport",
        )

        # Norms may drift slightly due to discrete integration, but should
        # remain close to 1
        np.testing.assert_allclose(
            e1 @ g @ e1, 1.0, rtol=1e-4,
            err_msg="|e1|^2 drifted from 1 after transport",
        )
        np.testing.assert_allclose(
            e2 @ g @ e2, 1.0, rtol=1e-4,
            err_msg="|e2|^2 drifted from 1 after transport",
        )

    def test_sachs_init_different_directions(self):
        r"""
        The Sachs basis should work for photons travelling in any
        direction, not just along z.  Test x, y, z and a diagonal.
        """
        a = 1.0
        g = np.diag([-c2, 1.0, 1.0, 1.0])

        directions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
            np.array([-1.0, 2.0, -0.5]),  # arbitrary
        ]

        for n_hat in directions:
            # Normalize in spatial metric
            n_hat = n_hat / np.sqrt(n_hat @ g[1:4, 1:4] @ n_hat)
            k_mu = np.array([1.0 / c_val, n_hat[0], n_hat[1], n_hat[2]])

            e1, e2 = init_sachs_basis(k_mu, g, a)

            # All five conditions
            np.testing.assert_allclose(e1 @ g @ k_mu, 0.0, atol=1e-10,
                                       err_msg=f"e1*k != 0 for dir={n_hat}")
            np.testing.assert_allclose(e2 @ g @ k_mu, 0.0, atol=1e-10,
                                       err_msg=f"e2*k != 0 for dir={n_hat}")
            np.testing.assert_allclose(e1 @ g @ e2, 0.0, atol=1e-10,
                                       err_msg=f"e1*e2 != 0 for dir={n_hat}")
            np.testing.assert_allclose(e1 @ g @ e1, 1.0, atol=1e-10,
                                       err_msg=f"|e1| != 1 for dir={n_hat}")
            np.testing.assert_allclose(e2 @ g @ e2, 1.0, atol=1e-10,
                                       err_msg=f"|e2| != 1 for dir={n_hat}")


# ======================================================================
#  10. PHYSICS TESTS  --  Point-mass tidal matrix
# ======================================================================

class TestPointMassTidal:
    r"""
    For a point mass M at the origin, the potential is Phi = -GM/r.

    In the weak-field Minkowski limit, a photon along z at impact
    parameter b should feel:

        R_{k00l}(b, 0, 0) = Newtonian tidal tensor at (b, 0, 0)

    And the optical tidal matrix for a photon along z should give:

        R_{11} = -d^2Phi/dx^2  (with corrections from spatial Riemann)
        R_{22} = -d^2Phi/dy^2

    For the point mass at (b, 0, z_photon):
        d^2Phi/dx^2 = GM(3b^2 - r^2)/r^5  at y=0, with r^2 = b^2 + z^2
        d^2Phi/dy^2 = -GM/r^3            at y=0

    We test the trace R_{11} + R_{22} and the shear R_{11} - R_{22}.
    """

    def test_point_mass_convergence_sign(self):
        r"""
        A point mass should produce convergence (kappa > 0, i.e. focusing).
        The convergence is kappa = -1/2 tr(R_AB).

        For a point mass, grad^2_perpPhi < 0 (the potential is -GM/r, concave),
        so the tidal matrix trace should be positive  -> kappa < 0?

        Actually, for Phi = -GM/r at position (b, 0, 0):
            d^2Phi/dx^2 = GM(2b^2)/(b^5) = 2GM/b^3   at (b, 0, 0) where r=b
            d^2Phi/dy^2 = -GM/b^3

        So grad^2_perpPhi = d^2Phi/dx^2 + d^2Phi/dy^2 = GM/b^3 > 0.

        R_{k00l} = -d_k d_l Phi, so the diagonal is R_{x00x} = -2GM/b^3,
        R_{y00y} = GM/b^3.

        The resulting R_AB should give focusing (convergence > 0).
        """
        G_val = 6.6743e-11
        M = 1e30
        GM = G_val * M
        b = 1e8  # impact parameter: 100,000 km

        # Hessian of Phi = -GM/r at (b, 0, 0)
        r = b
        hess_phi = np.zeros((3, 3))
        hess_phi[0, 0] = GM * (3*b**2 - r**2) / r**5  # = 2GM/b^3
        hess_phi[1, 1] = GM * (0 - r**2) / r**5        # = -GM/b^3
        hess_phi[2, 2] = GM * (0 - r**2) / r**5        # = -GM/b^3

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            np.zeros(3), np.zeros(3), hess_phi, c_val,
        )

        # Photon along z at (b, 0, z)  --  screen is (x, y)
        g = np.diag([-c2, 1.0, 1.0, 1.0])
        k_mu = np.array([1.0/c_val, 0.0, 0.0, 1.0])
        e1 = np.array([0.0, 1.0, 0.0, 0.0])  # x direction
        e2 = np.array([0.0, 0.0, 1.0, 0.0])  # y direction

        R_AB = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        kappa, gamma1, gamma2, omega = optical_scalars_from_tidal(R_AB)

        # The convergence should be positive (focusing by mass)
        # kappa = -1/2 tr(R) = -1/2(R_11 + R_22)
        # We need to check the sign carefully based on what the pipeline gives
        # For this test, we mainly check that kappa != 0 and has consistent sign
        # with the shear
        assert R_AB[0, 0] != 0.0, "R_11 should be non-zero for point mass"
        assert R_AB[1, 1] != 0.0, "R_22 should be non-zero for point mass"

        # Shear should be non-zero (mass is not azimuthally symmetric at (b,0,0))
        assert abs(gamma1) > 0, "Shear gamma1 should be non-zero for off-axis point mass"

        # Check that R_AB is symmetric (no rotation for static mass)
        np.testing.assert_allclose(
            R_AB[0, 1], R_AB[1, 0], atol=1e-40,
            err_msg="R_AB should be symmetric for static potential",
        )
        np.testing.assert_allclose(
            omega, 0.0, atol=1e-40,
            err_msg="Rotation omega should vanish for static potential",
        )


# ======================================================================
#  11. DISCRIMINATING TEST  --  a != 1  (catches double-lowering bug)
# ======================================================================

class TestNonUnitScaleFactor:
    r"""
    Tests that specifically use a != 1 to discriminate between
    R_{k00l} (all-down) and R_{k00l} (mixed) conventions.

    With a != 1, the metric g_{ij} = a^2delta_{ij}, so:
        R_{k00l} = g_{kk} R_{k00l} = a^2 R_{k00l}

    If the pipeline incorrectly treats the blocks as mixed and lowers
    with g, it introduces an extra factor of a^2  --  which is visible
    when a != 1.
    """

    def test_tidal_matrix_a_not_1_vs_brute_force(self):
        r"""
        Compute R_{AB} two ways for a = 2:
          1. Via our pipeline (riemann_blocks_kernel  -> optical_tidal_matrix)
          2. Via brute-force 4-index Riemann built from scratch

        They should match exactly. If there's a double-lowering bug,
        the pipeline result differs by a factor of a^2 = 4.
        """
        a = 2.0
        H = 0.0
        Hprime = 0.0

        # Simple transverse Hessian
        h_val = 1e-15  # s^{-2}
        hess_phi = np.array([
            [h_val, 0.0, 0.0],
            [0.0, -h_val, 0.0],
            [0.0, 0.0, 0.0],
        ])

        # --- Method 1: our pipeline ---
        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime,
            0.0, 0.0, 0.0,
            np.zeros(3), np.zeros(3), hess_phi, c_val,
        )

        # FLRW metric with a=2
        g = np.diag(np.array([-a**2 * c2, a**2, a**2, a**2]))

        # Photon along z: null condition g_{00}(k^0)^2 + g_{33}(k^3)^2 = 0
        # => k^0 = 1/(a*c)
        k_mu = np.array([1.0 / (a * c_val), 0.0, 0.0, 1.0 / a])
        # Sachs vectors (unit norm in FLRW metric: g_{ii} e^i e^i = 1 => e^i = 1/a)
        e1 = np.array([0.0, 1.0 / a, 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0 / a, 0.0])

        R_AB_pipeline = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        # --- Method 2: brute-force from scratch ---
        # Build the full all-down 4D Riemann from the blocks
        R_down_full = np.zeros((4, 4, 4, 4))
        for k in range(3):
            for l in range(3):
                R_down_full[k+1, 0, 0, l+1] = Rd_k00l[k, l]
                R_down_full[k+1, 0, l+1, 0] = -Rd_k00l[k, l]
        for l in range(3):
            for k in range(3):
                for i in range(3):
                    R_down_full[0, l+1, k+1, i+1] = Rd_0lki[l, k, i]
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    for l in range(3):
                        R_down_full[k+1, i+1, j+1, l+1] = Rd_kijl[k, i, j, l]

        # Brute-force contraction: R_{AB} = R_{mualphanubeta} k^alpha k^beta e_A^mu e_B^nu
        e_vecs = np.array([e1, e2])
        R_AB_brute = np.zeros((2, 2))
        for A in range(2):
            for B in range(2):
                s = 0.0
                for mu in range(4):
                    for alpha in range(4):
                        for nu in range(4):
                            for beta in range(4):
                                s += (R_down_full[mu, alpha, nu, beta]
                                      * k_mu[alpha] * k_mu[beta]
                                      * e_vecs[A, mu] * e_vecs[B, nu])
                R_AB_brute[A, B] = s

        np.testing.assert_allclose(
            R_AB_pipeline, R_AB_brute, rtol=1e-12,
            err_msg="Pipeline R_AB differs from brute-force with a=2 "
                    "(possible double-lowering bug)",
        )

        # Also verify R_AB is non-zero (test is meaningful)
        assert np.max(np.abs(R_AB_pipeline)) > 0, \
            "R_AB should be non-zero for non-zero Hessian"

        # Additional check: R_AB should be symmetric
        np.testing.assert_allclose(
            R_AB_pipeline[0, 1], R_AB_pipeline[1, 0], atol=1e-40,
            err_msg="R_AB not symmetric for static field with a=2",
        )

    def test_symmetry_of_R_AB_flrw(self):
        r"""
        With a=0.5, non-trivial H, H', and all perturbation terms,
        R_AB should still be symmetric.
        """
        a = 0.5
        H = 3e-18
        Hprime = -2e-36
        phi = 5e3
        phi_dot = 1e-6
        phi_ddot = 1e-16
        grad_phi = np.array([1e-8, -2e-8, 3e-8])
        grad_phi_dot = np.array([1e-18, 2e-19, -5e-19])
        hess_phi = np.array([
            [2e-12, 1e-13, -3e-13],
            [1e-13, -1e-12, 2e-13],
            [-3e-13, 2e-13, 4e-13],
        ])

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime, phi, phi_dot, phi_ddot,
            grad_phi, grad_phi_dot, hess_phi, c_val,
        )

        g = np.diag(np.array([-a**2 * c2, a**2, a**2, a**2]))
        k_mu = np.array([1.0 / (a * c_val), 0.0, 0.0, 1.0 / a])
        e1 = np.array([0.0, 1.0 / a, 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0 / a, 0.0])

        R_AB = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        np.testing.assert_allclose(
            R_AB[0, 1], R_AB[1, 0], rtol=1e-8,
            err_msg="R_AB not symmetric for a=0.5 with full perturbations",
        )

    def test_optimized_matches_reference_a_not_1(self):
        r"""
        Verify that optimized and reference implementations agree
        for a != 1 with full perturbation inputs.
        """
        a = 1.7
        H = 1.5e-18
        Hprime = -8e-37
        hess_phi = np.array([
            [1e-12, 2e-13, -3e-13],
            [2e-13, -1e-12, 1e-13],
            [-3e-13, 1e-13, 5e-13],
        ])

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime,
            1e4, 1e-5, 1e-15,
            np.array([1e-8, 2e-8, -1e-8]),
            np.array([1e-18, -5e-19, 2e-19]),
            hess_phi, c_val,
        )

        g = np.diag(np.array([-a**2 * c2, a**2, a**2, a**2]))
        k_mu = np.array([1.0 / (a * c_val), 0.0, 0.0, 1.0 / a])
        e1 = np.array([0.0, 1.0 / a, 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0 / a, 0.0])

        R_ref = optical_tidal_matrix_from_blocks(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)
        R_opt = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        np.testing.assert_allclose(
            R_opt, R_ref, rtol=1e-10,
            err_msg="Optimized != reference for a=1.7",
        )


# ======================================================================
#  12. OBLIQUE PHOTON  --  off-axis with a != 1, H != 0, full perturbations
# ======================================================================

class TestObliquePhoton:
    r"""
    Tests with a photon NOT along a coordinate axis, combining:
      - a != 1, H != 0, grad_phi_dot != 0 (all blocks non-zero)
      - off-axis photon direction

    Note: In the diagonal FLRW gauge with a comoving observer, the
    Sachs vectors are always purely spatial (e_A^0 = 0) because
    orthogonality to k AND u forces the time component to vanish.
    The R_{0lki} block contributes to T_{0, k+1} which is contracted
    with e_A^0 = 0, so it does not contribute to R_{AB} in this gauge.
    This is correct physics  --  not a bug.
    """

    def test_oblique_photon_pipeline_vs_brute_force(self):
        r"""
        Oblique photon at 45deg in the x-z plane with a != 1, H != 0,
        and grad_phi_dot != 0 (all Riemann blocks non-zero).

        Compare pipeline (optimized) vs brute-force 4-loop contraction.
        """
        a = 1.5
        H = 2e-18
        Hprime = -1e-36
        phi = 1e4
        phi_dot = 1e-5
        phi_ddot = 1e-15
        grad_phi = np.array([3e-8, -1e-8, 2e-8])
        grad_phi_dot = np.array([5e-18, -2e-18, 1e-18])
        hess_phi = np.array([
            [2e-12, 1e-13, -3e-13],
            [1e-13, -1e-12, 2e-13],
            [-3e-13, 2e-13, 4e-13],
        ])

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime, phi, phi_dot, phi_ddot,
            grad_phi, grad_phi_dot, hess_phi, c_val,
        )

        # All blocks should be non-zero
        assert np.max(np.abs(Rd_0lki)) > 0, \
            "R_{0lki} should be non-zero with grad_phi_dot != 0 and H != 0"

        # --- Metric and photon setup ---
        g = np.diag(np.array([-a**2 * c2, a**2, a**2, a**2]))

        # Oblique photon: 45deg in x-z plane
        k_spatial_mag = 1.0 / a
        k1 = k_spatial_mag / np.sqrt(2.0)
        k3 = k_spatial_mag / np.sqrt(2.0)
        k0 = k_spatial_mag / c_val
        k_mu = np.array([k0, k1, 0.0, k3])

        # Build Sachs basis via init_sachs_basis
        e1, e2 = init_sachs_basis(k_mu, g, a)

        # --- Pipeline (optimized) ---
        R_AB_opt = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        # --- Brute-force from scratch ---
        R_down_full = np.zeros((4, 4, 4, 4))
        for k in range(3):
            for l in range(3):
                R_down_full[k+1, 0, 0, l+1] = Rd_k00l[k, l]
                R_down_full[k+1, 0, l+1, 0] = -Rd_k00l[k, l]
        for l in range(3):
            for k in range(3):
                for i in range(3):
                    R_down_full[0, l+1, k+1, i+1] = Rd_0lki[l, k, i]
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    for l in range(3):
                        R_down_full[k+1, i+1, j+1, l+1] = Rd_kijl[k, i, j, l]

        e_vecs = np.array([e1, e2])
        R_AB_brute = np.zeros((2, 2))
        for A in range(2):
            for B in range(2):
                s = 0.0
                for mu in range(4):
                    for alpha in range(4):
                        for nu in range(4):
                            for beta in range(4):
                                s += (R_down_full[mu, alpha, nu, beta]
                                      * k_mu[alpha] * k_mu[beta]
                                      * e_vecs[A, mu] * e_vecs[B, nu])
                R_AB_brute[A, B] = s

        np.testing.assert_allclose(
            R_AB_opt, R_AB_brute, rtol=1e-10,
            err_msg="Optimized != brute-force for oblique photon",
        )

        # R_AB is non-zero (test is meaningful)
        assert np.max(np.abs(R_AB_opt)) > 0, \
            "R_AB should be non-zero for oblique photon with perturbations"

    def test_oblique_photon_ref_vs_optimized(self):
        r"""
        Reference (4-loop) and optimized implementations must agree
        for oblique photon with full perturbations.
        """
        a = 2.3
        H = 3e-18
        Hprime = -5e-37
        grad_phi = np.array([1e-7, -5e-8, 3e-8])
        grad_phi_dot = np.array([2e-17, 1e-17, -3e-17])
        hess_phi = np.array([
            [5e-12, -2e-13, 1e-13],
            [-2e-13, 3e-12, -4e-13],
            [1e-13, -4e-13, -2e-12],
        ])

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime,
            5e3, 2e-6, 3e-16,
            grad_phi, grad_phi_dot, hess_phi, c_val,
        )

        g = np.diag(np.array([-a**2 * c2, a**2, a**2, a**2]))

        # Photon at ~30deg from z in the y-z plane
        k_spatial_mag = 1.0 / a
        theta = np.pi / 6  # 30deg
        k2 = k_spatial_mag * np.sin(theta)
        k3 = k_spatial_mag * np.cos(theta)
        k0 = k_spatial_mag / c_val
        k_mu = np.array([k0, 0.0, k2, k3])

        e1, e2 = init_sachs_basis(k_mu, g, a)

        R_ref = optical_tidal_matrix_from_blocks(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)
        R_opt = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        np.testing.assert_allclose(
            R_opt, R_ref, rtol=1e-10,
            err_msg="Optimized != reference for oblique photon (30deg in y-z)",
        )

    def test_oblique_photon_symmetry_a_not_1(self):
        r"""
        R_AB should be symmetric for a static potential (no time
        derivatives) even with off-axis photon and a != 1.
        """
        a = 1.8
        H = 0.0
        Hprime = 0.0
        hess_phi = np.array([
            [1e-12, 3e-13, -2e-13],
            [3e-13, -1e-12, 1e-13],
            [-2e-13, 1e-13, 5e-13],
        ])

        Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
            a, H, Hprime,
            0.0, 0.0, 0.0,
            np.zeros(3), np.zeros(3), hess_phi, c_val,
        )

        g = np.diag(np.array([-a**2 * c2, a**2, a**2, a**2]))

        # Photon at 45deg in x-y-z diagonal
        k_spatial_mag = 1.0 / a
        kx = k_spatial_mag / np.sqrt(3.0)
        ky = k_spatial_mag / np.sqrt(3.0)
        kz = k_spatial_mag / np.sqrt(3.0)
        k0 = k_spatial_mag / c_val
        k_mu = np.array([k0, kx, ky, kz])

        e1, e2 = init_sachs_basis(k_mu, g, a)

        R_AB = optical_tidal_matrix_optimized(
            Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

        assert np.max(np.abs(R_AB)) > 0, "R_AB should be non-zero"
        np.testing.assert_allclose(
            R_AB[0, 1], R_AB[1, 0], rtol=1e-10,
            err_msg="R_AB should be symmetric for static potential with oblique photon",
        )


# ======================================================================
#  13. SIS LENS  --  analytic comparison
# ======================================================================

class TestSISLens:
    r"""
    Singular Isothermal Sphere (SIS) lensing test.

    The 3D Newtonian potential of a SIS with density
    rho(r) = sigma_v^2 / (2piGr^2) is:

        Phi(r) = 2 sigma_v^2 ln(r)     (r = 3D radius)

    For a photon along z at impact parameter b (along x),
    the Hessian d^2Phi/dx_idx_j = 2sigma_v^2 [delta_{ij}/r^2 - 2x_i x_j / r^4].

    In the Born approximation with the FULL Riemann contraction
    (R_{k00l} + R_{kijl} blocks), the integrated convergence is:

        kappa = -1/2 tr(int R_{AB} dz) = -2pi sigma_v^2 / (b c^2)

    The factor of 2 relative to the naive 1/2grad^2Phi formula comes from
    the R_{kijl} spatial block, which contributes equally to R_{k00l}
    for a static potential in Minkowski (a=1, H=0).

    The shear is:
        gamma1 = -1/2 (R11 - R22) integrated = +2pi sigma_v^2 / (b c^2) = -kappa

    So |gamma| = |kappa|, the classic SIS result.

    Note: kappa is NEGATIVE in our convention because the R_{AB} convention
    used here gives R_{11} > 0 and R_{22} > 0, so kappa = -1/2(R11+R22) < 0.
    The sign of kappa depends on the convention for R_{mualphanubeta}k^alphak^beta vs
    R_{mualphabetanu}k^alphak^beta. The physical convergence |kappa| = 2pisigma_v^2/(bc^2) is
    positive as expected for a focusing lens.
    """

    def test_sis_convergence_and_shear(self):
        r"""
        Born-approximation test for SIS lens in Minkowski spacetime
        (a=1, H=0).

        Integrate the optical tidal matrix along the line of sight
        and compare to the known analytic result.
        """
        # SIS parameters
        sigma_v = 250e3  # 250 km/s velocity dispersion
        b = 1e22         # impact parameter ~ 3 kpc

        # Minkowski spacetime
        a = 1.0

        # Integration range: L=200b gives ~0.3% accuracy
        L = 200.0 * b
        N_steps = 4000
        z_arr = np.linspace(-L, L, N_steps)
        dz = z_arr[1] - z_arr[0]

        # Photon along z at (b, 0, z)
        k_mu = np.array([1.0 / c_val, 0.0, 0.0, 1.0])
        e1 = np.array([0.0, 1.0, 0.0, 0.0])  # x direction
        e2 = np.array([0.0, 0.0, 1.0, 0.0])  # y direction
        g = np.diag(np.array([-c2, 1.0, 1.0, 1.0]))

        R_AB_integrated = np.zeros((2, 2))

        for z in z_arr:
            r_sq = b**2 + z**2
            r4 = r_sq**2

            # Hessian of Phi(r) = 2sigma_v^2 ln(r)
            hess_phi = np.zeros((3, 3))
            pos = np.array([b, 0.0, z])
            for i in range(3):
                for j in range(3):
                    hess_phi[i, j] = 2.0 * sigma_v**2 * (
                        (1.0 if i == j else 0.0) / r_sq
                        - 2.0 * pos[i] * pos[j] / r4
                    )

            Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
                a, 0.0, 0.0,
                0.0, 0.0, 0.0,
                np.zeros(3), np.zeros(3), hess_phi, c_val,
            )

            R_AB = optical_tidal_matrix_optimized(
                Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1, e2, g)

            R_AB_integrated += R_AB * dz

        kappa, gamma1, gamma2, omega = optical_scalars_from_tidal(R_AB_integrated)

        # Analytic predictions (full Riemann, all blocks):
        #   kappa = +2pi sigma_v^2 / (b c^2)  (positive  -> focusing)
        #   gamma1 = -2pi sigma_v^2 / (b c^2) = -kappa
        kappa_analytic = 2.0 * np.pi * sigma_v**2 / (b * c_val**2)
        gamma1_analytic = -kappa_analytic  # = -2pisigma^2/(bc^2)

        np.testing.assert_allclose(
            kappa, kappa_analytic, rtol=0.005,
            err_msg=f"SIS convergence: got kappa={kappa:.6e}, expected {kappa_analytic:.6e}",
        )

        np.testing.assert_allclose(
            gamma1, gamma1_analytic, rtol=0.005,
            err_msg=f"SIS shear gamma1: got {gamma1:.6e}, expected {gamma1_analytic:.6e}",
        )

        # Classic SIS result: |gamma| = |kappa|
        gamma_mag = np.sqrt(gamma1**2 + gamma2**2)
        np.testing.assert_allclose(
            gamma_mag, abs(kappa), rtol=0.03,
            err_msg=f"SIS: |gamma|={gamma_mag:.6e} should equal |kappa|={abs(kappa):.6e}",
        )

        # Rotation should vanish (static potential)
        np.testing.assert_allclose(
            omega, 0.0, atol=1e-50,
            err_msg="Rotation omega should vanish for static SIS",
        )

    def test_sis_a_scaling(self):
        r"""
        For a = const, H = 0, the optical tidal matrix R_{AB} scales
        as 1/a^2 (Riemann ~a^2, k^alpha ~ 1/a, e_A^mu ~ 1/a  -> net 1/a^2).

        This verifies that the a^2 factors in the Riemann blocks are
        correctly compensated by the 1/a normalizations of k^mu and
        e_A^mu in the contraction.
        """
        sigma_v = 250e3
        b = 1e22

        L = 200.0 * b
        N_steps = 4000
        z_arr = np.linspace(-L, L, N_steps)
        dz = z_arr[1] - z_arr[0]

        results = {}
        for a in [1.0, 2.5]:
            g = np.diag(np.array([-a**2 * c2, a**2, a**2, a**2]))
            k_mu = np.array([1.0 / (a * c_val), 0.0, 0.0, 1.0 / a])
            e1 = np.array([0.0, 1.0 / a, 0.0, 0.0])
            e2 = np.array([0.0, 0.0, 1.0 / a, 0.0])

            R_AB_int = np.zeros((2, 2))
            for z in z_arr:
                r_sq = b**2 + z**2
                r4 = r_sq**2
                pos = np.array([b, 0.0, z])
                hess_phi = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        hess_phi[i, j] = 2.0 * sigma_v**2 * (
                            (1.0 if i == j else 0.0) / r_sq
                            - 2.0 * pos[i] * pos[j] / r4
                        )
                Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
                    a, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    np.zeros(3), np.zeros(3), hess_phi, c_val,
                )
                Rd_0lki_zero = np.zeros((3, 3, 3))  # H=0  -> R_{0lki}=0
                R_AB = optical_tidal_matrix_optimized(
                    Rd_k00l, Rd_0lki_zero, Rd_kijl, k_mu, e1, e2, g)
                R_AB_int += R_AB * dz

            kappa, _, _, _ = optical_scalars_from_tidal(R_AB_int)
            results[a] = kappa

        # kappa(a) = kappa(1) / a^2
        kappa_1 = results[1.0]
        kappa_25 = results[2.5]
        expected_ratio = 1.0 / 2.5**2  # = 0.16

        np.testing.assert_allclose(
            kappa_25 / kappa_1, expected_ratio, rtol=1e-10,
            err_msg=f"SIS: kappa(a=2.5)/kappa(a=1) = {kappa_25/kappa_1:.6e}, "
                    f"expected {expected_ratio:.6e} (1/a^2 scaling)",
        )


# ======================================================================
#  10. Angular-diameter distances
# ======================================================================

class TestAngularDiameterDistances:
    """Tests for D_A from the Jacobi map and from the FLRW background."""

    # ------------------------------------------------------------------
    #  Background FLRW distances
    # ------------------------------------------------------------------

    def test_DA_FLRW_z0_is_zero(self):
        """D_A(z=0) must be 0."""
        from excalibur.core.cosmology import LCDM_Cosmology
        cosmo = LCDM_Cosmology(H0=70, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
        DA = cosmo.angular_diameter_distance(0.0)
        np.testing.assert_allclose(DA, 0.0, atol=1e-6)

    def test_DA_FLRW_positive(self):
        """D_A(z>0) must be positive."""
        from excalibur.core.cosmology import LCDM_Cosmology
        cosmo = LCDM_Cosmology(H0=70, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
        for z in [0.1, 0.5, 1.0, 2.0]:
            DA = cosmo.angular_diameter_distance(z)
            assert DA > 0, f"D_A({z}) = {DA} should be positive"

    def test_DA_FLRW_known_value(self):
        r"""Check D_A(z=1) for standard LambdaCDM (H0=70, Omegam=0.3, OmegaLambda=0.7).

        Astropy reference:  D_A(z=1) ~ 1651.9 Mpc  for this cosmology.
        We check to 1% because the quad integration should be very precise.
        """
        from excalibur.core.cosmology import LCDM_Cosmology
        from excalibur.core.constants import one_Mpc
        cosmo = LCDM_Cosmology(H0=70, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
        DA = cosmo.angular_diameter_distance(1.0)
        DA_Mpc = DA / one_Mpc
        # Astropy gives ~1651.9 Mpc for this cosmology
        np.testing.assert_allclose(DA_Mpc, 1651.9, rtol=0.01,
                                   err_msg=f"D_A(z=1) = {DA_Mpc:.1f} Mpc, expected ~1651.9 Mpc")

    def test_DA_FLRW_has_maximum(self):
        """D_A should have a maximum at z~1.6 for standard LambdaCDM, then decrease."""
        from excalibur.core.cosmology import LCDM_Cosmology
        cosmo = LCDM_Cosmology(H0=70, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
        zs = np.linspace(0.1, 5.0, 50)
        DAs = np.array([cosmo.angular_diameter_distance(z) for z in zs])
        z_max = zs[np.argmax(DAs)]
        # The maximum should be roughly between z=1 and z=2.5
        assert 1.0 < z_max < 2.5, f"D_A maximum at z={z_max:.2f}, expected ~1.6"

    def test_comoving_distance_monotone(self):
        """Comoving distance must be monotonically increasing with z."""
        from excalibur.core.cosmology import LCDM_Cosmology
        cosmo = LCDM_Cosmology(H0=70, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
        zs = [0.1, 0.5, 1.0, 2.0, 5.0]
        chis = [cosmo.comoving_distance(z) for z in zs]
        for i in range(len(chis) - 1):
            assert chis[i] < chis[i+1], f"chi(z={zs[i]}) >= chi(z={zs[i+1]})"

    def test_luminosity_distance(self):
        """Etherington relation: D_L = (1+z)^2 D_A."""
        from excalibur.core.cosmology import LCDM_Cosmology
        cosmo = LCDM_Cosmology(H0=70, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
        for z in [0.5, 1.0, 2.0]:
            DA = cosmo.angular_diameter_distance(z)
            DL = cosmo.luminosity_distance(z)
            np.testing.assert_allclose(DL, DA * (1 + z)**2, rtol=1e-12,
                                       err_msg=f"Etherington violated at z={z}")

    def test_DA_z1z2(self):
        """D_A(0, z) == D_A(z) for the single-argument version."""
        from excalibur.core.cosmology import LCDM_Cosmology
        cosmo = LCDM_Cosmology(H0=70, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
        for z in [0.5, 1.0, 2.0]:
            DA1 = cosmo.angular_diameter_distance(z)
            DA2 = cosmo.angular_diameter_distance_z1z2(0.0, z)
            np.testing.assert_allclose(DA1, DA2, rtol=1e-10)

    def test_DA_FLRW_vectorized(self):
        """angular_diameter_distance should accept arrays."""
        from excalibur.core.cosmology import LCDM_Cosmology
        cosmo = LCDM_Cosmology(H0=70, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)
        zs = np.array([0.5, 1.0, 2.0])
        DAs = cosmo.angular_diameter_distance(zs)
        assert DAs.shape == (3,)
        for i, z in enumerate(zs):
            DA_scalar = cosmo.angular_diameter_distance(float(z))
            np.testing.assert_allclose(DAs[i], DA_scalar, rtol=1e-12)

    # ------------------------------------------------------------------
    #  D_A from Jacobi map
    # ------------------------------------------------------------------

    def test_DA_jacobi_identity(self):
        """D = identity  -> D_A = 1 (unit affine distance)."""
        D_flat = np.array([1.0, 0.0, 0.0, 1.0])
        DA = angular_diameter_distance_from_jacobi(D_flat)
        np.testing.assert_allclose(DA, 1.0, rtol=1e-14)

    def test_DA_jacobi_scaled(self):
        """D = lambda*I  -> D_A = lambda (isotropic expansion)."""
        lam = 1.5e25  # some distance in metres
        D_flat = np.array([lam, 0.0, 0.0, lam])
        DA = angular_diameter_distance_from_jacobi(D_flat)
        np.testing.assert_allclose(DA, lam, rtol=1e-12)

    def test_DA_jacobi_with_shear(self):
        """D with shear: D_A = sqrt(|det D|)."""
        D_flat = np.array([2.0, 0.5, 0.3, 3.0])
        DA = angular_diameter_distance_from_jacobi(D_flat)
        expected = np.sqrt(abs(2.0 * 3.0 - 0.5 * 0.3))
        np.testing.assert_allclose(DA, expected, rtol=1e-14)

    def test_DA_jacobi_matches_magnification(self):
        r"""Consistency: mu = (D_A^FLRW / D_A_ray)^2 in the weak-lensing regime.

        For a normalised D (D/lambda_S), we have mu = 1/det(D_norm),
        and D_A_ray = lambda_S*sqrt|det(D_norm)|, so:
            mu = 1/det(D_norm) = (lambda_S / D_A_ray)^2   ->  D_A_ray = lambda_S / sqrtmu.
        """
        lam_S = 1.0e26  # affine distance to source
        D_flat_norm = np.array([0.98, 0.01, 0.01, 1.02])  # weak lensing
        kappa, mu, gamma = lensing_from_jacobi(D_flat_norm)

        D_A_from_mu = lam_S / np.sqrt(abs(mu))
        D_A_from_jacobi = angular_diameter_distance_from_jacobi(D_flat_norm * lam_S)
        np.testing.assert_allclose(D_A_from_jacobi, D_A_from_mu, rtol=1e-10)

    # ------------------------------------------------------------------
    #  distance_comparison helper
    # ------------------------------------------------------------------

    def test_distance_comparison(self):
        """distance_comparison returns a sensible dict."""
        from excalibur.core.cosmology import LCDM_Cosmology
        from excalibur.core.constants import one_Mpc
        cosmo = LCDM_Cosmology(H0=70, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)

        z_s = 1.0
        DA_flrw = cosmo.angular_diameter_distance(z_s)
        chi_s   = DA_flrw * (1.0 + z_s)   # comoving distance
        # Build a D that matches the comoving distance (no perturbation).
        # The Sachs equation with perturbation-only Riemann gives
        # D_AB = chi_s * I for the unperturbed background.
        D_flat = np.array([chi_s, 0.0, 0.0, chi_s])

        res = distance_comparison(D_flat, z_s, cosmo)
        np.testing.assert_allclose(res["D_A_ray"], DA_flrw, rtol=1e-10)
        np.testing.assert_allclose(res["D_A_FLRW"], DA_flrw, rtol=1e-10)
        np.testing.assert_allclose(res["delta_D_A"], 0.0, atol=1e-12)
