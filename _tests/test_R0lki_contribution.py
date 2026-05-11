"""
Test to verify that R_{0lki} Riemann block correctly contributes to R_{AB}.

Spatial reference formula in the stored block conventions:
    R_{AB} = - s_A^k s_B^l ( R_{k00l} k^0 k^0
                           + (R_{0lki} - R_{0kil}) k^0 k^i
                           + R_{kijl} k^i k^j )

where s_A^k are the PURELY SPATIAL Sachs components.

The code's 4D approach:
    T_{mu,nu} = R_{mu,alpha,nu,beta} k^alpha k^beta
    R_{AB} = T_{mu,nu} e_A^mu e_B^nu

The overall minus sign comes from the slot ordering in the full contraction:
the stored blocks are R_{k00l}, R_{0lki}, R_{kijl}, whereas the contraction
uses R_{k0l0}, R_{k0li}, R_{kil0}, R_{kilj}; each block therefore picks up one
antisymmetry when rewritten in the stored conventions.

For these to agree, the 4D Riemann must be FULLY assembled with all
symmetries.  Currently the code only stores a partial set of components
from the R_{0lki} block:
    - Brute-force: R_down[0, l+1, k+1, i+1] only
    - Optimized: T_down[0, kk+1] only

Missing symmetry partners from R_{0lki}:
    pair symmetry:  R_{0,l+1,k+1,i+1} = R_{k+1,i+1,0,l+1}
    antisymmetry:   R_{k+1,i+1,l+1,0} = -R_{k+1,i+1,0,l+1}

These generate T_{k+1, 0} and T_{k+1, l+1} contributions from Block 2.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.observables.riemann_perturbed_flrw import riemann_blocks_kernel
from excalibur.observables.optical_tidal_matrix import (
    optical_tidal_matrix_from_blocks,
    optical_tidal_matrix_optimized,
)


def build_full_riemann_from_blocks(Rd_k00l, Rd_0lki, Rd_kijl):
    """
    Build the COMPLETE 4x4x4x4 Riemann tensor from the three blocks,
    applying ALL Riemann symmetries:
        1. R_{abcd} = -R_{bacd}       (antisym first pair)
        2. R_{abcd} = -R_{abdc}       (antisym second pair)
        3. R_{abcd} = R_{cdab}         (pair symmetry)
    """
    R = np.zeros((4, 4, 4, 4))

    # First, set the explicit block values
    # Block 1: R_{k+1, 0, 0, l+1} = Rd_k00l[k, l]
    for k in range(3):
        for l in range(3):
            R[k+1, 0, 0, l+1] = Rd_k00l[k, l]

    # Block 2: R_{0, l+1, k+1, i+1} = Rd_0lki[l, k, i]
    for l_idx in range(3):
        for k in range(3):
            for i in range(3):
                R[0, l_idx+1, k+1, i+1] = Rd_0lki[l_idx, k, i]

    # Block 3: R_{k+1, i+1, j+1, l+1} = Rd_kijl[k, i, j, l]
    for k in range(3):
        for i in range(3):
            for j in range(3):
                for l_idx in range(3):
                    R[k+1, i+1, j+1, l_idx+1] = Rd_kijl[k, i, j, l_idx]

    # Now apply ALL symmetries to fill the full tensor
    R_full = np.zeros((4, 4, 4, 4))

    # Set all symmetry partners from each explicitly set component
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    if R[a, b, c, d] != 0.0:
                        val = R[a, b, c, d]
                        # Apply all symmetries
                        R_full[a, b, c, d] = val
                        R_full[b, a, c, d] = -val     # antisym first pair
                        R_full[a, b, d, c] = -val     # antisym second pair
                        R_full[b, a, d, c] = val       # both antisym
                        R_full[c, d, a, b] = val       # pair symmetry
                        R_full[d, c, a, b] = -val      # pair + antisym first
                        R_full[c, d, b, a] = -val      # pair + antisym second
                        R_full[d, c, b, a] = val       # pair + both antisym

    return R_full


def contract_full_riemann(R_full, k_mu, e1_mu, e2_mu):
    """Contract full 4D Riemann: R_{AB} = R_{mu,alpha,nu,beta} k^alpha k^beta e_A^mu e_B^nu"""
    e = np.array([e1_mu, e2_mu])
    R_AB = np.zeros((2, 2))
    for A in range(2):
        for B in range(2):
            s = 0.0
            for mu in range(4):
                for alpha in range(4):
                    for nu in range(4):
                        for beta in range(4):
                            s += (R_full[mu, alpha, nu, beta]
                                  * k_mu[alpha] * k_mu[beta]
                                  * e[A, mu] * e[B, nu])
            R_AB[A, B] = s
    return R_AB


def spatial_formula_RAB(Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu):
    """
    Compute R_{AB} using the user's spatial formula:
        R_{AB} = - s_A^k s_B^l ( R_{k00l} k^0 k^0
                                + (R_{0lki} - R_{0kil}) k^0 k^i
                                + R_{kijl} k^i k^j )

    where s_A^k = e_A^{k+1} are the spatial components of the Sachs vectors.
    """
    k0 = k_mu[0]
    ki = k_mu[1:4]  # spatial components of k^mu

    # Spatial Sachs components
    s1 = e1_mu[1:4]
    s2 = e2_mu[1:4]
    s = np.array([s1, s2])

    R_AB = np.zeros((2, 2))
    for A in range(2):
        for B in range(2):
            val = 0.0
            for k in range(3):
                for l in range(3):
                    # Term 1: R_{k00l} k^0 k^0
                    val += s[A, k] * s[B, l] * Rd_k00l[k, l] * k0 * k0

                    # Term 2: (R_{0lki} - R_{0kil}) k^0 k^i
                    for i in range(3):
                        val += s[A, k] * s[B, l] * (
                            Rd_0lki[l, k, i] - Rd_0lki[k, i, l]
                        ) * k0 * ki[i]

                    # Term 3: R_{kijl} k^i k^j
                    for i in range(3):
                        for j in range(3):
                            val += s[A, k] * s[B, l] * Rd_kijl[k, i, j, l] * ki[i] * ki[j]

                R_AB[A, B] = -val
    return R_AB


def test_R0lki_contribution():
    """
    Compare three methods of computing R_{AB}:
        1. Full 4D Riemann with all symmetries (reference)
        2. User's spatial formula
        3. Current code (optical_tidal_matrix_optimized)
        4. Current code (optical_tidal_matrix_from_blocks)

    Use a non-trivial photon direction (k not along coordinate axis)
    and a non-trivial Sachs basis where e_A^0 may be small but nonzero.
    """
    # Physical parameters (FLRW + perturbation)
    c_val = 3e8
    a = 0.95
    H = 2.3e-18   # conformal Hubble
    Hprime = -1.0e-36
    phi = -1e5     # Phi in m^2/s^2
    phi_dot = 0.0
    phi_ddot = 0.0
    grad_phi = np.array([1e-10, -2e-10, 0.5e-10])  # non-trivial gradient
    grad_phi_dot = np.array([1.0e-11, -2.0e-11, 0.5e-11])
    hess_phi = np.array([
        [1e-20, 0.3e-20, -0.1e-20],
        [0.3e-20, -0.5e-20, 0.2e-20],
        [-0.1e-20, 0.2e-20, 0.8e-20],
    ])

    # Compute Riemann blocks
    Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
        a, H, Hprime, phi, phi_dot, phi_ddot,
        grad_phi, grad_phi_dot, hess_phi, c_val
    )

    # Metric components for a perturbed FLRW
    g00 = -a**2 * (1 + 2*phi/c_val**2) * c_val**2
    gii_val = a**2 * (1 - 2*phi/c_val**2)
    g_mu_nu = np.diag([g00, gii_val, gii_val, gii_val])

    # Non-trivial photon direction (diagonal-ish)
    # k^i chosen, then k^0 from null condition: g_{00}(k^0)^2 + g_{ii} sum(k^i)^2 = 0
    ki_spatial = np.array([0.6, 0.3, -0.7])  # arbitrary direction
    k_spatial_sq = gii_val * np.sum(ki_spatial**2)
    k0 = np.sqrt(-k_spatial_sq / g00)
    k_mu = np.array([k0, ki_spatial[0], ki_spatial[1], ki_spatial[2]])

    # Sachs basis: two spatial vectors orthogonal to k_spatial and to each other
    # Then lift to 4D using g_{0i} k^i / (g_{00} k^0) for the time component
    # Since diagonal metric, e_A^0 = -g_spatial @ e_spatial @ k_spatial / (g00 * k0)
    # Actually for diagonal metric: e_A^0 = -gii * (e_spatial . k_spatial) / (g00 * k0)

    # Construct spatial orthonormal basis perpendicular to k_spatial
    k_hat = ki_spatial / np.linalg.norm(ki_spatial)
    # Choose an arbitrary vector not parallel to k_hat
    v = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v, k_hat)) > 0.9:
        v = np.array([0.0, 1.0, 0.0])
    e1_spatial = v - np.dot(v, k_hat) * k_hat
    e1_spatial /= np.linalg.norm(e1_spatial)
    e2_spatial = np.cross(k_hat, e1_spatial)
    e2_spatial /= np.linalg.norm(e2_spatial)

    # Lift to 4D:
    # Orthogonality g_{mu nu} e_A^mu k^nu = 0 gives:
    # g00 * e_A^0 * k^0 + gii * e_A^i * k^i = 0
    # => e_A^0 = -gii * (e_A_spatial . k_spatial) / (g00 * k^0)
    e1_0 = -gii_val * np.dot(e1_spatial, ki_spatial) / (g00 * k0)
    e2_0 = -gii_val * np.dot(e2_spatial, ki_spatial) / (g00 * k0)

    e1_mu = np.array([e1_0, e1_spatial[0], e1_spatial[1], e1_spatial[2]])
    e2_mu = np.array([e2_0, e2_spatial[0], e2_spatial[1], e2_spatial[2]])

    print(f"k^mu = {k_mu}")
    print(f"e1^mu = {e1_mu}")
    print(f"e2^mu = {e2_mu}")
    print(f"e1^0 = {e1_0:.6e}")
    print(f"e2^0 = {e2_0:.6e}")
    print()

    # Verify orthogonality
    print("g(e1, k) =", np.einsum('i,ij,j', e1_mu, g_mu_nu, k_mu))
    print("g(e2, k) =", np.einsum('i,ij,j', e2_mu, g_mu_nu, k_mu))
    print("g(e1, e1) =", np.einsum('i,ij,j', e1_mu, g_mu_nu, e1_mu))
    print("g(e2, e2) =", np.einsum('i,ij,j', e2_mu, g_mu_nu, e2_mu))
    print()

    # Check that R_{0lki} block is non-trivial
    print(f"max|Rd_0lki| = {np.max(np.abs(Rd_0lki)):.6e}")
    print(f"max|Rd_k00l| = {np.max(np.abs(Rd_k00l)):.6e}")
    print(f"max|Rd_kijl| = {np.max(np.abs(Rd_kijl)):.6e}")
    print()

    # Method 1: Full Riemann with all symmetries
    R_full = build_full_riemann_from_blocks(Rd_k00l, Rd_0lki, Rd_kijl)
    R_AB_full = contract_full_riemann(R_full, k_mu, e1_mu, e2_mu)
    print("Method 1 (full Riemann, all symmetries):")
    print(f"  R_AB = {R_AB_full}")
    print()

    # Method 2: User's spatial formula
    R_AB_spatial = spatial_formula_RAB(Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu)
    print("Method 2 (spatial formula):")
    print(f"  R_AB = {R_AB_spatial}")
    print()

    # Method 3: Code's optimized implementation
    R_AB_opt = optical_tidal_matrix_optimized(
        Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu, g_mu_nu
    )
    print("Method 3 (code optimized):")
    print(f"  R_AB = {R_AB_opt}")
    print()

    # Method 4: Code's brute-force implementation
    R_AB_brute = optical_tidal_matrix_from_blocks(
        Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu, g_mu_nu
    )
    print("Method 4 (code brute-force):")
    print(f"  R_AB = {R_AB_brute}")
    print()

    # Check if the code agrees with the correct methods
    print("=" * 60)
    err_spatial_vs_full = np.max(np.abs(R_AB_spatial - R_AB_full))
    err_opt_vs_full = np.max(np.abs(R_AB_opt - R_AB_full))
    err_brute_vs_full = np.max(np.abs(R_AB_brute - R_AB_full))

    scale = np.max(np.abs(R_AB_full))
    print(f"Scale of R_AB: {scale:.6e}")
    print(f"|spatial - full| / scale = {err_spatial_vs_full/scale:.6e}")
    print(f"|optimized - full| / scale = {err_opt_vs_full/scale:.6e}")
    print(f"|brute-force - full| / scale = {err_brute_vs_full/scale:.6e}")
    print()

    np.testing.assert_allclose(R_AB_spatial, R_AB_full, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(R_AB_opt, R_AB_full, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(R_AB_brute, R_AB_full, rtol=1e-12, atol=0.0)

    # Now check: what is the contribution from Block 2 alone?
    # Zero out Block 2 and recompute
    zero_0lki = np.zeros_like(Rd_0lki)
    R_AB_no_block2 = contract_full_riemann(
        build_full_riemann_from_blocks(Rd_k00l, zero_0lki, Rd_kijl),
        k_mu, e1_mu, e2_mu
    )
    block2_contribution = R_AB_full - R_AB_no_block2
    print(f"\nBlock 2 (R_0lki) contribution to R_AB:")
    print(f"  {block2_contribution}")
    print(f"  Relative magnitude: {np.max(np.abs(block2_contribution))/scale:.4f}")

    assert np.max(np.abs(block2_contribution)) > 1e-6 * scale


if __name__ == '__main__':
    test_R0lki_contribution()
