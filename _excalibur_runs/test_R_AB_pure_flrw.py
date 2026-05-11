#!/usr/bin/env python3
"""
Direct probe of R_AB in pure FLRW (Phi = 0 everywhere).

Hypothesis: from the formula in riemann_perturbed_flrw.py, in pure FLRW
the contributions from block 1 (Rd_k00l = a^2 H' delta_kl) and block 3
(Rd_kijl with second_scalar = H') should give equal and opposite
contributions to R_AB, yielding R_AB = 0.

If R_AB is not zero, the code does not match its own documented formula.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.observables.riemann_perturbed_flrw import riemann_blocks_kernel
from excalibur.observables.optical_tidal_matrix import (
    optical_tidal_matrix_optimized,
)


def compute_R_AB_at_a(a_val, k_eta_factor=1.0, e_factor=None):
    """Compute R_AB in pure FLRW at scale factor a_val.
    k_eta_factor: multiplier on k^eta (default -1 -> k^eta = -k_eta_factor)
    e_factor: multiplier on Sachs spatial component (default 1/a)."""
    H0 = 70e3 / (3.086e22)
    Om, Ol = 0.3, 0.7
    Hcon = a_val * H0 * np.sqrt(Om/a_val**3 + Ol)
    Hpr  = H0**2 * (-0.5*Om/a_val + Ol*a_val*a_val)
    c_val = 3e8

    Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
        a_val, Hcon, Hpr, 0.0, 0.0, 0.0,
        np.zeros(3), np.zeros(3), np.zeros((3,3)), c_val,
    )
    g_mu_nu = np.diag([-(a_val*c_val)**2, a_val*a_val, a_val*a_val, a_val*a_val])

    # Photon along x with k^eta = -k_eta_factor, k^x = c * k_eta_factor (null)
    k_mu = np.array([-k_eta_factor, c_val * k_eta_factor, 0.0, 0.0])
    null = (g_mu_nu @ k_mu) @ k_mu
    if e_factor is None:
        e_factor = 1.0/a_val
    e1_mu = np.array([0.0, 0.0, e_factor, 0.0])
    e2_mu = np.array([0.0, 0.0, 0.0, e_factor])

    R_AB = optical_tidal_matrix_optimized(Rd_k00l, Rd_0lki, Rd_kijl,
                                           k_mu, e1_mu, e2_mu, g_mu_nu)
    return R_AB, Hpr, null


def main():
    print("="*78)
    print(" R_AB at multiple points along a pure-FLRW trajectory")
    print("="*78)
    print(f" If cancellation analytic, R_AB / H' should be ~0 everywhere\n")

    # Scan scale factor along path (observer a=1 -> source a~0.67)
    print(f"  {'a':>5}  {'H_prime':>12}  {'R_AB[0,0]':>12}  {'R_AB[0,0]/H_prime':>16}")
    for a_val in (1.0, 0.9, 0.8, 0.7, 0.5):
        R_AB, Hpr, null = compute_R_AB_at_a(a_val)
        r = R_AB[0,0] / Hpr if abs(Hpr) > 0 else 0
        print(f"  {a_val:5.2f}  {Hpr:+.4e}  {R_AB[0,0]:+.4e}  {r:+.4e}")

    # What if k^eta is not exactly 1?  (numerical drift along the path)
    print("\n  Effect of small null-condition violation (k^eta drifted by 1%):")
    print(f"  {'a':>5}  {'k_eta_factor':>12}  {'null_violation':>16}  {'R_AB[0,0]':>12}  {'R/H':>10}")
    for kf in (1.0, 1.01, 1.05, 0.95):
        R_AB, Hpr, null = compute_R_AB_at_a(0.85, k_eta_factor=kf)
        r = R_AB[0,0] / Hpr if abs(Hpr) > 0 else 0
        print(f"  {0.85:5.2f}  {kf:12.4f}  {null:+.4e}  {R_AB[0,0]:+.4e}  {r:+.4e}")

    # What if Sachs vector is not exactly orthonormal?
    print("\n  Effect of Sachs basis non-orthonormality (e_factor scaled):")
    print(f"  {'a':>5}  {'e_factor':>12}  {'expected':>12}  {'R_AB[0,0]':>12}  {'R/H':>10}")
    for ef_mult in (1.0, 1.001, 1.01, 1.1):
        a_val = 0.85
        ef = (1.0/a_val) * ef_mult
        R_AB, Hpr, null = compute_R_AB_at_a(a_val, e_factor=ef)
        r = R_AB[0,0] / Hpr if abs(Hpr) > 0 else 0
        print(f"  {a_val:5.2f}  {ef:12.4e}  {1.0/a_val:12.4e}  {R_AB[0,0]:+.4e}  {r:+.4e}")
    return

    # (legacy debug below, kept for comparison)
    a    = 1.0                # scale factor today
    H0   = 70e3 / (3.086e22)  # s^-1, H_0 = 70 km/s/Mpc in SI
    Om   = 0.3
    Ol   = 0.7
    Hcon = a * H0 * np.sqrt(Om/a**3 + Ol)  # conformal Hubble at a=1
    Hpr  = H0**2 * (-0.5*Om/a + Ol*a*a)
    print(f"  a    = {a}")
    print(f"  H_co = {Hcon:.4e} 1/s")
    print(f"  H'   = {Hpr:.4e} 1/s^2  (Om=0.3, Ol=0.7  ->  positive, Lambda dominates today)")

    c_val   = 3e8
    phi     = 0.0
    phi_dot = 0.0
    phi_ddot= 0.0
    grad_phi      = np.zeros(3)
    grad_phi_dot  = np.zeros(3)
    hess_phi      = np.zeros((3, 3))

    # Compute Riemann blocks
    Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
        a, Hcon, Hpr, phi, phi_dot, phi_ddot,
        grad_phi, grad_phi_dot, hess_phi, c_val,
    )

    print("\n  Pure-FLRW Riemann blocks (Phi=0):")
    print(f"    Rd_k00l (3x3):  {Rd_k00l}")
    print(f"      diag values:  {[Rd_k00l[i,i] for i in range(3)]}")
    print(f"      expected:     a^2 * H' = {a*a*Hpr:.4e}  (each diagonal)")
    print(f"      off-diagonal max: {np.max(np.abs(Rd_k00l - np.diag(np.diag(Rd_k00l)))):.2e}")

    print(f"\n    Rd_0lki should be 0 (depends only on grad phi):  max abs = {np.max(np.abs(Rd_0lki)):.2e}")

    print(f"\n    Rd_kijl: Hessian part vanishes, only FLRW scalar remains")
    # In pure FLRW: Rd_kijl[k,i,j,l] = (a^2/c^2) * H' * (delta_li * delta_kj - delta_lk * delta_ij)
    print(f"      expected coefficient: (a^2/c^2)*H' = {a*a*Hpr/(c_val**2):.4e}")
    # Print a representative slice  k=0, i=1
    print(f"      Rd_kijl[0,1,:,:] = {Rd_kijl[0,1,:,:]}")
    print(f"      Rd_kijl[1,0,:,:] = {Rd_kijl[1,0,:,:]}")

    # Set up photon: along x-direction (spatial 0)
    g_mu_nu = np.diag([-(a*c_val)**2, a*a, a*a, a*a])
    k_mu = np.array([-1.0, c_val, 0.0, 0.0])
    # Verify null
    null = (g_mu_nu @ k_mu) @ k_mu
    print(f"\n  k^mu = {k_mu}")
    print(f"  g_mu_nu k^mu k^nu = {null:.4e}  (should be 0)")

    # Sachs basis: e1 along y, e2 along z, normalized in spatial metric
    # spatial metric = a^2 delta_ij; unit vector in y has e^i = delta^i_y / a
    e1_mu = np.array([0.0, 0.0, 1.0/a, 0.0])
    e2_mu = np.array([0.0, 0.0, 0.0, 1.0/a])

    # Compute R_AB
    R_AB = optical_tidal_matrix_optimized(Rd_k00l, Rd_0lki, Rd_kijl,
                                           k_mu, e1_mu, e2_mu, g_mu_nu)
    print(f"\n  R_AB =\n{R_AB}")
    print(f"  trace(R_AB) = {R_AB[0,0] + R_AB[1,1]:.4e}")

    # Compare to expected:
    # Hand calc:
    #   R_{2,0,2,0} = -R_{2,0,0,2} = -a^2 * H'
    #     -> contribution to R_11 = R_{2,0,2,0} * (k^0)^2 * (1/a)^2 = -H'
    #   R_{2,1,2,1} = (a^2/c^2) * H' * 1 (FLRW kron part: delta_li*delta_kj - delta_lk*delta_ij = 1*1 - 0)
    #     -> contribution to R_11 = R_{2,1,2,1} * (k^1)^2 * (1/a)^2 = H'
    #   Total R_11 (FLRW) = -H' + H' = 0
    print(f"\n  Hand-calculation expects R_AB = 0 (block 1 contribution -H' cancels block 3 contribution +H')")
    print(f"  Code gives R_AB[0,0] = {R_AB[0,0]:.4e}, expected ratio R_AB[0,0]/H' = {R_AB[0,0]/Hpr:.4f}")


if __name__ == "__main__":
    main()
