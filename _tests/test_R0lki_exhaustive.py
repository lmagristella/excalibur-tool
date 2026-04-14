#!/usr/bin/env python3
r"""
EXHAUSTIVE verification that:

1.  riemann_blocks_kernel produces the correct 9 + 27 + 81 components.
2.  The contraction  R_{AB} = R_{μανβ} k^α k^β e_A^μ e_B^ν
    reproduces the user's spatial formula EXACTLY:

        R_{AB} = s_A^k s_B^l ( R_{k00l} k^0 k^0
                              + (R_{0lki} − R_{0kil}) k^0 k^i
                              + R_{kijl} k^i k^j )

    where s_A^k are the spatial components of the Sachs vectors.

We use a NON-TRIVIAL set of inputs (H, ∇Φ, ∇Φ̇, Hess Φ all nonzero)
and a photon direction off all coordinate axes so that every term
contributes.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from excalibur.observables.riemann_perturbed_flrw import riemann_blocks_kernel
from excalibur.observables.optical_tidal_matrix import (
    optical_tidal_matrix_from_blocks,
    optical_tidal_matrix_optimized,
)

# =====================================================================
# 1. REFERENCE BLOCK FORMULAS — independent re-implementation
# =====================================================================

def reference_Rd_k00l(a, H, Hp, psi, psi_dot, psi_ddot, hess, c):
    """Independent implementation of R_{k00l}."""
    c2 = c * c
    a2 = a * a
    diag = Hp * (1 - 2*psi/c2) + psi_ddot/c2 + H*(psi_dot + psi_dot)/c2
    R = np.empty((3, 3))
    for k in range(3):
        for l in range(3):
            R[k, l] = -hess[k, l]
            if k == l:
                R[k, l] += diag
    return a2 * R


def reference_Rd_0lki(a, H, grad_phi_dot, grad_phi, c):
    """Independent implementation of R_{0lki}."""
    c2 = c * c
    a2 = a * a
    combo = grad_phi_dot + H * grad_phi  # vector (3,)
    R = np.zeros((3, 3, 3))
    for l in range(3):
        for k in range(3):
            for i in range(3):
                v = 0.0
                if i == l:
                    v += combo[k]
                if l == k:
                    v -= combo[i]
                R[l, k, i] = (a2 / c2) * v
    return R


def reference_Rd_kijl(a, H, Hp, psi, psi_dot, phi, hess, c):
    """Independent implementation of R_{kijl}."""
    c2 = c * c
    a2 = a * a
    scalar2 = Hp - (2*H*psi_dot + 2*H*H*phi + 4*H*H*psi) / c2
    R = np.zeros((3, 3, 3, 3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    v = 0.0
                    if k == j: v += hess[i, l]
                    if k == l: v -= hess[i, j]
                    if i == j: v -= hess[k, l]
                    if i == l: v += hess[k, j]
                    kron = 0.0
                    if l == i and k == j: kron += 1.0
                    if l == k and i == j: kron -= 1.0
                    v += scalar2 * kron
                    R[k, i, j, l] = (a2 / c2) * v
    return R

# =====================================================================
# 2. SPATIAL FORMULA — direct implementation of the user's equation
# =====================================================================

def spatial_formula_RAB(Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu):
    r"""
    R_{AB} = s_A^k s_B^l ( R_{k00l} k^0 k^0
                          + (R_{0lki} − R_{0kil}) k^0 k^i
                          + R_{kijl} k^i k^j )
    """
    k0 = k_mu[0]
    ki = k_mu[1:4]
    s = np.array([e1_mu[1:4], e2_mu[1:4]])  # spatial Sachs: s_A^k

    R_AB = np.zeros((2, 2))
    for A in range(2):
        for B in range(2):
            val = 0.0
            for k in range(3):
                for l in range(3):
                    # Term 1: R_{k00l} k^0 k^0
                    val += s[A, k] * s[B, l] * Rd_k00l[k, l] * k0 * k0

                    # Term 2: (R_{0lki} − R_{0kil}) k^0 k^i
                    for i in range(3):
                        val += s[A, k] * s[B, l] * (
                            Rd_0lki[l, k, i] - Rd_0lki[k, i, l]
                        ) * k0 * ki[i]

                    # Term 3: R_{kijl} k^i k^j
                    for i in range(3):
                        for j in range(3):
                            val += s[A, k] * s[B, l] * Rd_kijl[k, i, j, l] * ki[i] * ki[j]

            R_AB[A, B] = val
    return R_AB

# =====================================================================
# 3. FULL 4D RIEMANN — build with all symmetries, contract over all μανβ
# =====================================================================

def build_full_riemann(Rd_k00l, Rd_0lki, Rd_kijl):
    """Build complete R[4,4,4,4] with all Riemann symmetries."""
    R = np.zeros((4, 4, 4, 4))
    def sym(a, b, c, d, v):
        R[a,b,c,d]=v;  R[b,a,c,d]=-v; R[a,b,d,c]=-v; R[b,a,d,c]=v
        R[c,d,a,b]=v;  R[d,c,a,b]=-v; R[c,d,b,a]=-v; R[d,c,b,a]=v

    for k in range(3):
        for l in range(3):
            sym(k+1, 0, 0, l+1, Rd_k00l[k, l])
    for l in range(3):
        for k in range(3):
            for i in range(3):
                sym(0, l+1, k+1, i+1, Rd_0lki[l, k, i])
    for k in range(3):
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    sym(k+1, i+1, j+1, l+1, Rd_kijl[k, i, j, l])
    return R


def full_4d_contraction(R_full, k_mu, e1_mu, e2_mu):
    """R_{AB} = R_{μανβ} k^α k^β e_A^μ e_B^ν — brute force over all 4^4."""
    e = np.array([e1_mu, e2_mu])
    R_AB = np.zeros((2, 2))
    for A in range(2):
        for B in range(2):
            s = 0.0
            for mu in range(4):
                for al in range(4):
                    for nu in range(4):
                        for be in range(4):
                            s += R_full[mu,al,nu,be] * k_mu[al] * k_mu[be] * e[A,mu] * e[B,nu]
            R_AB[A, B] = s
    return R_AB


# =====================================================================
# 4. DERIVE the spatial formula from the 4D one — step by step
# =====================================================================

def derive_spatial_from_4d(R_full, k_mu, e1_mu, e2_mu):
    r"""
    Expand R_{μανβ} k^α k^β e_A^μ e_B^ν keeping ONLY
    the spatial parts of e_A^μ  (i.e.  e_A^0 = 0, e_A^{k+1} = s_A^k).

    This is the "exact-if-e_A^0=0" version of the spatial formula.
    Useful to check whether the spatial formula matches when e_A^0 ≈ 0.
    """
    k0 = k_mu[0]
    ki = k_mu[1:4]
    s = np.array([e1_mu[1:4], e2_mu[1:4]])

    R_AB = np.zeros((2, 2))
    for A in range(2):
        for B in range(2):
            val = 0.0
            for k in range(3):
                for l in range(3):
                    # Enumerate all (α, β) in {0, spatial}
                    # α=0, β=0:  R_{k+1,0,l+1,0} * k^0 * k^0
                    val += s[A,k] * s[B,l] * R_full[k+1, 0, l+1, 0] * k0 * k0

                    # α=0, β=j+1:  R_{k+1,0,l+1,j+1} * k^0 * k^j
                    for j in range(3):
                        val += s[A,k] * s[B,l] * R_full[k+1, 0, l+1, j+1] * k0 * ki[j]

                    # α=i+1, β=0:  R_{k+1,i+1,l+1,0} * k^i * k^0
                    for i in range(3):
                        val += s[A,k] * s[B,l] * R_full[k+1, i+1, l+1, 0] * ki[i] * k0

                    # α=i+1, β=j+1:  R_{k+1,i+1,l+1,j+1} * k^i * k^j
                    for i in range(3):
                        for j in range(3):
                            val += s[A,k] * s[B,l] * R_full[k+1, i+1, l+1, j+1] * ki[i] * ki[j]

            R_AB[A, B] = val
    return R_AB


# =====================================================================
# 5. MAP the 4D components to block names
# =====================================================================

def identify_4d_to_blocks(R_full, k_mu, e1_mu, e2_mu):
    r"""
    Same as derive_spatial_from_4d but print which 4D components map
    to which block, and what the user's formula says.

    R_{k+1, 0, l+1, 0}   → antisymmetry on 2nd pair: = -R_{k+1, 0, 0, l+1} = -R_{k00l}
    R_{k+1, 0, l+1, j+1} → antisymmetry on 1st pair: R_{k+1,0,l+1,j+1} = -R_{0,k+1,l+1,j+1}
                            and R_{0,k+1,l+1,j+1} = Rd_0lki[k,l,j]  (Block 2 with 1st spatial index=k)
                            So R_{k+1,0,l+1,j+1} = -Rd_0lki[k,l,j]
    R_{k+1, i+1, l+1, 0} → pair symmetry of R_{l+1,0,k+1,i+1}
                            = R_{0,l+1,...} → need antisym on 1st pair:
                            R_{l+1,0,k+1,i+1} = -R_{0,l+1,k+1,i+1} = -Rd_0lki[l,k,i]
                            So R_{k+1,i+1,l+1,0} = -Rd_0lki[l,k,i]
    R_{k+1,i+1,l+1,j+1}  → Rd_kijl[k,i,l,j]   but wait, need to check antisymmetry...
                            R_{k+1,i+1,l+1,j+1} = Rd_kijl[k,i,l,j]  — NO!
                            Our block is R_{k,i,j,l} → R[k+1,i+1,j+1,l+1]
                            So R[k+1,i+1,l+1,j+1] = Rd_kijl[k,i,l,j]?
                            Only if kijl already has the right antisymmetry.
                            Actually, Rd_kijl has antisym on (k,i) and (j,l) by construction,
                            so R[k+1,i+1,l+1,j+1] = -R[k+1,i+1,j+1,l+1] = -Rd_kijl[k,i,j,l]
                            Hmm, let me just read from R_full.
    """
    k0 = k_mu[0]
    ki = k_mu[1:4]
    s = np.array([e1_mu[1:4], e2_mu[1:4]])

    # Just compute and compare term by term
    term1 = np.zeros((2, 2))  # from R_{k+1,0,l+1,0} k^0 k^0
    term2a = np.zeros((2, 2)) # from R_{k+1,0,l+1,j+1} k^0 k^j
    term2b = np.zeros((2, 2)) # from R_{k+1,i+1,l+1,0} k^i k^0
    term3 = np.zeros((2, 2))  # from R_{k+1,i+1,l+1,j+1} k^i k^j

    for A in range(2):
        for B in range(2):
            for k in range(3):
                for l in range(3):
                    term1[A,B] += s[A,k] * s[B,l] * R_full[k+1, 0, l+1, 0] * k0 * k0
                    for j in range(3):
                        term2a[A,B] += s[A,k] * s[B,l] * R_full[k+1, 0, l+1, j+1] * k0 * ki[j]
                    for i in range(3):
                        term2b[A,B] += s[A,k] * s[B,l] * R_full[k+1, i+1, l+1, 0] * ki[i] * k0
                    for i in range(3):
                        for j in range(3):
                            term3[A,B] += s[A,k] * s[B,l] * R_full[k+1, i+1, l+1, j+1] * ki[i] * ki[j]

    return term1, term2a, term2b, term3


# =====================================================================
# MAIN TEST
# =====================================================================

def main():
    np.set_printoptions(precision=6, linewidth=120)

    # --- Physical parameters (everything non-trivial) ---
    c = 3e8
    a = 0.95
    H = 2.3e-18
    Hp = -1.0e-36
    phi = -1e5
    phi_dot = 1e3      # Φ̇ ≠ 0
    phi_ddot = 5e1     # Φ̈ ≠ 0
    grad_phi = np.array([1e-10, -2e-10, 0.5e-10])
    grad_phi_dot = np.array([5e-8, -3e-8, 2e-8])
    hess_phi = np.array([
        [ 1e-20,  0.3e-20, -0.1e-20],
        [ 0.3e-20, -0.5e-20,  0.2e-20],
        [-0.1e-20,  0.2e-20,  0.8e-20],
    ])

    print("=" * 70)
    print("PART 1: BLOCK GENERATION — code vs independent reference")
    print("=" * 70)

    Rd_k00l, Rd_0lki, Rd_kijl = riemann_blocks_kernel(
        a, H, Hp, phi, phi_dot, phi_ddot,
        grad_phi, grad_phi_dot, hess_phi, c
    )

    ref_k00l = reference_Rd_k00l(a, H, Hp, phi, phi_dot, phi_ddot, hess_phi, c)
    ref_0lki = reference_Rd_0lki(a, H, grad_phi_dot, grad_phi, c)
    ref_kijl = reference_Rd_kijl(a, H, Hp, phi, phi_dot, phi, hess_phi, c)

    err1 = np.max(np.abs(Rd_k00l - ref_k00l))
    err2 = np.max(np.abs(Rd_0lki - ref_0lki))
    err3 = np.max(np.abs(Rd_kijl - ref_kijl))
    sc1 = np.max(np.abs(ref_k00l))
    sc2 = np.max(np.abs(ref_0lki))
    sc3 = np.max(np.abs(ref_kijl))

    print(f"  Block 1  R_k00l  (3×3  =  9 comp):  max|err|/scale = {err1/sc1:.2e}")
    print(f"  Block 2  R_0lki  (3×3×3 = 27 comp):  max|err|/scale = {err2/sc2:.2e}")
    print(f"  Block 3  R_kijl  (3^4  = 81 comp):  max|err|/scale = {err3/sc3:.2e}")
    ok1 = err1/sc1 < 1e-14 and err2/sc2 < 1e-14 and err3/sc3 < 1e-14
    print(f"  → Blocks correct: {'✅ YES' if ok1 else '❌ NO'}")
    print()

    # Count non-trivially-nonzero entries
    nz1 = np.count_nonzero(np.abs(Rd_k00l) > 1e-100)
    nz2 = np.count_nonzero(np.abs(Rd_0lki) > 1e-100)
    nz3 = np.count_nonzero(np.abs(Rd_kijl) > 1e-100)
    print(f"  Non-zero entries: Block1={nz1}/9, Block2={nz2}/27, Block3={nz3}/81")
    print()

    # --- Photon 4-velocity (null, off-axis) ---
    g00 = -a**2 * (1 + 2*phi/c**2) * c**2
    gii = a**2 * (1 - 2*phi/c**2)
    g_mu_nu = np.diag([g00, gii, gii, gii])

    ki_spatial = np.array([0.6, 0.3, -0.7])
    k0 = np.sqrt(-gii * np.sum(ki_spatial**2) / g00)
    k_mu = np.array([k0, *ki_spatial])

    # Sachs basis (spatial, orthogonal to k)
    k_hat = ki_spatial / np.linalg.norm(ki_spatial)
    v = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v, k_hat)) > 0.9:
        v = np.array([0.0, 1.0, 0.0])
    e1_s = v - np.dot(v, k_hat) * k_hat
    e1_s /= np.linalg.norm(e1_s)
    e2_s = np.cross(k_hat, e1_s)
    e2_s /= np.linalg.norm(e2_s)

    # Lift to 4D:  g_{μν} e_A^μ k^ν = 0  →  e_A^0 = -gii (e_s · k_s) / (g00 k^0)
    e1_0 = -gii * np.dot(e1_s, ki_spatial) / (g00 * k0)
    e2_0 = -gii * np.dot(e2_s, ki_spatial) / (g00 * k0)
    e1_mu = np.array([e1_0, *e1_s])
    e2_mu = np.array([e2_0, *e2_s])

    print(f"  k^μ = [{k0:.4e}, {ki_spatial[0]}, {ki_spatial[1]}, {ki_spatial[2]}]")
    print(f"  e1^0 = {e1_0:.4e},  e2^0 = {e2_0:.4e}")
    print(f"  g(k,k) = {np.einsum('i,ij,j', k_mu, g_mu_nu, k_mu):.2e}  (should be 0)")
    print(f"  g(e1,k) = {np.einsum('i,ij,j', e1_mu, g_mu_nu, k_mu):.2e}  (should be 0)")
    print(f"  g(e2,k) = {np.einsum('i,ij,j', e2_mu, g_mu_nu, k_mu):.2e}  (should be 0)")
    print()

    # =================================================================
    print("=" * 70)
    print("PART 2: CONTRACTION — four methods compared")
    print("=" * 70)

    # Method A: full 4D Riemann with all symmetries + 4^4 contraction
    R_full = build_full_riemann(Rd_k00l, Rd_0lki, Rd_kijl)
    R_AB_4d = full_4d_contraction(R_full, k_mu, e1_mu, e2_mu)

    # Method B: user's spatial formula (s_A^k purely spatial)
    R_AB_spatial = spatial_formula_RAB(Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu)

    # Method C: code's optimized
    R_AB_opt = optical_tidal_matrix_optimized(
        Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu, g_mu_nu)

    # Method D: code's brute-force
    R_AB_brute = optical_tidal_matrix_from_blocks(
        Rd_k00l, Rd_0lki, Rd_kijl, k_mu, e1_mu, e2_mu, g_mu_nu)

    scale = np.max(np.abs(R_AB_4d))

    print(f"  A  (full 4D, all sym):        {R_AB_4d}")
    print(f"  B  (user spatial formula):     {R_AB_spatial}")
    print(f"  C  (code optimized):           {R_AB_opt}")
    print(f"  D  (code brute-force):         {R_AB_brute}")
    print()
    print(f"  |C − A| / scale = {np.max(np.abs(R_AB_opt - R_AB_4d))/scale:.2e}  (code opt vs ref)")
    print(f"  |D − A| / scale = {np.max(np.abs(R_AB_brute - R_AB_4d))/scale:.2e}  (code brute vs ref)")
    print(f"  |B − A| / scale = {np.max(np.abs(R_AB_spatial - R_AB_4d))/scale:.2e}  (spatial vs ref)")
    print(f"  |B + A| / scale = {np.max(np.abs(R_AB_spatial + R_AB_4d))/scale:.2e}  (spatial = -ref ?)")
    print()

    # =================================================================
    print("=" * 70)
    print("PART 3: DECOMPOSE the 4D formula into the spatial terms")
    print("=" * 70)

    # Using the full Riemann, extract the 4 terms of the spatial expansion
    t1, t2a, t2b, t3 = identify_4d_to_blocks(R_full, k_mu, e1_mu, e2_mu)

    # Now compare with user's formula terms
    # Term 1 of 4D expansion: s_A^k s_B^l R_{k+1,0,l+1,0} k^0 k^0
    # R_{k+1,0,l+1,0} = -R_{k+1,0,0,l+1} = -Rd_k00l[k,l]  (antisym on 2nd pair)
    # So term 1 = -s_A^k s_B^l Rd_k00l[k,l] k^0^2
    # User's formula has: +s_A^k s_B^l Rd_k00l[k,l] k^0^2
    # → SIGN DIFFERENCE!

    k0v = k_mu[0]
    kiv = k_mu[1:4]
    sv = np.array([e1_mu[1:4], e2_mu[1:4]])

    # Manually compute each user formula term
    user_t1 = np.zeros((2,2))   # R_{k00l} k^0 k^0
    user_t2 = np.zeros((2,2))   # (R_{0lki} - R_{0kil}) k^0 k^i
    user_t3 = np.zeros((2,2))   # R_{kijl} k^i k^j
    for A in range(2):
        for B in range(2):
            for k in range(3):
                for l in range(3):
                    user_t1[A,B] += sv[A,k] * sv[B,l] * Rd_k00l[k,l] * k0v * k0v
                    for i in range(3):
                        user_t2[A,B] += sv[A,k] * sv[B,l] * (Rd_0lki[l,k,i] - Rd_0lki[k,i,l]) * k0v * kiv[i]
                    for i in range(3):
                        for j in range(3):
                            user_t3[A,B] += sv[A,k] * sv[B,l] * Rd_kijl[k,i,j,l] * kiv[i] * kiv[j]

    print(f"  4D term1 (α=0,β=0):       {t1[0,0]:.6e}")
    print(f"  User term1 (R_k00l k0²):  {user_t1[0,0]:.6e}")
    print(f"  Ratio t1_4d / t1_user:     {t1[0,0]/user_t1[0,0]:.6f}")
    print()
    print(f"  4D term2a (α=0,β=j+1):    {t2a[0,0]:.6e}")
    print(f"  4D term2b (α=i+1,β=0):    {t2b[0,0]:.6e}")
    print(f"  4D term2a+2b:              {(t2a+t2b)[0,0]:.6e}")
    print(f"  User term2 (R_0lki cross): {user_t2[0,0]:.6e}")
    print(f"  Ratio (t2a+t2b)/t2_user:   {(t2a[0,0]+t2b[0,0])/user_t2[0,0]:.6f}")
    print()
    print(f"  4D term3 (α=i+1,β=j+1):   {t3[0,0]:.6e}")
    print(f"  User term3 (R_kijl ki kj): {user_t3[0,0]:.6e}")
    print(f"  Ratio t3_4d / t3_user:     {t3[0,0]/user_t3[0,0]:.6f}")
    print()
    print(f"  4D total (e_A^0≈0 part):   {(t1+t2a+t2b+t3)[0,0]:.6e}")
    print(f"  User formula total:        {(user_t1+user_t2+user_t3)[0,0]:.6e}")
    print(f"  Full 4D (with e_A^0):      {R_AB_4d[0,0]:.6e}")
    print()

    # Now let's identify the sign mapping for term 1:
    # R_{k+1, 0, l+1, 0} from the full Riemann:
    # R_{k+1, 0, 0, l+1} = Rd_k00l[k,l] (by set_sym)
    # R_{k+1, 0, l+1, 0} = -R_{k+1, 0, 0, l+1} = -Rd_k00l[k,l] (antisym 2nd pair)
    # So 4D term1 = s_A^k s_B^l × (-Rd_k00l[k,l]) × k0² = -user_t1
    print("  SIGN ANALYSIS:")
    print(f"  R[k+1,0,l+1,0] = -R_k00l[k,l] (antisym 2nd pair)")
    print(f"  → 4D term1 = -user_t1. Ratio = {t1[0,0]/user_t1[0,0]:.1f}")
    print()

    # For term2a: R_{k+1,0,l+1,j+1}
    # R_{k+1,0,l+1,j+1}: from antisym 1st pair of R_{0,k+1,l+1,j+1}
    #   R_{0,k+1,l+1,j+1} — Block 2 has R_{0,l+1,k+1,i+1} = Rd_0lki[l,k,i]
    #   So R_{0,k+1,l+1,j+1} = Rd_0lki[k,l,j]
    #   → R_{k+1,0,l+1,j+1} = -R_{0,k+1,l+1,j+1} = -Rd_0lki[k,l,j]
    # Contribution: s_A^k s_B^l (-Rd_0lki[k,l,j]) k^0 k^j
    # = -s_A^k s_B^l Rd_0lki[k,l,j] k^0 k^j

    # For term2b: R_{k+1,i+1,l+1,0}
    # R_{k+1,i+1,l+1,0} = -R_{k+1,i+1,0,l+1}  (antisym 2nd pair)
    # R_{k+1,i+1,0,l+1} = R_{0,l+1,k+1,i+1}   (pair symmetry)
    # = Rd_0lki[l,k,i]
    # → R_{k+1,i+1,l+1,0} = -Rd_0lki[l,k,i]
    # Contribution: s_A^k s_B^l (-Rd_0lki[l,k,i]) k^i k^0

    # So 4D term2a + term2b = s_A^k s_B^l k^0 × sum_i [ -Rd_0lki[k,l,i]*k^i - Rd_0lki[l,k,i]*k^i ]
    # User formula:          s_A^k s_B^l k^0 × sum_i [ (Rd_0lki[l,k,i] - Rd_0lki[k,i,l])*k^i ]

    # Are they equal?  Need: -(Rd_0lki[k,l,i] + Rd_0lki[l,k,i]) = Rd_0lki[l,k,i] - Rd_0lki[k,i,l]
    # i.e. -Rd_0lki[k,l,i] - 2*Rd_0lki[l,k,i] + Rd_0lki[k,i,l] = 0 ?
    # This is NOT trivially true. Let's check numerically.
    print(f"  4D cross term mapping:")
    print(f"    term2a: R[k+1,0,l+1,j+1] = -Rd_0lki[k,l,j]")
    print(f"    term2b: R[k+1,i+1,l+1,0] = -Rd_0lki[l,k,i]")
    print(f"    → 4D: -sum_i (Rd_0lki[k,l,i] + Rd_0lki[l,k,i]) k^0 k^i")
    print(f"    User: +sum_i (Rd_0lki[l,k,i] - Rd_0lki[k,i,l]) k^0 k^i")
    print()

    # Check element by element
    for k in range(2):
        for l in range(2):
            for i in range(2):
                v_4d = -(Rd_0lki[k,l,i] + Rd_0lki[l,k,i])
                v_user = Rd_0lki[l,k,i] - Rd_0lki[k,i,l]
                match = "✅" if abs(v_4d - v_user) < 1e-40 * max(abs(v_4d), 1) else "❌"
                print(f"    (k,l,i)=({k},{l},{i}): 4D={v_4d:.4e}  user={v_user:.4e}  {match}")
    print()

    # For term3: R_{k+1,i+1,l+1,j+1}
    # R_{k+1,i+1,l+1,j+1} = -R_{k+1,i+1,j+1,l+1}  (antisym 2nd pair)
    # = -Rd_kijl[k,i,j,l]
    # Contribution: s_A^k s_B^l (-Rd_kijl[k,i,j,l]) k^i k^j
    # User formula: s_A^k s_B^l Rd_kijl[k,i,j,l] k^i k^j
    # → term3_4d = -user_t3

    print(f"  R[k+1,i+1,l+1,j+1] = -Rd_kijl[k,i,j,l] (antisym 2nd pair)")
    print(f"  → 4D term3 = -user_t3. Ratio = {t3[0,0]/user_t3[0,0]:.1f}")
    print()

    # =================================================================
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # The 4D formula expands as:
    # R_{AB}^{4D} = s_A^k s_B^l [ -R_{k00l} k0² + cross_terms + (-R_{kijl} ki kj) ]
    # i.e. R_{AB}^{4D} = -R_{AB}^{user_formula}
    #
    # This means the user's formula has the OPPOSITE SIGN convention.
    # The difference is in R_{μανβ} vs R_{μαβν} slot ordering.

    print(f"  R_AB (full 4D):      {R_AB_4d[0,0]:.6e}")
    print(f"  R_AB (user formula): {R_AB_spatial[0,0]:.6e}")
    print(f"  Ratio:               {R_AB_4d[0,0] / R_AB_spatial[0,0]:.6f}")
    print()

    if abs(R_AB_4d[0,0] / R_AB_spatial[0,0] + 1.0) < 0.01:
        print("  ⚠️  SIGN CONVENTION DIFFERENCE:")
        print("  The user's formula uses R_{AB} = s_A^k s_B^l R_{kανβ} k^α k^β")
        print("  with the SLOT ORDER   R_{μ α ν β}  →  R_{k 0 0 l}")
        print("  But the 4D expansion of R_{μανβ} k^α k^β with spatial-only e_A gives:")
        print("  s_A^k s_B^l R_{k+1, α, l+1, β} k^α k^β")
        print("  = s_A^k s_B^l [ R_{k+1,0,l+1,0}k0² + ... + R_{k+1,i+1,l+1,j+1}ki kj ]")
        print("  = s_A^k s_B^l [ -R_{k00l}k0² + ... + (-R_{kijl})ki kj ]")
        print("  = -( user formula )")
        print()
        print("  This is because  R_{k+1,0,l+1,0} = -R_{k00l}  (antisym 2nd pair).")
        print("  In the user's formula, the index convention is R_{k,α=0,β=0,l}")
        print("  = R_{k00l}  (slot positions 1,2,3,4 → μ,α,β,ν)")
        print("  while the code uses slot positions → μ,α,ν,β.")
        print()
        print("  KEY QUESTION: does the Jacobi equation use the same sign convention")
        print("  as the code?  If yes, the code is internally consistent and correct.")
        print("  The user's formula simply uses a different slot ordering.")

    # Final check: code vs 4D reference
    print()
    print(f"  CODE optimized vs 4D ref:   |err|/scale = {np.max(np.abs(R_AB_opt - R_AB_4d))/scale:.2e}")
    print(f"  CODE brute-force vs 4D ref: |err|/scale = {np.max(np.abs(R_AB_brute - R_AB_4d))/scale:.2e}")

    code_ok = np.max(np.abs(R_AB_opt - R_AB_4d))/scale < 1e-14
    print(f"  → Code matches 4D reference: {'✅ YES' if code_ok else '❌ NO'}")


if __name__ == '__main__':
    main()
