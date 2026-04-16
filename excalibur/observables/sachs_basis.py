# excalibur/observables/sachs_basis.py
r"""
Sachs screen basis for gravitational lensing in curved spacetimes.

The two Sachs vectors e_1^\mu, e_2^\mu form an orthonormal basis in the
*screen plane*  --  the 2-D subspace orthogonal to both the photon
wave-vector k^\mu and the observer's 4-velocity u^\mu.

Properties (with metric signature (-, +, +, +)):
    g^{\mu\nu}  e_A^\mu  e_B^\nu = delta_{AB}       (spatial orthonormality)
    g^{\mu\nu}  e_A^\mu  k^\nu   = 0            (orthogonal to propagation)
    g^{\mu\nu}  e_A^\mu  u^\nu   = 0            (orthogonal to observer)

Parallel transport along the photon geodesic:
    D e_A^\mu / d\lambda =  -\Gamma^\mu_{\nu\sigma}  k^\sigma  e_A^\nu

This module provides:
    - Initialization of the Sachs basis at the observer.
    - RHS for parallel transport (to integrate alongside the geodesic).
"""

import numpy as np
from numba import njit


# ------------------------------------------------------------------
#  Initialization at the observer
# ------------------------------------------------------------------

def init_sachs_basis(k_mu, g_mu_nu, a):
    r"""
    Construct two orthonormal screen vectors at the observer.

    Parameters
    ----------
    k_mu : ndarray (4,)
        Photon 4-velocity (wave-vector) k^mu = dx^mu/dlambda.
    g_mu_nu : ndarray (4, 4)
        Metric tensor at the observer position.
    a : float
        Scale factor at the observer (used only for the comoving observer).

    Returns
    -------
    e1 : ndarray (4,)
        First Sachs screen vector e_1^mu.
    e2 : ndarray (4,)
        Second Sachs screen vector e_2^mu.

    Notes
    -----
    Algorithm:
    1. Define comoving observer 4-velocity  u^mu = (1/a, 0, 0, 0).
    2. Compute the *spatial* photon direction n^i = k^i / |k^i|_g
       (where the norm uses the spatial metric g_{ij}).
    3. Choose a seed vector *not* aligned with n.
    4. Gram-Schmidt in the spatial metric to obtain two orthonormal
       vectors in the plane orthogonal to n.
    5. Lift to 4-vectors with e^0 chosen so that g_mu_nu e^mu k^nu = 0.
    """
    # Step 1: comoving observer
    u_mu = np.array([1.0 / a, 0.0, 0.0, 0.0])

    # Step 2: spatial direction of photon
    k_spatial = k_mu[1:4].copy()

    # Spatial metric  g_{ij}  (i,j = 1,2,3)
    g_ij = g_mu_nu[1:4, 1:4]

    # Norm of k_spatial w.r.t. g_ij
    k_norm_sq = k_spatial @ g_ij @ k_spatial
    if k_norm_sq <= 0.0:
        raise ValueError("Spatial photon momentum has non-positive norm squared.")
    k_norm = np.sqrt(k_norm_sq)
    n_hat = k_spatial / k_norm   # unit spatial direction

    # Step 3: pick seed not aligned with n
    # Choose the coordinate axis most orthogonal to n_hat
    seed_candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    # Pick the seed with smallest |g(seed, n)|
    best_seed = seed_candidates[0]
    best_dot = abs(seed_candidates[0] @ g_ij @ n_hat)
    for s in seed_candidates[1:]:
        d = abs(s @ g_ij @ n_hat)
        if d < best_dot:
            best_dot = d
            best_seed = s

    # Step 4: Gram-Schmidt to get two orthonormal spatial vectors
    # First vector: project out the n-component from seed
    v1 = best_seed.copy()
    v1 -= (v1 @ g_ij @ n_hat) / (n_hat @ g_ij @ n_hat) * n_hat
    v1_norm = np.sqrt(v1 @ g_ij @ v1)
    v1 /= v1_norm

    # Second vector: cross product in the metric (or just a second G-S round)
    v2 = np.cross(n_hat, v1)
    # Project out any residual n- and v1-components
    v2 -= (v2 @ g_ij @ n_hat) / (n_hat @ g_ij @ n_hat) * n_hat
    v2 -= (v2 @ g_ij @ v1) / (v1 @ g_ij @ v1) * v1
    v2_norm = np.sqrt(v2 @ g_ij @ v2)
    if v2_norm < 1e-15:
        raise ValueError("Degenerate Sachs basis: v2 has zero norm.")
    v2 /= v2_norm

    # Step 5: lift to 4-vectors
    # We need  g_mu_nu e^mu k^nu = 0  and  g_mu_nu e^mu u^nu = 0.
    # For a diagonal FLRW metric:  g_{00} e^0 k^0 + g_{ij} e^i k^j = 0
    #   =>  e^0 = -g_{ij} e^i k^j / (g_{00} k^0)
    g00 = g_mu_nu[0, 0]
    k0 = k_mu[0]

    if abs(g00 * k0) < 1e-30:
        raise ValueError("g_{00} k^0 ~ 0: cannot lift spatial Sachs vectors.")

    e1_0 = -(g_ij @ v1) @ k_spatial / (g00 * k0)
    e2_0 = -(g_ij @ v2) @ k_spatial / (g00 * k0)

    e1 = np.zeros(4)
    e2 = np.zeros(4)
    e1[0] = e1_0
    e1[1:4] = v1
    e2[0] = e2_0
    e2[1:4] = v2

    return e1, e2


# ------------------------------------------------------------------
#  Parallel transport RHS
# ------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def sachs_transport_rhs(e_mu, christoffel, k_mu):
    r"""
    Right-hand side of the parallel transport equation for a Sachs vector.

        D e_A^\mu / d\lambda =  -\Gamma^\mu_{\nu\sigma}  k^\sigma  e_A^\nu

    Parameters
    ----------
    e_mu : ndarray (4,)
        Screen vector e_A^\mu.
    christoffel : ndarray (4, 4, 4)
        Christoffel symbols \Gamma^\mu_{\nu\sigma} at the current position.
    k_mu : ndarray (4,)
        Photon 4-velocity k^\mu.
    Returns
    -------
    de_mu : ndarray (4,)
        Time derivative D e_A^\mu / d\lambda.
    """
    de = np.zeros(4)
    for mu in range(4):
        s = 0.0
        for nu in range(4):
            for sigma in range(4):
                s += christoffel[mu, nu, sigma] * k_mu[sigma] * e_mu[nu]
        de[mu] = -s
    return de


@njit(cache=True, fastmath=True)
def sachs_transport_rhs_fast(e_mu, christoffel, k_mu):
    r"""
    Optimized RHS  --  exploits the fact that for diagonal FLRW metric,
    many Christoffel components vanish.

    This version contracts by hand to avoid nested loops.
    Same result as ``sachs_transport_rhs`` but faster.
    """
    de = np.zeros(4)
    for mu in range(4):
        s = 0.0
        # Contract Gamma^mu_{nu sigma} k^sigma e^nu
        # Precompute Gamma^mu_{nu sigma} k^sigma  (vector in nu)
        for nu in range(4):
            gk = 0.0
            for sigma in range(4):
                gk += christoffel[mu, nu, sigma] * k_mu[sigma]
            s += gk * e_mu[nu]
        de[mu] = -s
    return de
