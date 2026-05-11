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

For a varying observer congruence u^\mu(v), the Sachs basis should satisfy
the screen-projected transport law

    S^\mu_\nu D e_A^\nu / d\lambda = 0

rather than full parallel transport.  In conformal FLRW coordinates with the
comoving observer congruence, this is implemented below as the usual
coordinate-space parallel-transport RHS plus a correction along k^\mu that
keeps e_A^\mu inside the screen.

This module provides:
    - Initialization of the Sachs basis at the observer.
    - RHS for full parallel transport (kept as a low-level helper).
    - RHS for screen-projected Sachs transport used by the lensing solvers.
"""

import numpy as np
from numba import njit


_SCREEN_CONVENTIONS = (
    "metric",
    "physical_metric",
    "conformal_metric",
    "euclidean_local",
)


def _pick_seed(direction, dot_fn):
    """Pick the coordinate axis that is most transverse to the supplied direction."""
    seed_candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    best_seed = seed_candidates[0]
    best_dot = abs(dot_fn(best_seed, direction))
    for seed in seed_candidates[1:]:
        dot_val = abs(dot_fn(seed, direction))
        if dot_val < best_dot:
            best_dot = dot_val
            best_seed = seed
    return best_seed


def _build_transverse_basis(k_spatial, g_ij, convention):
    """Construct the local 3D transverse basis for the requested screen convention."""
    if convention in ("metric", "physical_metric", "conformal_metric"):
        def dot_fn(lhs, rhs):
            return lhs @ g_ij @ rhs

        k_norm_sq = dot_fn(k_spatial, k_spatial)
        if k_norm_sq <= 0.0:
            raise ValueError("Spatial photon momentum has non-positive metric norm squared.")
        n_hat = k_spatial / np.sqrt(k_norm_sq)
        best_seed = _pick_seed(n_hat, dot_fn)

        v1 = best_seed.copy()
        v1 -= dot_fn(v1, n_hat) / dot_fn(n_hat, n_hat) * n_hat
        v1 /= np.sqrt(dot_fn(v1, v1))

        v2 = np.cross(n_hat, v1)
        v2 -= dot_fn(v2, n_hat) / dot_fn(n_hat, n_hat) * n_hat
        v2 -= dot_fn(v2, v1) / dot_fn(v1, v1) * v1
        v2_norm = np.sqrt(dot_fn(v2, v2))
        if v2_norm < 1e-15:
            raise ValueError("Degenerate Sachs basis: v2 has zero metric norm.")
        v2 /= v2_norm
        return v1, v2

    if convention == "euclidean_local":
        k_norm = np.linalg.norm(k_spatial)
        if k_norm <= 0.0:
            raise ValueError("Spatial photon momentum has non-positive Euclidean norm.")
        n_hat = k_spatial / k_norm
        best_seed = _pick_seed(n_hat, np.dot)

        v1 = best_seed.copy()
        v1 -= np.dot(v1, n_hat) * n_hat
        v1 /= np.linalg.norm(v1)

        v2 = np.cross(n_hat, v1)
        v2 -= np.dot(v2, n_hat) * n_hat
        v2 -= np.dot(v2, v1) * v1
        v2_norm = np.linalg.norm(v2)
        if v2_norm < 1e-15:
            raise ValueError("Degenerate Euclidean screen basis: v2 has zero norm.")
        v2 /= v2_norm
        return v1, v2

    raise ValueError(
        f"Unsupported screen convention '{convention}'. Expected one of {_SCREEN_CONVENTIONS}."
    )


# ------------------------------------------------------------------
#  Initialization at the observer
# ------------------------------------------------------------------

def init_sachs_basis(k_mu, g_mu_nu, a, convention="metric"):
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

    convention : {"metric", "physical_metric", "conformal_metric", "euclidean_local"}, optional
        Local screen convention used to build the spatial basis before lifting
        it to 4-vectors.

        ``metric`` and ``physical_metric`` preserve the historical behaviour:
        the screen is the Sachs basis orthonormalized in the physical spatial
        metric g_ij.

        ``conformal_metric`` builds the Sachs basis in the conformal metric
        g_tilde = g / a^2, matching Fleury's conformal dictionary for beam
        optics in FLRW cosmology.

        ``euclidean_local`` instead uses the coordinate Euclidean norm in
        3-space.  This is a non-Sachs diagnostic projection surface useful for
        thin-lens comparisons, not the GR Sachs basis.

    Returns
    -------
    e1 : ndarray (4,)
        First Sachs screen vector e_1^mu.
    e2 : ndarray (4,)
        Second Sachs screen vector e_2^mu.

    Notes
    -----
    The default ``metric`` convention matches the historical production path.
    ``conformal_metric`` provides Fleury's conformal Sachs basis for FLRW beam
    optics.  ``euclidean_local`` is a deliberate non-Sachs diagnostic surface:
    it uses Euclidean 3-space orthogonality before the final 4D lift, so it can
    be compared directly to thin-lens local screen constructions.
    """
    if convention not in _SCREEN_CONVENTIONS:
        raise ValueError(
            f"Unsupported screen convention '{convention}'. Expected one of {_SCREEN_CONVENTIONS}."
        )

    k_spatial = k_mu[1:4].copy()
    if convention == "conformal_metric":
        screen_g_mu_nu = g_mu_nu / (a * a)
        basis_convention = "conformal_metric"
    else:
        screen_g_mu_nu = g_mu_nu
        basis_convention = convention

    screen_g_ij = screen_g_mu_nu[1:4, 1:4]
    v1, v2 = _build_transverse_basis(k_spatial, screen_g_ij, basis_convention)

    # We need  g_mu_nu e^mu k^nu = 0  and  g_mu_nu e^mu u^nu = 0.
    # For a diagonal FLRW metric:  g_{00} e^0 k^0 + g_{ij} e^i k^j = 0
    #   =>  e^0 = -g_{ij} e^i k^j / (g_{00} k^0)
    lift_g_mu_nu = screen_g_mu_nu if convention == "conformal_metric" else g_mu_nu
    lift_g_ij = lift_g_mu_nu[1:4, 1:4]
    g00 = lift_g_mu_nu[0, 0]
    k0 = k_mu[0]

    if abs(g00 * k0) < 1e-30:
        raise ValueError("g_{00} k^0 ~ 0: cannot lift spatial Sachs vectors.")

    e1_0 = -(lift_g_ij @ v1) @ k_spatial / (g00 * k0)
    e2_0 = -(lift_g_ij @ v2) @ k_spatial / (g00 * k0)

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
def screen_projected_sachs_transport_rhs(e_mu, christoffel, k_mu, g_mu_nu):
    r"""
    Coordinate-space RHS for the screen-projected Sachs transport law.

    We follow the Sachs-basis transport condition used in Fleury's notes,

        S^\mu_\nu \nabla_k e_A^\nu = 0,

    instead of full parallel transport.  A convenient representative is

        \nabla_k e_A^\mu = \alpha_A k^\mu,

    where \alpha_A is fixed by preserving the observer orthogonality
    e_A^\mu u_\mu = 0 along the beam.  For the comoving observer congruence in
    these coordinates, u^\mu is proportional to \delta^\mu_0, so

        \alpha_A = - (e_A \cdot \nabla_k u) / (u \cdot k).

    The coordinate derivative is therefore the full parallel-transport RHS
    plus a correction along k^\mu.
    """
    de = sachs_transport_rhs(e_mu, christoffel, k_mu)

    # In conformal FLRW coordinates, the comoving observer congruence is
    # proportional to delta^mu_0.  The overall scalar factor is irrelevant
    # for alpha_A because it cancels between numerator and denominator.
    nabla_k_u = np.zeros(4)
    for mu in range(4):
        s = 0.0
        for sigma in range(4):
            s += christoffel[mu, 0, sigma] * k_mu[sigma]
        nabla_k_u[mu] = s

    u_dot_k = 0.0
    for nu in range(4):
        u_dot_k += g_mu_nu[0, nu] * k_mu[nu]

    if abs(u_dot_k) < 1e-30:
        return de

    e_dot_nabla_u = 0.0
    for mu in range(4):
        e_cov_mu = 0.0
        for nu in range(4):
            e_cov_mu += g_mu_nu[mu, nu] * e_mu[nu]
        e_dot_nabla_u += e_cov_mu * nabla_k_u[mu]

    alpha = -e_dot_nabla_u / u_dot_k
    for mu in range(4):
        de[mu] += alpha * k_mu[mu]
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
