# excalibur/observables/optical_tidal_matrix.py
r"""
Optical tidal matrix (Sachs optical scalars) for gravitational lensing.

The 2×2 optical tidal matrix is:

    R_{AB} = R_{\mu\alpha\nu\beta}  k^\alpha k^\beta  e_A^\mu  e_B^\nu

where:
    - R_{\mu\alpha\nu\beta} is the Riemann tensor with *all indices down*.
    - k^\mu is the photon 4-velocity.
    - e_A^\mu  (A = 1, 2) are the two Sachs screen vectors.

This matrix drives the Sachs equation (Jacobi map evolution):

    d²D_{AB}/dλ² = -R_{AC} D_{CB}

or equivalently, defining P = dD/dλ:

    dD/dλ = P,    dP/dλ = -R · D

The minus sign follows from the geodesic deviation equation
d²ξ^μ/dλ² = -R^μ_{αβν} k^α k^β ξ^ν  (MTW 11.10).

Physical decomposition (of the effective tidal tensor T = -R):
    - Convergence κ = -1/2 tr(T) = + 1/2 tr(R)  — related to Ricci focusing (∇²Φ)
    - Shear γ₁ + iγ₂                       — related to Weyl (tidal) focusing

This module provides:
    - ``optical_tidal_matrix``  : compute R_{AB} from Riemann blocks + Sachs basis.
    - ``jacobi_rhs``           : RHS for the 2×2 Jacobi map ODE.
"""

import numpy as np
from numba import njit


# ------------------------------------------------------------------
#  Riemann symmetry helper
# ------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _set_riemann_sym(R, a, b, c, d, val):
    r"""
    Set R[a,b,c,d] = val and all 7 symmetry partners of the Riemann tensor:

        R_{abcd} = -R_{bacd}    (antisym first pair)
        R_{abcd} = -R_{abdc}    (antisym second pair)
        R_{abcd} =  R_{cdab}    (pair symmetry)

    This overwrites any previous value, so it is safe to call repeatedly
    for non-overlapping Riemann blocks (which our three blocks are).
    """
    R[a, b, c, d] = val
    R[b, a, c, d] = -val
    R[a, b, d, c] = -val
    R[b, a, d, c] = val
    R[c, d, a, b] = val
    R[d, c, a, b] = -val
    R[c, d, b, a] = -val
    R[d, c, b, a] = val


# ------------------------------------------------------------------
#  Full 4-index Riemann → R_{AB} contraction
# ------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def optical_tidal_matrix_from_blocks(
    Rd_k00l,    # (3, 3)          R_{k00l}   (all indices down)
    Rd_0lki,    # (3, 3, 3)       R_{0lki}   (all indices down)
    Rd_kijl,    # (3, 3, 3, 3)    R_{kijl}   (all indices down)
    k_mu,       # (4,)            photon 4-velocity
    e1_mu,      # (4,)            Sachs vector 1
    e2_mu,      # (4,)            Sachs vector 2
    g_mu_nu,    # (4, 4)          metric tensor  (unused — kept for API compat)
):
    r"""
    Compute the 2x2 optical tidal matrix R_{AB}.

    The Riemann blocks are already fully covariant (all indices down),
    so the contraction is direct:

        R_{AB} = R_{\mu\alpha\nu\beta} k^\alpha k^\beta e_A^\mu e_B^\nu

    We build the full R_{\mu\alpha\nu\beta} from the three spatial blocks
    plus antisymmetry  R_{\mu\alpha\nu\beta} = -R_{\mu\alpha\beta\nu}.

    Parameters
    ----------
    Rd_k00l : ndarray (3, 3)
        R_{k00l} with spatial indices k,l ∈ {0,1,2}.
    Rd_0lki : ndarray (3, 3, 3)
        R_{0lki} with spatial indices l,k,i ∈ {0,1,2}.
    Rd_kijl : ndarray (3, 3, 3, 3)
        R_{kijl} with spatial indices k,i,j,l ∈ {0,1,2}.
    k_mu, e1_mu, e2_mu : ndarray (4,)
    g_mu_nu : ndarray (4, 4)
        Metric tensor (unused — blocks are already all-down).

    Returns
    -------
    R_AB : ndarray (2, 2)
        Optical tidal matrix.
    """
    # Build the full all-down Riemann tensor R_{mu alpha nu beta}
    # from the three spatial blocks, applying ALL Riemann symmetries:
    #   1. R_{abcd} = -R_{bacd}       (antisymmetry in first pair)
    #   2. R_{abcd} = -R_{abdc}       (antisymmetry in second pair)
    #   3. R_{abcd} =  R_{cdab}       (pair symmetry)
    #
    # Block mapping (spatial index i maps to 4-index i+1):
    #   R_{k,0,0,l}  → R_down[k+1, 0, 0, l+1]
    #   R_{0,l,k,i}  → R_down[0, l+1, k+1, i+1]
    #   R_{k,i,j,l}  → R_down[k+1, i+1, j+1, l+1]
    #
    # The three blocks have no overlapping components (verified numerically),
    # so set_sym can be applied independently for each block.

    R_down = np.zeros((4, 4, 4, 4))

    # Block 1:  R_{k, 0, 0, l}  (k,l spatial)
    for k in range(3):
        for l in range(3):
            val = Rd_k00l[k, l]
            _set_riemann_sym(R_down, k+1, 0, 0, l+1, val)

    # Block 2:  R_{0, l, k, i}  (l,k,i spatial)
    for l in range(3):
        for k in range(3):
            for i in range(3):
                val = Rd_0lki[l, k, i]
                _set_riemann_sym(R_down, 0, l+1, k+1, i+1, val)

    # Block 3:  R_{k, i, j, l}  (k,i,j,l spatial)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    val = Rd_kijl[k, i, j, l]
                    _set_riemann_sym(R_down, k+1, i+1, j+1, l+1, val)

    # Contract:  R_{AB} = R_{mu alpha nu beta} k^alpha k^beta e_A^mu e_B^nu
    e = np.empty((2, 4))
    e[0, :] = e1_mu
    e[1, :] = e2_mu

    R_AB = np.zeros((2, 2))
    for A in range(2):
        for B in range(2):
            s = 0.0
            for mu in range(4):
                for alpha in range(4):
                    for nu in range(4):
                        for beta in range(4):
                            s += (R_down[mu, alpha, nu, beta]
                                  * k_mu[alpha] * k_mu[beta]
                                  * e[A, mu] * e[B, nu])
            R_AB[A, B] = s

    return R_AB


@njit(cache=True, fastmath=True)
def optical_tidal_matrix_optimized(
    Rd_k00l, Rd_0lki, Rd_kijl,
    k_mu, e1_mu, e2_mu, g_mu_nu,
):
    r"""
    Same as ``optical_tidal_matrix_from_blocks`` but uses pre-contraction
    with k^α, k^β before the final e_A, e_B contraction.

    The Riemann blocks are fully covariant (all indices down).

    Strategy — starting from
        R_{AB} = R_{\mu\alpha\nu\beta} k^\alpha k^\beta e_A^\mu e_B^\nu

    we define the pre-contracted 2-tensor (all-down):

        T_{\mu\nu} = \sum_{\alpha,\beta} R_{\mu\alpha\nu\beta} k^\alpha k^\beta

    (contraction over 2nd and 4th slots of R_{\mu\alpha\nu\beta}).

    Then  R_{AB} = T_{\mu\nu} e_A^\mu e_B^\nu.

    The full Riemann tensor is built from the three blocks with all
    symmetries (antisymmetry in each index pair + pair symmetry) before
    contraction to ensure ALL cross-term contributions are included.

    Parameters
    ----------
    Rd_k00l : ndarray (3, 3)
        R_{k00l} (all indices down).
    Rd_0lki : ndarray (3, 3, 3)
        R_{0lki} (all indices down).
    Rd_kijl : ndarray (3, 3, 3, 3)
        R_{kijl} (all indices down).
    """
    # Build the full 4×4×4×4 Riemann with all symmetries
    R_down = np.zeros((4, 4, 4, 4))

    for k in range(3):
        for l in range(3):
            _set_riemann_sym(R_down, k+1, 0, 0, l+1, Rd_k00l[k, l])

    for l in range(3):
        for k in range(3):
            for i in range(3):
                _set_riemann_sym(R_down, 0, l+1, k+1, i+1, Rd_0lki[l, k, i])

    for k in range(3):
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    _set_riemann_sym(R_down, k+1, i+1, j+1, l+1,
                                     Rd_kijl[k, i, j, l])

    # Pre-contract: T_{mu,nu} = R_{mu,alpha,nu,beta} k^alpha k^beta
    T_down = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            s = 0.0
            for alpha in range(4):
                for beta in range(4):
                    s += R_down[mu, alpha, nu, beta] * k_mu[alpha] * k_mu[beta]
            T_down[mu, nu] = s

    # Final contraction:  R_{AB} = T_{mu nu} e_A^mu e_B^nu
    e = np.empty((2, 4))
    e[0, :] = e1_mu
    e[1, :] = e2_mu

    R_AB = np.zeros((2, 2))
    for A in range(2):
        for B in range(2):
            s = 0.0
            for mu in range(4):
                for nu in range(4):
                    s += T_down[mu, nu] * e[A, mu] * e[B, nu]
            R_AB[A, B] = s

    return R_AB


# ------------------------------------------------------------------
#  Jacobi map ODE
# ------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def jacobi_rhs(D_flat, R_AB):
    r"""
    Right-hand side for the Jacobi map equation (geodesic deviation):

        dD/dλ = P
        dP/dλ = -R · D

    where D and P are 2×2 matrices stored as flat (4,) arrays
    in row-major order:  [D_11, D_12, D_21, D_22].

    The minus sign follows from the standard geodesic deviation
    equation  d²ξ^μ/dλ² = -R^μ_{αβν} k^α k^β ξ^ν  (MTW 11.10),
    projected onto the Sachs screen basis.

    Parameters
    ----------
    D_flat : ndarray (8,)
        Flat array [D_11, D_12, D_21, D_22, P_11, P_12, P_21, P_22].
    R_AB : ndarray (2, 2)
        Optical tidal matrix  R_{AB} = R_{μανβ} k^α k^β e_A^μ e_B^ν.

    Returns
    -------
    dstate : ndarray (8,)
        Time derivatives [dD/dλ, dP/dλ] = [P, -R·D].
    """
    # Unpack
    D11, D12, D21, D22 = D_flat[0], D_flat[1], D_flat[2], D_flat[3]
    P11, P12, P21, P22 = D_flat[4], D_flat[5], D_flat[6], D_flat[7]

    # dD/dλ = P
    # dP/dλ = -R · D
    #
    # The minus sign comes from the geodesic deviation equation
    #   d²ξ^μ/dλ² = -R^μ_{αβν} k^α k^β ξ^ν       [MTW eq. 11.10]
    # projected onto the Sachs screen basis, where
    #   R_{AB} = R_{μανβ} k^α k^β e_A^μ e_B^ν .
    #
    # (-R · D)_{AC} = -sum_B R_{AB} D_{BC}
    dP11 = -(R_AB[0, 0] * D11 + R_AB[0, 1] * D21)
    dP12 = -(R_AB[0, 0] * D12 + R_AB[0, 1] * D22)
    dP21 = -(R_AB[1, 0] * D11 + R_AB[1, 1] * D21)
    dP22 = -(R_AB[1, 0] * D12 + R_AB[1, 1] * D22)

    dstate = np.empty(8)
    dstate[0] = P11
    dstate[1] = P12
    dstate[2] = P21
    dstate[3] = P22
    dstate[4] = dP11
    dstate[5] = dP12
    dstate[6] = dP21
    dstate[7] = dP22
    return dstate


# ------------------------------------------------------------------
#  Optical scalar extraction
# ------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def optical_scalars_from_tidal(R_AB):
    r"""
    Extract convergence and shear from the optical tidal matrix.

    The effective tidal tensor in the Sachs equation dP/dλ = T·D is
    T = -R_{AB}.  The standard decomposition of T is:

        κ = -½ tr(T) = +½ tr(R)
        γ₁ = -½ (T_{11} - T_{22}) = +½ (R_{11} - R_{22})
        γ₂ = -T_{12} = +R_{12}

    Parameters
    ----------
    R_AB : ndarray (2, 2)

    Returns
    -------
    kappa : float
        Convergence  κ = +½ tr(R)
    gamma1 : float
        Shear component  γ₁ = +½ (R_{11} - R_{22})
    gamma2 : float
        Shear component  γ₂ = +R_{12}    (= +R_{21} by symmetry)
    omega : float
        Rotation ω = ½(R_{12} - R_{21})  (should vanish for geodesic light)
    """
    kappa = 0.5 * (R_AB[0, 0] + R_AB[1, 1])
    gamma1 = 0.5 * (R_AB[0, 0] - R_AB[1, 1])
    gamma2 = R_AB[0, 1]
    omega = 0.5 * (R_AB[0, 1] - R_AB[1, 0])
    return kappa, gamma1, gamma2, omega


@njit(cache=True, fastmath=True)
def lensing_from_jacobi(D_flat):
    r"""
    Extract lensing observables from the Jacobi map matrix D.

    The standard Jacobi initial conditions are  D(0) = 0, P(0) = I.
    The caller must normalise D by the affine distance λ_S
    (i.e. pass D_flat / λ_S) so that the unlensed beam corresponds
    to D_norm = I.

    Parameters
    ----------
    D_flat : ndarray (4,)
        [D_11, D_12, D_21, D_22]  — normalised by λ_S.

    Returns
    -------
    convergence : float
        κ = 1 - ½ tr(D) / D_FRW   (requires normalization by caller)
    magnification : float
        μ = 1 / det(D)
    shear_magnitude : float
        |γ|
    """
    D11, D12, D21, D22 = D_flat[0], D_flat[1], D_flat[2], D_flat[3]
    det_D = D11 * D22 - D12 * D21
    magnification = 1.0 / det_D if abs(det_D) > 1e-30 else 0.0

    # Trace and tracefree part
    tr_D = D11 + D22
    # Convergence (relative to identity): κ = 1 - tr(D)/2
    convergence = 1.0 - 0.5 * tr_D

    # Shear from off-diagonal/trace-free
    gamma1 = 0.5 * (D11 - D22)
    gamma2 = 0.5 * (D12 + D21)
    shear_magnitude = np.sqrt(gamma1**2 + gamma2**2)

    return convergence, magnification, shear_magnitude


# ------------------------------------------------------------------
#  Angular-diameter distance from the Jacobi map
# ------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def angular_diameter_distance_from_jacobi(D_flat_raw):
    r"""
    Angular-diameter distance extracted from the **raw** Jacobi map.

    The Jacobi map  D_{AB}  maps the initial beam opening angle (in rad)
    to the physical transverse separation at the source.  Its determinant
    gives the solid-angle–to–area mapping, hence:

    .. math::
        D_A = \sqrt{|\det D_{AB}|}

    This is the *true* (lensed) angular-diameter distance — it includes
    all perturbations along the ray.

    Parameters
    ----------
    D_flat_raw : ndarray (4,)
        **Un-normalised** Jacobi map at the source:
        ``[D_11, D_12, D_21, D_22]``  in the same length units
        as the simulation (metres for SI).

    Returns
    -------
    D_A : float
        Angular-diameter distance (same units as input, e.g. metres).
    """
    D11 = D_flat_raw[0]
    D12 = D_flat_raw[1]
    D21 = D_flat_raw[2]
    D22 = D_flat_raw[3]
    det_D = D11 * D22 - D12 * D21
    return np.sqrt(abs(det_D))


def distance_comparison(D_flat_raw, z_source, cosmology):
    r"""
    Compare the ray-traced angular-diameter distance with the FLRW background.

    Parameters
    ----------
    D_flat_raw : ndarray (4,)
        **Un-normalised** Jacobi map ``[D_11, D_12, D_21, D_22]``.
    z_source : float
        Redshift of the source.
    cosmology : LCDM_Cosmology
        Cosmology object with ``angular_diameter_distance(z)`` method.

    Returns
    -------
    result : dict
        ``D_A_ray``   – angular-diameter distance from the Jacobi map (metres).
        ``D_A_FLRW``  – background FLRW angular-diameter distance (metres).
        ``delta_D_A``  – relative difference  (D_A_ray - D_A_FLRW) / D_A_FLRW.
    """
    D_A_ray = float(angular_diameter_distance_from_jacobi(D_flat_raw))
    D_A_FLRW = float(cosmology.angular_diameter_distance(z_source))
    delta = (D_A_ray - D_A_FLRW) / D_A_FLRW if abs(D_A_FLRW) > 0 else 0.0
    return {
        "D_A_ray": D_A_ray,
        "D_A_FLRW": D_A_FLRW,
        "delta_D_A": delta,
    }
