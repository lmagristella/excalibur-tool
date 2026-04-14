# excalibur/observables/riemann_perturbed_flrw.py
r"""
Minimal Riemann tensor blocks for the perturbed FLRW metric.

Metric (conformal Newtonian gauge, psi = phi):
    ds² = a²(eta) [ -(1 + 2psi/c²) c² deta²  +  (1 - 2phi/c²) delta_ij dx^i dx^j ]

Throughout this module:
    - Greek indices mu,nu,… ∈ {0,1,2,3} where 0 = conformal time eta.
    - Latin indices i,j,k,l ∈ {1,2,3} (spatial).
    - H = a'/a  is the conformal Hubble parameter (prime = d/deta).
    - H' = dH/deta.
    - phi' = ∂phi/∂eta,  phi'' = ∂²phi/∂eta².
    - d_i phi  = spatial gradient component.
    - d_i d_j phi = spatial Hessian component.
    - d_i phi' = mixed derivative  ∂²phi/(∂x^i ∂eta).

The three independent blocks of the Riemann tensor (with *all indices
down*, R_{alpha beta gamma delta}) needed for the optical tidal matrix are:

    R_{k00l}, R_{0lki}, R_{kijl}

All formulas assume psi = phi (no anisotropic stress).

Convention: Returned arrays use *spatial* indices 0,1,2 ↔ x,y,z.
All blocks are COVARIANT (all indices down) — no index raising needed
before the contraction  R_{AB} = R_{alpha beta mu nu} k^alpha k^beta e_A^mu e_B^nu.
Considering that the Sachs basis vectors are spatial only (e_A^0 = 0), we only need the spatial components of 
the Riemann tensor. R_{AB} becomes 
R_
"""

import numpy as np
from numba import njit

# ------------------------------------------------------------------
#  Core kernel — returns the three blocks as flat tuples
# ------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def riemann_blocks_kernel(
    a,              # scale factor
    H,              # conformal Hubble  H = a'/a  [s^{-1}]
    Hprime,         # dH/dη                       [s^{-2}]
    phi,            # Φ  (SI: m²/s²)
    phi_dot,        # ∂Φ/∂η  (SI: m²/s³)
    phi_ddot,       # ∂²Φ/∂η² (SI: m²/s⁴)
    grad_phi,       # (∂_x Φ, ∂_y Φ, ∂_z Φ)  each m/s²  (length-3 array)
    grad_phi_dot,   # (∂_x Φ', ∂_y Φ', ∂_z Φ')  each m/s³ (length-3 array)
    hess_phi,       # spatial Hessian ∂_i ∂_j Φ  (3×3 symmetric, m/s² /m = 1/s²)
    c_val,          # speed of light
):
    r"""
    Compute the three covariant Riemann blocks for perturbed FLRW (Ψ = Φ).

    Returns
    -------
    Rd_k00l : ndarray (3, 3)
        R_{k00l}  (all indices down) with spatial indices k,l ∈ {0,1,2}.
    Rd_0lki : ndarray (3, 3, 3)
        R_{0lki}  (all indices down) with spatial indices l,k,i ∈ {0,1,2}.
    Rd_kijl : ndarray (3, 3, 3, 3)
        R_{kijl}  (all indices down) with spatial indices k,i,j,l ∈ {0,1,2}.

    Notes
    -----
    The formulas (Ψ = Φ, all indices down):

    R_{k00l} = a² [ -∂_k ∂_l Ψ
                      + δ_{kl} (H'(1 - 2Ψ/c²) + Ψ''/c² + H(Φ'+Ψ')/c²) ]

    R_{0lki} = (a²/c²) [ δ_{il}(∂_k Ψ' + H ∂_k Φ)
                          -δ_{lk}(∂_i Ψ' + H ∂_i Φ) ]

    R_{kijl} = (a²/c²) [ δ_{kj} ∂_i ∂_l Ψ  -  δ_{kl} ∂_i ∂_j Ψ
                          -δ_{ij} ∂_k ∂_l Ψ  +  δ_{il} ∂_k ∂_j Ψ ]
              + (a²/c²) [ H' - (2H Ψ' + 2H² Φ + 4H² Ψ)/c² ]
                         × (δ_{li} δ_{kj} - δ_{lk} δ_{ij})
    """
    c2 = c_val * c_val
    a2 = a * a
    psi = phi          # Ψ = Φ
    psi_dot = phi_dot
    psi_ddot = phi_ddot

    # ---- R_{k00l} ----
    # Scalar part of the diagonal
    diag_scalar = (Hprime * (1.0 - 2.0 * psi / c2)
                   + psi_ddot / c2
                   + H * (phi_dot + psi_dot) / c2)
    Rd_k00l = np.empty((3, 3))
    for k in range(3):
        for l in range(3):
            Rd_k00l[k, l] = -hess_phi[k, l]
            if k == l:
                Rd_k00l[k, l] += diag_scalar
    Rd_k00l *= a2

    # ---- R_{0lki} ----
    # For each (l,k,i):
    #   R_{0lki} = (a²/c²) [ δ_{il}(∂_k Ψ' + H ∂_k Φ)
    #                         -δ_{lk}(∂_i Ψ' + H ∂_i Φ) ]
    combo = np.empty(3)
    for i in range(3):
        combo[i] = grad_phi_dot[i] + H * grad_phi[i]  # ∂_i Ψ' + H ∂_i Φ

    Rd_0lki = np.zeros((3, 3, 3))
    fac_0 = a2 / c2
    for l in range(3):
        for k in range(3):
            for i in range(3):
                val = 0.0
                if i == l:
                    val += combo[k]
                if l == k:
                    val -= combo[i]
                Rd_0lki[l, k, i] = fac_0 * val

    # ---- R_{kijl} ----
    # First term:  (a²/c²)[δ_{kj} ∂_i∂_l Ψ - δ_{kl} ∂_i∂_j Ψ
    #                      -δ_{ij} ∂_k∂_l Ψ + δ_{il} ∂_k∂_j Ψ]
    # Second term: (a²/c²)[H' - (2HΨ' + 2H²Φ + 4H²Ψ)/c²]
    #              × (δ_{li}δ_{kj} - δ_{lk}δ_{ij})
    second_scalar = Hprime - (2.0*H*psi_dot + 2.0*H*H*phi + 4.0*H*H*psi) / c2

    Rd_kijl = np.zeros((3, 3, 3, 3))
    fac_k = a2 / c2
    for k_idx in range(3):
        for i_idx in range(3):
            for j_idx in range(3):
                for l_idx in range(3):
                    val = 0.0
                    # First group (Hessian terms)
                    if k_idx == j_idx:
                        val += hess_phi[i_idx, l_idx]
                    if k_idx == l_idx:
                        val -= hess_phi[i_idx, j_idx]
                    if i_idx == j_idx:
                        val -= hess_phi[k_idx, l_idx]
                    if i_idx == l_idx:
                        val += hess_phi[k_idx, j_idx]
                    # Second group (scalar × Kronecker combination)
                    kron = 0.0
                    if l_idx == i_idx and k_idx == j_idx:
                        kron += 1.0
                    if l_idx == k_idx and i_idx == j_idx:
                        kron -= 1.0
                    val += second_scalar * kron
                    Rd_kijl[k_idx, i_idx, j_idx, l_idx] = fac_k * val

    return Rd_k00l, Rd_0lki, Rd_kijl


# ------------------------------------------------------------------
#  High-level helper that takes interpolator outputs directly
# ------------------------------------------------------------------

def compute_riemann_blocks(a, H, Hprime, interp, pos, eta, c_val):
    r"""
    Compute the three covariant Riemann blocks from grid-interpolated fields.

    Parameters
    ----------
    a : float
        Scale factor.
    H : float
        Conformal Hubble parameter  H = a'/a.
    Hprime : float
        dH/dη.
    interp : InterpolatorFast
        Grid interpolator (must support ``value_gradient_hessian_and_time_derivative``).
    pos : array (3,)
        Spatial position [x, y, z].
    eta : float
        Conformal time.
    c_val : float
        Speed of light.

    Returns
    -------
    Rd_k00l, Rd_0lki, Rd_kijl : ndarrays
        All-indices-down Riemann blocks.
    """
    # Main interpolation: value, spatial gradient, spatial Hessian, time derivative
    val, grad3, hess3_tuple, dtd = interp.value_gradient_hessian_and_time_derivative(pos, "Phi", eta)
    gx, gy, gz = grad3
    hxx, hyy, hzz, hxy, hxz, hyz = hess3_tuple
    grad_phi = np.array([gx, gy, gz])

    # Spatial Hessian (3×3 symmetric)
    hess_phi = np.array([
        [hxx, hxy, hxz],
        [hxy, hyy, hyz],
        [hxz, hyz, hzz],
    ])

    # Temporal derivatives
    phi = val
    phi_dot = dtd          # ∂Φ/∂η

    # Mixed temporal-spatial derivatives via finite differences in η
    dt_fd = max(1e-6 * abs(eta), 1.0)

    _, grad3_p, _, dtd_p = interp.value_gradient_hessian_and_time_derivative(pos, "Phi", eta + dt_fd)
    _, grad3_m, _, dtd_m = interp.value_gradient_hessian_and_time_derivative(pos, "Phi", eta - dt_fd)

    gxp, gyp, gzp = grad3_p
    gxm, gym, gzm = grad3_m

    # ∂_i Φ' = ∂²Φ/(∂x^i ∂η)
    grad_phi_dot = np.array([
        (gxp - gxm) / (2.0 * dt_fd),
        (gyp - gym) / (2.0 * dt_fd),
        (gzp - gzm) / (2.0 * dt_fd),
    ])

    # ∂²Φ/∂η²
    phi_ddot = (dtd_p - dtd_m) / (2.0 * dt_fd)

    return riemann_blocks_kernel(
        a, H, Hprime,
        phi, phi_dot, phi_ddot,
        grad_phi, grad_phi_dot, hess_phi,
        c_val,
    )
