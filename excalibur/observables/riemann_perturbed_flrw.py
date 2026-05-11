# excalibur/observables/riemann_perturbed_flrw.py
r"""
Minimal Riemann tensor blocks for the perturbed FLRW metric.

Metric (conformal Newtonian gauge, psi = phi):
    ds^2 = a^2(eta) [ -(1 + 2psi/c^2) c^2 deta^2  +  (1 - 2phi/c^2) delta_ij dx^i dx^j ]

Throughout this module:
    - Greek indices mu,nu,...  in  {0,1,2,3} where 0 = conformal time eta.
    - Latin indices i,j,k,l  in  {1,2,3} (spatial).
    - H = a'/a  is the conformal Hubble parameter (prime = d/deta).
    - H' = dH/deta.
    - phi' = dphi/deta,  phi'' = d^2phi/deta^2.
    - d_i phi  = spatial gradient component.
    - d_i d_j phi = spatial Hessian component.
    - d_i phi' = mixed derivative  d^2phi/(dx^i deta).

The three independent blocks of the Riemann tensor (with *all indices
down*, R_{alpha beta gamma delta}) needed for the optical tidal matrix are:

    R_{k00l}, R_{0lki}, R_{kijl}

All formulas assume psi = phi (no anisotropic stress).

Convention: Returned arrays use *spatial* indices 0,1,2 <-> x,y,z.
All blocks are COVARIANT (all indices down)  --  no index raising needed
before the contraction  R_{AB} = R_{alpha beta mu nu} k^alpha k^beta e_A^mu e_B^nu.
Considering that the Sachs basis vectors are spatial only (e_A^0 = 0), we only need the spatial components of 
the Riemann tensor. R_{AB} becomes 
R_
"""

import numpy as np
from numba import njit

# ------------------------------------------------------------------
#  Core kernel  --  returns the three blocks as flat tuples
# ------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def riemann_blocks_kernel(
    a,              # scale factor
    H,              # conformal Hubble  H = a'/a  [s^{-1}]
    Hprime,         # dH/deta                       [s^{-2}]
    phi,            # Phi  (SI: m^2/s^2)
    phi_dot,        # dPhi/deta  (SI: m^2/s^3)
    phi_ddot,       # d^2Phi/deta^2 (SI: m^2/s^4)
    grad_phi,       # (d_x Phi, d_y Phi, d_z Phi)  each m/s^2  (length-3 array)
    grad_phi_dot,   # (d_x Phi', d_y Phi', d_z Phi')  each m/s^3 (length-3 array)
    hess_phi,       # spatial Hessian d_i d_j Phi  (3x3 symmetric, m/s^2 /m = 1/s^2)
    c_val,          # speed of light
):
    r"""
    Compute the three covariant Riemann blocks for perturbed FLRW (Psi = Phi).

    Returns
    -------
    Rd_k00l : ndarray (3, 3)
        R_{k00l}  (all indices down) with spatial indices k,l  in  {0,1,2}.
    Rd_0lki : ndarray (3, 3, 3)
        R_{0lki}  (all indices down) with spatial indices l,k,i  in  {0,1,2}.
    Rd_kijl : ndarray (3, 3, 3, 3)
        R_{kijl}  (all indices down) with spatial indices k,i,j,l  in  {0,1,2}.

    Notes
    -----
    The formulas (Psi = Phi, all indices down):

    R_{k00l} = a^2 [ -d_k d_l Psi
                      + delta_{kl} (H'(1 - 2Psi/c^2) + Psi''/c^2 + H(Phi'+Psi')/c^2) ]

    R_{0lki} = (a^2/c^2) [ delta_{il}(d_k Psi' + H d_k Phi)
                          -delta_{lk}(d_i Psi' + H d_i Phi) ]

    R_{kijl} = (a^2/c^2) [ delta_{kj} d_i d_l Psi  -  delta_{kl} d_i d_j Psi
                          -delta_{ij} d_k d_l Psi  +  delta_{il} d_k d_j Psi ]
              + (a^2/c^2) [ H' - (2H Psi' + 2H^2 Phi + 4H^2 Psi)/c^2 ]
                         x (delta_{li} delta_{kj} - delta_{lk} delta_{ij})
    """
    c2 = c_val * c_val
    a2 = a * a
    psi = phi          # Psi = Phi
    psi_dot = phi_dot
    psi_ddot = phi_ddot

    # ---- R_{k00l} ----
    # Linearized lapse derivation at fixed a:
    #   g_{00} = -a^2 (1 + 2 Psi/c^2) c^2  ->  Gamma_{k00} = -1/2 d_k g_{00} = a^2 d_k Psi,
    # hence the Hessian piece is  R_{k00l}^{(H)} = -d_l Gamma_{k00} = -a^2 d_k d_l Psi.
    # The remaining diagonal term below is the homogeneous time-dependent scalar piece.
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
    #   R_{0lki} = (a^2/c^2) [ delta_{il}(d_k Psi' + H d_k Phi)
    #                         -delta_{lk}(d_i Psi' + H d_i Phi) ]
    combo = np.empty(3)
    for i in range(3):
        combo[i] = grad_phi_dot[i] + H * grad_phi[i]  # d_i Psi' + H d_i Phi

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
    # Linearized spatial-metric derivation at fixed a:
    #   Gamma_{kij} = (a^2/c^2)(-delta_{kj} d_i Psi - delta_{ki} d_j Psi + delta_{ij} d_k Psi)
    # so d_j Gamma_{kil} - d_l Gamma_{kij} gives the four Hessian terms below;
    # the two pieces proportional to delta_{ki} cancel because d_j d_l Psi = d_l d_j Psi.
    # First term:  (a^2/c^2)[delta_{kj} d_id_l Psi - delta_{kl} d_id_j Psi
    #                      -delta_{ij} d_kd_l Psi + delta_{il} d_kd_j Psi]
    # Second term: (a^2/c^2)[H' - (2HPsi' + 2H^2Phi + 4H^2Psi)/c^2]
    #              x (delta_{li}delta_{kj} - delta_{lk}delta_{ij})
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
                    # Second group (scalar x Kronecker combination)
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
        dH/deta.
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

    # Spatial Hessian (3x3 symmetric)
    hess_phi = np.array([
        [hxx, hxy, hxz],
        [hxy, hyy, hyz],
        [hxz, hyz, hzz],
    ])

    # Temporal derivatives
    phi = val
    phi_dot = dtd          # dPhi/deta

    # Mixed temporal-spatial derivatives via finite differences in eta
    dt_fd = max(1e-6 * abs(eta), 1.0)

    _, grad3_p, _, dtd_p = interp.value_gradient_hessian_and_time_derivative(pos, "Phi", eta + dt_fd)
    _, grad3_m, _, dtd_m = interp.value_gradient_hessian_and_time_derivative(pos, "Phi", eta - dt_fd)

    gxp, gyp, gzp = grad3_p
    gxm, gym, gzm = grad3_m

    # d_i Phi' = d^2Phi/(dx^i deta)
    grad_phi_dot = np.array([
        (gxp - gxm) / (2.0 * dt_fd),
        (gyp - gym) / (2.0 * dt_fd),
        (gzp - gzm) / (2.0 * dt_fd),
    ])

    # d^2Phi/deta^2
    phi_ddot = (dtd_p - dtd_m) / (2.0 * dt_fd)

    return riemann_blocks_kernel(
        a, H, Hprime,
        phi, phi_dot, phi_ddot,
        grad_phi, grad_phi_dot, hess_phi,
        c_val,
    )
