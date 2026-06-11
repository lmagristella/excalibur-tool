"""
Curvature for the quasi-spherical Szekeres metric (Ricci focusing + Weyl shear).

Phase-1 (correctness-first) strategy
------------------------------------
The Riemann tensor is built **numerically** from the analytic Christoffel
symbols of :class:`~excalibur.metrics.szekeres_metric.SzekeresMetric`:

    R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
                          + Gamma^rho_{mu lam} Gamma^lam_{nu sigma}
                          - Gamma^rho_{nu lam} Gamma^lam_{mu sigma}

(the partial derivatives are central finite differences of the *analytic*
``christoffel`` field, which is smooth -- this is the same robust pattern used
for the Christoffel-vs-FD validation).  No ``sympy`` dependency and no
hand-transcription of the paper's Appendix; an analytic / codegen backend is a
Phase-2 performance optimisation.

From the Riemann tensor we expose what the optical (Sachs/Jacobi) solver needs:

* ``ricci_focusing`` -- ``R_{alpha beta} k^alpha k^beta`` (Ricci/Weyl-free beam
  convergence).  For dust + Lambda and a *null* ``k`` the cosmological constant
  drops out (``g_{ab} k^a k^b = 0``) and the field equations give the exact
  closed form ``R_{ab} k^a k^b = 8 pi G rho (k^t)^2`` (units cancel; Celerier
  2024).  This is validated against the numerical Riemann in the test suite.

* ``optical_tidal_matrix`` -- the 2x2 ``R_{AB} = R_{mu alpha nu beta}
  k^alpha k^beta e_A^mu e_B^nu`` driving the Jacobi map (eqs. 25-27).
"""
import numpy as np

from excalibur.core.constants import G, c as C_LIGHT


# ----------------------------------------------------------------------
#  Numerical Riemann tensor from the analytic Christoffel field
# ----------------------------------------------------------------------
def riemann_tensor(metric, x, h=None):
    r"""Riemann tensor ``R^rho_{sigma mu nu}`` (upper first index) at ``x``.

    Parameters
    ----------
    metric : SzekeresMetric
        Must expose ``christoffel(x)`` (analytic) and ``metric_tensor(x)``.
    x : ndarray (4,)
        Position ``(t, r, p, q)``.
    h : ndarray (4,), optional
        Per-coordinate finite-difference steps for ``d_mu Gamma``.  Defaults to a
        relative step on ``t, r`` and an absolute step on ``p, q``.

    Returns
    -------
    R : ndarray (4, 4, 4, 4)
        ``R[rho, sigma, mu, nu]`` = ``R^rho_{sigma mu nu}``.
    """
    if h is None:
        h = np.array([1e-6 * max(abs(x[0]), 1.0),
                      1e-6 * max(abs(x[1]), 1.0),
                      1e-6, 1e-6])

    Gam = metric.christoffel(x)                      # Gamma^rho_{sigma tau}

    # d_mu Gamma^rho_{nu sigma}  via central differences
    dGam = np.zeros((4, 4, 4, 4))                    # dGam[mu, rho, a, b]
    for mu in range(4):
        xp = x.copy(); xp[mu] += h[mu]
        xm = x.copy(); xm[mu] -= h[mu]
        dGam[mu] = (metric.christoffel(xp) - metric.christoffel(xm)) / (2 * h[mu])

    # R^rho_{sigma mu nu}
    R = np.zeros((4, 4, 4, 4))
    # gamma-gamma terms via einsum:  G[rho,mu,l] G[l,nu,sigma] - G[rho,nu,l] G[l,mu,sigma]
    GG = np.einsum('rml,lns->rsmn', Gam, Gam) - np.einsum('rnl,lms->rsmn', Gam, Gam)
    # derivative terms:  dGam[mu, rho, nu, sigma] - dGam[nu, rho, mu, sigma]
    dterm = np.einsum('mrns->rsmn', dGam) - np.einsum('nrms->rsmn', dGam)
    R = dterm + GG
    return R


def riemann_tensor_lower(metric, x, h=None):
    """Fully covariant Riemann ``R_{rho sigma mu nu} = g_{rho lam} R^lam_{sigma mu nu}``."""
    R_up = riemann_tensor(metric, x, h)
    g = metric.metric_tensor(x)
    return np.einsum('rl,lsmn->rsmn', g, R_up)


def ricci_tensor(metric, x, h=None):
    r"""Ricci tensor ``R_{sigma nu} = R^mu_{sigma mu nu}`` (contraction of Riemann)."""
    R_up = riemann_tensor(metric, x, h)
    return np.einsum('msmn->sn', R_up)


# ----------------------------------------------------------------------
#  Ricci focusing  R_{alpha beta} k^alpha k^beta
# ----------------------------------------------------------------------
def ricci_focusing(metric, x, k, h=None):
    r"""``R_{alpha beta} k^alpha k^beta`` from the numerical Ricci tensor."""
    Ric = ricci_tensor(metric, x, h)
    return float(k @ (Ric @ k))


def ricci_focusing_analytic(model, x, kt):
    r"""Exact Ricci focusing for dust + Lambda: ``8 pi G rho (k^t)^2``.

    For a *null* ``k`` the ``Lambda g_{ab} k^a k^b`` piece vanishes and the dust
    Einstein equation gives ``R_{ab} k^a k^b = 8 pi G rho (k^t)^2`` -- the factors
    of ``c`` cancel between ``T_{tt}`` and ``(u_a k^a)^2``.  ``rho`` is the rest
    mass density at ``x`` (Celerier 2024 eq. 4).
    """
    t, r, p, q = x[0], x[1], x[2], x[3]
    rho = model.rho(t, r, p, q)
    return 8.0 * np.pi * G * rho * kt * kt


# ----------------------------------------------------------------------
#  Optical tidal matrix  R_{AB}
# ----------------------------------------------------------------------
# ======================================================================
#  Analytic curvature  (Celerier 2024, Appendix A)
# ======================================================================
#
#  The paper gives the Ricci and Weyl components in the H, F notation
#  (eq. 74):  H^2 = g_rr = (Phi_,r - Phi E_,r/E)^2/(eps-k),  F^2 = g_pp = Phi^2/E^2,
#  in *geometric* units (g_tt = -1, i.e. a "geometric time" t~ = c t).  We
#  therefore (i) compute H, F and their cosmo-time derivatives from the model,
#  (ii) convert every time derivative to geometric time (divide by c per t
#  index), (iii) evaluate the Appendix expressions, and (iv) contract with the
#  *geometric* photon vector k_geom = (c k^t, k^r, k^p, k^q).  Scalars built this
#  way are unit-independent and match the numerical Riemann.
#
#  Only the second derivatives H_,tt, H_,tp, H_,tq, H_,pp, H_,qq, H_,pq and
#  F_,tt, F_,tr, F_,tp, F_,tq, F_,rr, F_,rp, F_,rq, F_,pp, F_,qq appear -- in
#  particular *no* H_,rr / H_,tr -- so the only high-order areal-radius
#  derivatives needed are Phi_,tt and Phi_,ttr, which have closed forms (the LTB
#  acceleration equation), avoiding any third-order spline differentiation:
#
#      Phi_,tt  = -G M/Phi^2 + (Lambda c^2/3) Phi
#      Phi_,ttr = -G M_,r/Phi^2 + 2 G M Phi_,r/Phi^3 + (Lambda c^2/3) Phi_,r
#
def hf_bundle(model, x):
    r"""Return ``H``, ``F`` and the geometric-time derivatives needed by Appendix A.

    Keys: ``H, Ht, Hr, Hp, Hq, Htt, Htp, Htq, Hpp, Hqq, Hpq`` and the analogous
    ``F...`` (``Ftt, Ftr, Ftp, Ftq, Frr, Frp, Frq, Fpp, Fqq``).  Time derivatives
    are already converted to geometric time (``t~ = c t``).
    """
    t, r, p, q = x[0], x[1], x[2], x[3]
    cc = C_LIGHT
    eps = model.epsilon

    Phi = model.Phi(t, r)
    Phi_t = model.Phi_t(t, r)
    Phi_r = model.Phi_r(t, r)
    Phi_tr = model.Phi_tr(t, r)
    Phi_rr = model.Phi_rr(t, r)
    M = model.M(r); Mr = model.dM(r); Lam = model.Lambda
    # closed-form higher areal-radius derivatives (LTB acceleration eq.)
    Phi_tt = -G * M / Phi ** 2 + (Lam * cc * cc / 3.0) * Phi
    Phi_ttr = (-G * Mr / Phi ** 2 + 2.0 * G * M * Phi_r / Phi ** 3
               + (Lam * cc * cc / 3.0) * Phi_r)

    E = model.E(r, p, q)
    E_p = model.E_p(r, p, q); E_q = model.E_q(r, p, q)
    E_r = model.E_r(r, p, q); E_rr = model.E_rr(r, p, q)
    E_rp = model.E_rp(r, p, q); E_rq = model.E_rq(r, p, q)
    S = model.S(r); Sr = model.dS(r)
    E_pp = 1.0 / S; E_qq = 1.0 / S          # E_,pq = 0
    E_rpp = -Sr / (S * S); E_rqq = -Sr / (S * S)   # E_,rpq = 0

    E2 = E * E; E3 = E2 * E
    ER = E_r / E
    ER_r = E_rr / E - ER * ER
    ER_p = E_rp / E - E_r * E_p / E2
    ER_q = E_rq / E - E_r * E_q / E2
    ER_pp = E_rpp / E - E_r * E_pp / E2 - 2.0 * E_p * E_rp / E2 + 2.0 * E_r * E_p * E_p / E3
    ER_qq = E_rqq / E - E_r * E_qq / E2 - 2.0 * E_q * E_rq / E2 + 2.0 * E_r * E_q * E_q / E3
    ER_pq = (E_rp * E_q - E_rq * E_p) / E2 - 2.0 * E_q * (E_rp * E - E_r * E_p) / E3

    # --- H = D / sqrt(eps - k),  D = Phi_,r - Phi E_,r/E ---
    k = model.k(r); kr = model.dk(r)
    sk = np.sqrt(eps - k)
    D = Phi_r - Phi * ER
    D_t = Phi_tr - Phi_t * ER
    D_tt = Phi_ttr - Phi_tt * ER
    D_r = Phi_rr - Phi_r * ER - Phi * ER_r
    D_p = -Phi * ER_p
    D_q = -Phi * ER_q
    D_tp = -Phi_t * ER_p
    D_tq = -Phi_t * ER_q
    D_pp = -Phi * ER_pp
    D_qq = -Phi * ER_qq
    D_pq = -Phi * ER_pq

    H = D / sk
    Ht = D_t / sk
    Hr = D_r / sk + D * kr / (2.0 * sk ** 3)
    Hp = D_p / sk
    Hq = D_q / sk
    Htt = D_tt / sk
    Htp = D_tp / sk
    Htq = D_tq / sk
    Hpp = D_pp / sk
    Hqq = D_qq / sk
    Hpq = D_pq / sk

    # --- F = Phi / E ---
    F = Phi / E
    Ft = Phi_t / E
    Fr = Phi_r / E - Phi * E_r / E2
    Fp = -Phi * E_p / E2
    Fq = -Phi * E_q / E2
    Ftt = Phi_tt / E
    Ftr = Phi_tr / E - Phi_t * E_r / E2
    Ftp = -Phi_t * E_p / E2
    Ftq = -Phi_t * E_q / E2
    Frr = Phi_rr / E - 2.0 * Phi_r * E_r / E2 - Phi * E_rr / E2 + 2.0 * Phi * E_r * E_r / E3
    Frp = -Phi_r * E_p / E2 - Phi * E_rp / E2 + 2.0 * Phi * E_r * E_p / E3
    Frq = -Phi_r * E_q / E2 - Phi * E_rq / E2 + 2.0 * Phi * E_r * E_q / E3
    Fpp = -Phi * E_pp / E2 + 2.0 * Phi * E_p * E_p / E3
    Fqq = -Phi * E_qq / E2 + 2.0 * Phi * E_q * E_q / E3

    # --- convert time derivatives to geometric time (t~ = c t) ---
    Ht /= cc; Htt /= cc * cc; Htp /= cc; Htq /= cc
    Ft /= cc; Ftt /= cc * cc; Ftr /= cc; Ftp /= cc; Ftq /= cc

    return dict(H=H, Ht=Ht, Hr=Hr, Hp=Hp, Hq=Hq,
                Htt=Htt, Htp=Htp, Htq=Htq, Hpp=Hpp, Hqq=Hqq, Hpq=Hpq,
                F=F, Ft=Ft, Fr=Fr, Fp=Fp, Fq=Fq,
                Ftt=Ftt, Ftr=Ftr, Ftp=Ftp, Ftq=Ftq,
                Frr=Frr, Frp=Frp, Frq=Frq, Fpp=Fpp, Fqq=Fqq)


def ricci_tensor_analytic(model, x):
    r"""Ricci tensor (geometric basis ``(t~, r, p, q)``), Celerier 2024 Appendix A.

    Expressions generated symbolically by
    :mod:`excalibur.metrics._codegen.szekeres_curvature_codegen` (sympy), so they
    are exact -- no hand transcription.  Returns the symmetric 4x4 ``R_{ab}``;
    contract with the geometric photon vector ``(c k^t, k^r, k^p, k^q)``.
    """
    b = hf_bundle(model, x)
    H = b["H"]; Ht = b["Ht"]; Hr = b["Hr"]; Hp = b["Hp"]; Hq = b["Hq"]
    Htt = b["Htt"]; Htp = b["Htp"]; Htq = b["Htq"]
    Hpp = b["Hpp"]; Hqq = b["Hqq"]; Hpq = b["Hpq"]
    F = b["F"]; Ft = b["Ft"]; Fr = b["Fr"]; Fp = b["Fp"]; Fq = b["Fq"]
    Ftt = b["Ftt"]; Ftr = b["Ftr"]; Ftp = b["Ftp"]; Ftq = b["Ftq"]
    Frr = b["Frr"]; Frp = b["Frp"]; Frq = b["Frq"]; Fpp = b["Fpp"]; Fqq = b["Fqq"]

    R = np.zeros((4, 4))
    R[0, 0] = -Htt / H - 2 * Ftt / F
    R[0, 1] = R[1, 0] = 2 * (Fr * Ht - Ftr * H) / (F * H)
    R[0, 2] = R[2, 0] = -Htp / H + Ft * Hp / (F * H) - Ftp / F + Fp * Ft / F**2
    R[0, 3] = R[3, 0] = -Htq / H + Ft * Hq / (F * H) - Ftq / F + Fq * Ft / F**2
    R[1, 1] = (F**2 * H**2 * Htt + 2 * F * Fr * Hr
               + 2 * F * H * (-Frr + Ft * H * Ht) - H**2 * (Hpp + Hqq)) / (F**2 * H)
    R[1, 2] = R[2, 1] = Fr * Hp / (F * H) - Frp / F + Fp * Fr / F**2
    R[1, 3] = R[3, 1] = Fr * Hq / (F * H) - Frq / F + Fq * Fr / F**2
    R[2, 2] = (F * Fr * Hr / H**3 - F * Frr / H**2 + F * Ft * Ht / H + F * Ftt
               - Fr**2 / H**2 + Ft**2 - Hpp / H + Fp * Hp / (F * H) - Fpp / F
               - Fq * Hq / (F * H) - Fqq / F + Fp**2 / F**2 + Fq**2 / F**2)
    R[2, 3] = R[3, 2] = (-F * Hpq + Fp * Hq + Fq * Hp) / (F * H)
    R[3, 3] = (F * Fr * Hr / H**3 - F * Frr / H**2 + F * Ft * Ht / H + F * Ftt
               - Fr**2 / H**2 + Ft**2 - Hqq / H - Fp * Hp / (F * H) - Fpp / F
               + Fq * Hq / (F * H) - Fqq / F + Fp**2 / F**2 + Fq**2 / F**2)
    return R


# backward-compatible alias
ricci_tensor_appendix = ricci_tensor_analytic


def _k_geometric(k):
    """Photon vector in the geometric basis ``(t~, r, p, q)``: ``k^t~ = c k^t``."""
    return np.array([C_LIGHT * k[0], k[1], k[2], k[3]])


def tidal_tensor_analytic(model, x, k):
    r"""Screen tidal tensor ``T_{mu nu} = R_{mu alpha nu beta} k^alpha k^beta``.

    Exact symbolic expressions (Celerier 2024 Appendix A, regenerated by the
    codegen).  ``T`` is returned in the geometric basis; the optical tidal matrix
    is ``R_{AB} = T_{mu nu} e_A^mu e_B^nu`` with the Sachs vectors also expressed
    in the geometric basis.  Drives ``d^2 D/dlambda^2 = -R D`` (eqs. 25-27).
    """
    b = hf_bundle(model, x)
    H = b["H"]; Ht = b["Ht"]; Hr = b["Hr"]; Hp = b["Hp"]; Hq = b["Hq"]
    Htt = b["Htt"]; Htp = b["Htp"]; Htq = b["Htq"]
    Hpp = b["Hpp"]; Hqq = b["Hqq"]; Hpq = b["Hpq"]
    F = b["F"]; Ft = b["Ft"]; Fr = b["Fr"]; Fp = b["Fp"]; Fq = b["Fq"]
    Ftt = b["Ftt"]; Ftr = b["Ftr"]; Ftp = b["Ftp"]; Ftq = b["Ftq"]
    Frr = b["Frr"]; Frp = b["Frp"]; Frq = b["Frq"]; Fpp = b["Fpp"]; Fqq = b["Fqq"]
    kt, kr, kp, kq = _k_geometric(k)

    T = np.zeros((4, 4))
    T[0, 0] = -F * Ftt * kp**2 - F * Ftt * kq**2 - H * Htt * kr**2
    T[0, 1] = T[1, 0] = (F**2 * (kp**2 + kq**2) * (Fr * Ht - Ftr * H)
                         + F * H**2 * Htt * kr * kt
                         + H**2 * kr * (kp * (F * Htp - Ft * Hp)
                                        + kq * (F * Htq - Ft * Hq))) / (F * H)
    T[0, 2] = T[2, 0] = (-F**2 * kp * kr * (Fr * Ht - Ftr * H)
                         + F * H * (F * Ftt * kp * kt + kp * kq * (F * Ftq - Fq * Ft)
                                    + kq**2 * (-F * Ftp + Fp * Ft))
                         + H**2 * kr**2 * (-F * Htp + Ft * Hp)) / (F * H)
    T[0, 3] = T[3, 0] = (-F**2 * kq * kr * (Fr * Ht - Ftr * H)
                         + F * H * (F * Ftt * kq * kt + kp**2 * (-F * Ftq + Fq * Ft)
                                    + kp * kq * (F * Ftp - Fp * Ft))
                         + H**2 * kr**2 * (-F * Htq + Ft * Hq)) / (F * H)
    T[1, 1] = (-F * H**2 * Htt * kt**2
               + 2 * H**2 * (kp * kq * (-F * Hpq + Fp * Hq + Fq * Hp)
                             - kp * kt * (F * Htp - Ft * Hp)
                             - kq * kt * (F * Htq - Ft * Hq))
               + kp**2 * (F**2 * Fr * Hr - F * H * (F * Frr - F * Ft * H * Ht + H * Hpp)
                          + H**2 * (Fp * Hp - Fq * Hq))
               + kq**2 * (F**2 * Fr * Hr - F * H * (F * Frr - F * Ft * H * Ht + H * Hqq)
                          + H**2 * (-Fp * Hp + Fq * Hq))) / (F * H)
    T[1, 2] = T[2, 1] = (F * (-F * kp * kt * (Fr * Ht - Ftr * H)
                              - kp * kq * (F * Fr * Hq - H * (F * Frq - Fq * Fr))
                              + kq**2 * (F * Fr * Hp - H * (F * Frp - Fp * Fr)))
                         - H**2 * kr * (kq * (-F * Hpq + Fp * Hq + Fq * Hp)
                                        - kt * (F * Htp - Ft * Hp))
                         - kp * kr * (F**2 * Fr * Hr
                                      - F * H * (F * Frr - F * Ft * H * Ht + H * Hpp)
                                      + H**2 * (Fp * Hp - Fq * Hq))) / (F * H)
    T[1, 3] = T[3, 1] = (F * (-F * kq * kt * (Fr * Ht - Ftr * H)
                              + kp**2 * (F * Fr * Hq - H * (F * Frq - Fq * Fr))
                              - kp * kq * (F * Fr * Hp - H * (F * Frp - Fp * Fr)))
                         - H**2 * kr * (kp * (-F * Hpq + Fp * Hq + Fq * Hp)
                                        - kt * (F * Htq - Ft * Hq))
                         - kq * kr * (F**2 * Fr * Hr
                                      - F * H * (F * Frr - F * Ft * H * Ht + H * Hqq)
                                      + H**2 * (-Fp * Hp + Fq * Hq))) / (F * H)
    T[2, 2] = (-F * H**2 * kt * (F * Ftt * kt + 2 * kq * (F * Ftq - Fq * Ft))
               - 2 * F * H * kr * (F * kt * (-Fr * Ht + Ftr * H)
                                   - kq * (F * Fr * Hq + H * (-F * Frq + Fq * Fr)))
               - F * kq**2 * (F**2 * Fr**2 + H**2 * (-F**2 * Ft**2 + F * Fpp + F * Fqq
                                                     - Fp**2 - Fq**2))
               + H * kr**2 * (F**2 * Fr * Hr
                              - F * H * (F * Frr - F * Ft * H * Ht + H * Hpp)
                              + H**2 * (Fp * Hp - Fq * Hq))) / (F * H**2)
    T[2, 3] = T[3, 2] = (F * H**2 * kt * (kp * (F * Ftq - Fq * Ft) + kq * (F * Ftp - Fp * Ft))
                         - F * H * kr * (kp * (F * Fr * Hq - H * (F * Frq - Fq * Fr))
                                         + kq * (F * Fr * Hp - H * (F * Frp - Fp * Fr)))
                         + F * kp * kq * (F**2 * Fr**2
                                          + H**2 * (-F**2 * Ft**2 + F * Fpp + F * Fqq
                                                    - Fp**2 - Fq**2))
                         + H**3 * kr**2 * (-F * Hpq + Fp * Hq + Fq * Hp)) / (F * H**2)
    T[3, 3] = (-F * H**2 * kt * (F * Ftt * kt + 2 * kp * (F * Ftp - Fp * Ft))
               - 2 * F * H * kr * (F * kt * (-Fr * Ht + Ftr * H)
                                   - kp * (F * Fr * Hp + H * (-F * Frp + Fp * Fr)))
               - F * kp**2 * (F**2 * Fr**2 + H**2 * (-F**2 * Ft**2 + F * Fpp + F * Fqq
                                                     - Fp**2 - Fq**2))
               + H * kr**2 * (F**2 * Fr * Hr
                              - F * H * (F * Frr - F * Ft * H * Ht + H * Hqq)
                              + H**2 * (-Fp * Hp + Fq * Hq))) / (F * H**2)
    return T


def ricci_focusing_appendix(model, x, k):
    r"""``R_{ab} k^a k^b`` from the analytic (symbolic) Ricci tensor."""
    R = ricci_tensor_analytic(model, x)
    kg = _k_geometric(k)
    return float(kg @ (R @ kg))


def optical_tidal_matrix(metric, x, k, e1, e2, h=None):
    r"""2x2 optical tidal matrix ``R_{AB} = R_{mu alpha nu beta} k^alpha k^beta e_A^mu e_B^nu``.

    Drives the Jacobi map evolution ``d^2 D/dlambda^2 = -R D`` (Celerier 2024
    eqs. 25-27, via the geodesic-deviation/Sachs formalism).  ``e1, e2`` are the
    Sachs screen 4-vectors orthonormal in the screen plane.
    """
    R_low = riemann_tensor_lower(metric, x, h)       # R_{mu alpha nu beta}
    # Contract k on indices alpha (1) and beta (3): T_{mu nu} = R_{mu a n b} k^a k^b
    T = np.einsum('manb,a,b->mn', R_low, k, k)
    e = (e1, e2)
    RAB = np.empty((2, 2))
    for A in range(2):
        for B in range(2):
            RAB[A, B] = e[A] @ (T @ e[B])
    return RAB
