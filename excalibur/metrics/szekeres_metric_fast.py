r"""
Numba (Phase-2) backend for the quasi-spherical Szekeres metric.

The readable :class:`~excalibur.metrics.szekeres_metric.SzekeresMetric` calls
SciPy splines and assembles tensors in Python -- fine for correctness, slow in a
tight RK loop.  Here the model is **frozen into arrays** once (background
``Phi`` tables on the ``(t, r)`` grid + the free functions on the ``r`` grid) and
the entire per-step hot path runs under ``@njit``:

* ``_interp2d`` / ``_interp1d`` -- Catmull-Rom cubic interpolation (matches the
  SciPy spline to ~1e-6, so Phase 2 == Phase 1);
* ``_geodesic_rhs`` -- the null geodesic RHS in the explicit Celerier (2024)
  eqs. 7-10 form (validated equal to the Christoffel assembly at 2e-16);
* ``_rk4_geodesic`` -- a compiled fixed-step RK4 driver.

Build the frozen arrays with :func:`build_fast_szekeres` and integrate with
:func:`integrate_geodesic_fast`.  Run in cosmo units (``EXCALIBUR_UNITS=cosmo``).

Array layout
------------
``phi5[5, nt, nr]`` : ``Phi, Phi_,t, Phi_,r, Phi_,tr, Phi_,rr``.
``rfun[13, nr]``    : ``M, dM, k, dk, S, dS, ddS, P, dP, ddP, Q, dQ, ddQ``.
"""
import numpy as np
from numba import njit

from excalibur.core.constants import c as _c, G as _G
from excalibur.observables.sachs_basis import sachs_transport_rhs
from excalibur.observables.optical_tidal_matrix import jacobi_rhs
from excalibur.integration.integrator_numba_schemes import (
    _clamp_signed_dt, _error_norm, _proposed_dt)


# ----------------------------------------------------------------------
#  Catmull-Rom cubic interpolation
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _crw(f):
    f2 = f * f
    f3 = f2 * f
    return (-0.5 * f3 + f2 - 0.5 * f,
            1.5 * f3 - 2.5 * f2 + 1.0,
            -1.5 * f3 + 2.0 * f2 + 0.5 * f,
            0.5 * f3 - 0.5 * f2)


@njit(cache=True, fastmath=True)
def _interp2d(t, r, t0, dt, r0, dr, nt, nr, tab):
    ft = (t - t0) / dt
    fr = (r - r0) / dr
    it = int(np.floor(ft))
    ir = int(np.floor(fr))
    if it < 1:
        it = 1
    elif it > nt - 3:
        it = nt - 3
    if ir < 1:
        ir = 1
    elif ir > nr - 3:
        ir = nr - 3
    a = _crw(ft - it)
    b = _crw(fr - ir)
    val = 0.0
    for i in range(4):
        row = 0.0
        for j in range(4):
            row += b[j] * tab[it - 1 + i, ir - 1 + j]
        val += a[i] * row
    return val


@njit(cache=True, fastmath=True)
def _interp1d(r, r0, dr, nr, arr):
    fr = (r - r0) / dr
    ir = int(np.floor(fr))
    if ir < 1:
        ir = 1
    elif ir > nr - 3:
        ir = nr - 3
    b = _crw(fr - ir)
    return (b[0] * arr[ir - 1] + b[1] * arr[ir] + b[2] * arr[ir + 1]
            + b[3] * arr[ir + 2])


# ----------------------------------------------------------------------
#  Null geodesic RHS  (Celerier 2024 eqs. 7-10, cosmo units)
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _geodesic_rhs(state, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval):
    t = state[0]; r = state[1]; p = state[2]; q = state[3]
    ut = state[4]; ur = state[5]; up = state[6]; uq = state[7]

    Phi = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[0])
    Phi_t = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[1])
    Phi_r = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[2])
    Phi_tr = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[3])
    Phi_rr = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[4])

    k = _interp1d(r, r0, dr, nr, rfun[2])
    k_r = _interp1d(r, r0, dr, nr, rfun[3])
    S = _interp1d(r, r0, dr, nr, rfun[4])
    Sr = _interp1d(r, r0, dr, nr, rfun[5])
    P = _interp1d(r, r0, dr, nr, rfun[7])
    Pr = _interp1d(r, r0, dr, nr, rfun[8])
    Prr = _interp1d(r, r0, dr, nr, rfun[9])
    Q = _interp1d(r, r0, dr, nr, rfun[10])
    Qr = _interp1d(r, r0, dr, nr, rfun[11])
    Qrr = _interp1d(r, r0, dr, nr, rfun[12])

    dp_ = p - P
    dq_ = q - Q
    u_ = dp_ / S
    v_ = dq_ / S
    E = (dp_ * dp_ + dq_ * dq_) / (2.0 * S) + eps * S / 2.0
    E_p = dp_ / S
    E_q = dq_ / S
    E_r = -(Sr / 2.0) * (u_ * u_ + v_ * v_ - eps) - (dp_ * Pr + dq_ * Qr) / S
    E_rp = -Pr / S - dp_ * Sr / (S * S)
    E_rq = -Qr / S - dq_ * Sr / (S * S)
    # E_rr (closed form, mirrors SzekeresModel.E_rr)
    Srr = _interp1d(r, r0, dr, nr, rfun[6])
    B = dp_ * dp_ / (S * S) + dq_ * dq_ / (S * S) - eps
    Br = (-2.0 * (dp_ * Pr + dq_ * Qr) / (S * S)
          - 2.0 * (dp_ * dp_ + dq_ * dq_) * Sr / (S ** 3))
    C_r = ((-Pr * Pr + dp_ * Prr - Qr * Qr + dq_ * Qrr) / S
           - (dp_ * Pr + dq_ * Qr) * Sr / (S * S))
    E_rr = -(Srr / 2.0) * B - (Sr / 2.0) * Br - C_r

    ER = E_r / E
    D = Phi_r - Phi * ER
    ek = eps - k
    E2 = E * E

    # eq 7 (d^2 t/ds^2), 1/c^2 from g^{tt}
    A_ = ((Phi_tr - Phi_t * ER) / ek) * D
    Bb = Phi * Phi_t / E2
    du_t = -(1.0 / (cval * cval)) * (A_ * ur * ur + Bb * (up * up + uq * uq))
    # eq 8
    c1 = (Phi_tr - Phi_t * ER) / D
    c2 = (Phi_rr - Phi * E_rr / E) / D - ER + k_r / (2.0 * ek)
    c3 = (Phi / E2) * (E_r * E_p - E * E_rp) / D
    c4 = (Phi / E2) * (E_r * E_q - E * E_rq) / D
    c5 = (Phi / E2) * ek / D
    du_r = -(2.0 * c1 * ut * ur + c2 * ur * ur + 2.0 * c3 * ur * up
             + 2.0 * c4 * ur * uq - c5 * (up * up + uq * uq))
    # eq 9
    d1 = Phi_t / Phi
    d3 = D / Phi
    dp2 = (D / (Phi * ek)) * (E_r * E_p - E * E_rp)
    du_p = -(2.0 * d1 * ut * up - dp2 * ur * ur + 2.0 * d3 * ur * up
             - 2.0 * (E_q / E) * up * uq + (E_p / E) * (-up * up + uq * uq))
    # eq 10
    dq2 = (D / (Phi * ek)) * (E_r * E_q - E * E_rq)
    du_q = -(2.0 * d1 * ut * uq - dq2 * ur * ur + 2.0 * d3 * ur * uq
             - 2.0 * (E_p / E) * up * uq + (E_q / E) * (up * up - uq * uq))

    out = np.empty(8)
    out[0] = ut; out[1] = ur; out[2] = up; out[3] = uq
    out[4] = du_t; out[5] = du_r; out[6] = du_p; out[7] = du_q
    return out


@njit(cache=True, fastmath=True)
def _solve_null_kt(t, r, p, q, kr, kp, kq, t0, dt, r0, dr, nt, nr,
                   phi5, rfun, eps, cval, future):
    """Past-pointing ``k^t`` (``< 0``) making ``k`` null at the point."""
    Phi = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[0])
    Phi_r = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[2])
    k = _interp1d(r, r0, dr, nr, rfun[2])
    S = _interp1d(r, r0, dr, nr, rfun[4])
    P = _interp1d(r, r0, dr, nr, rfun[7])
    Q = _interp1d(r, r0, dr, nr, rfun[10])
    Sr = _interp1d(r, r0, dr, nr, rfun[5])
    Pr = _interp1d(r, r0, dr, nr, rfun[8])
    Qr = _interp1d(r, r0, dr, nr, rfun[11])
    dp_ = p - P; dq_ = q - Q
    E = (dp_ * dp_ + dq_ * dq_) / (2.0 * S) + eps * S / 2.0
    E_r = -(Sr / 2.0) * (dp_ * dp_ / (S * S) + dq_ * dq_ / (S * S) - eps) \
        - (dp_ * Pr + dq_ * Qr) / S
    ER = E_r / E
    H2 = (Phi_r - Phi * ER) ** 2 / (eps - k)
    F2 = Phi * Phi / (E * E)
    spatial = H2 * kr * kr + F2 * (kp * kp + kq * kq)
    kt = np.sqrt(spatial) / cval
    return kt if future else -kt


@njit(cache=True, fastmath=True)
def _rk4_geodesic(state0, ds, n_steps, t_min_stop, r_min_stop, r_max_stop,
                  t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval):
    """Integrate the 8-component geodesic; record [t, r, p, q, k^t] per step."""
    state = state0.copy()
    out = np.empty((n_steps, 5))
    n = 0
    for _ in range(n_steps):
        k1 = _geodesic_rhs(state, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval)
        k2 = _geodesic_rhs(state + 0.5 * ds * k1, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval)
        k3 = _geodesic_rhs(state + 0.5 * ds * k2, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval)
        k4 = _geodesic_rhs(state + ds * k3, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval)
        state = state + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if state[0] < t_min_stop or state[1] < r_min_stop or state[1] > r_max_stop:
            break
        out[n, 0] = state[0]; out[n, 1] = state[1]; out[n, 2] = state[2]
        out[n, 3] = state[3]; out[n, 4] = state[4]
        n += 1
    return out[:n]


@njit(cache=True, fastmath=True)
def _dopri5_geodesic(state0, ds_init, n_steps, ds_min, ds_max, rtol, atol,
                     max_rejected, t_min_stop, r_min_stop, r_max_stop,
                     t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval):
    """Adaptive DOPRI5(4) twin of :func:`_rk4_geodesic`.

    Same recorded columns ``[t, r, p, q, k^t]`` and domain-boundary stop, but
    the affine step ``ds`` is grown/shrunk from the embedded 4th-order error
    estimate (Dormand--Prince tableau, identical coefficients to
    :func:`~excalibur.integration.integrator_numba_schemes.dopri54_step`).
    ``n_steps`` is the *budget* of accepted steps; the returned trajectory has
    one row per accepted step.
    """
    state = state0.copy()
    out = np.empty((n_steps, 5))
    n = 0
    ds = ds_init
    rejected = 0
    attempts = 0
    max_attempts = n_steps * 20 + 64
    while n < n_steps and attempts < max_attempts:
        ds_eff = _clamp_signed_dt(ds, ds_min, ds_max)
        k1 = _geodesic_rhs(state, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval)
        k2 = _geodesic_rhs(state + ds_eff * (0.2 * k1),
                           t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval)
        k3 = _geodesic_rhs(state + ds_eff * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2),
                           t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval)
        k4 = _geodesic_rhs(state + ds_eff * ((44.0 / 45.0) * k1 - (56.0 / 15.0) * k2
                                             + (32.0 / 9.0) * k3),
                           t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval)
        k5 = _geodesic_rhs(state + ds_eff * ((19372.0 / 6561.0) * k1
                                             - (25360.0 / 2187.0) * k2
                                             + (64448.0 / 6561.0) * k3
                                             - (212.0 / 729.0) * k4),
                           t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval)
        k6 = _geodesic_rhs(state + ds_eff * ((9017.0 / 3168.0) * k1
                                             - (355.0 / 33.0) * k2
                                             + (46732.0 / 5247.0) * k3
                                             + (49.0 / 176.0) * k4
                                             - (5103.0 / 18656.0) * k5),
                           t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval)
        y5 = state + ds_eff * ((35.0 / 384.0) * k1 + (500.0 / 1113.0) * k3
                               + (125.0 / 192.0) * k4 - (2187.0 / 6784.0) * k5
                               + (11.0 / 84.0) * k6)
        k7 = _geodesic_rhs(y5, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval)
        y4 = state + ds_eff * ((5179.0 / 57600.0) * k1 + (7571.0 / 16695.0) * k3
                               + (393.0 / 640.0) * k4 - (92097.0 / 339200.0) * k5
                               + (187.0 / 2100.0) * k6 + (1.0 / 40.0) * k7)

        err_norm = _error_norm(y5 - y4, state, y5, rtol, atol)
        accepted = np.isfinite(err_norm) and err_norm <= 1.0
        ds_new = _proposed_dt(ds_eff, err_norm, accepted)

        if accepted:
            state = y5
            if state[0] < t_min_stop or state[1] < r_min_stop or state[1] > r_max_stop:
                break
            out[n, 0] = state[0]; out[n, 1] = state[1]; out[n, 2] = state[2]
            out[n, 3] = state[3]; out[n, 4] = state[4]
            n += 1
            rejected = 0
        else:
            rejected += 1
            if rejected > max_rejected:
                break

        if not np.isfinite(ds_new) or abs(ds_new) < 1e-30:
            break
        ds = ds_new
        attempts += 1
    return out[:n]


# ======================================================================
#  24-component path: geodesic + Sachs screen + Jacobi map (distances)
# ======================================================================
@njit(cache=True, fastmath=True)
def _christoffel(g_diag, dg):
    """Gamma^a_{bc} for a diagonal metric (cosmo basis)."""
    G3 = np.zeros((4, 4, 4))
    for a in range(4):
        inv = 0.5 / g_diag[a]
        for b in range(4):
            for cc in range(4):
                term = 0.0
                if a == cc:
                    term += dg[b, a]
                if a == b:
                    term += dg[cc, a]
                if b == cc:
                    term -= dg[a, b]
                if term != 0.0:
                    G3[a, b, cc] = inv * term
    return G3


@njit(cache=True, fastmath=True)
def _full_rhs(state, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam):
    """RHS of the 24-comp state [x(4), k(4), e1(4), e2(4), D(4), P(4)]."""
    t = state[0]; r = state[1]; p = state[2]; q = state[3]
    k = state[4:8]
    e1 = state[8:12]
    e2 = state[12:16]
    Dm = state[16:20]
    Pm = state[20:24]

    Phi = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[0])
    Phi_t = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[1])
    Phi_r = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[2])
    Phi_tr = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[3])
    Phi_rr = _interp2d(t, r, t0, dt, r0, dr, nt, nr, phi5[4])

    M = _interp1d(r, r0, dr, nr, rfun[0]); Mr = _interp1d(r, r0, dr, nr, rfun[1])
    kk = _interp1d(r, r0, dr, nr, rfun[2]); kr = _interp1d(r, r0, dr, nr, rfun[3])
    S = _interp1d(r, r0, dr, nr, rfun[4]); Sr = _interp1d(r, r0, dr, nr, rfun[5])
    Srr = _interp1d(r, r0, dr, nr, rfun[6])
    P = _interp1d(r, r0, dr, nr, rfun[7]); Pr = _interp1d(r, r0, dr, nr, rfun[8])
    Prr = _interp1d(r, r0, dr, nr, rfun[9])
    Q = _interp1d(r, r0, dr, nr, rfun[10]); Qr = _interp1d(r, r0, dr, nr, rfun[11])
    Qrr = _interp1d(r, r0, dr, nr, rfun[12])

    dp_ = p - P; dq_ = q - Q
    E = (dp_ * dp_ + dq_ * dq_) / (2.0 * S) + eps * S / 2.0
    E_p = dp_ / S; E_q = dq_ / S
    E_r = -(Sr / 2.0) * (dp_ * dp_ / (S * S) + dq_ * dq_ / (S * S) - eps) \
        - (dp_ * Pr + dq_ * Qr) / S
    E_rp = -Pr / S - dp_ * Sr / (S * S)
    E_rq = -Qr / S - dq_ * Sr / (S * S)
    B = dp_ * dp_ / (S * S) + dq_ * dq_ / (S * S) - eps
    Br = (-2.0 * (dp_ * Pr + dq_ * Qr) / (S * S)
          - 2.0 * (dp_ * dp_ + dq_ * dq_) * Sr / (S ** 3))
    C_r = ((-Pr * Pr + dp_ * Prr - Qr * Qr + dq_ * Qrr) / S
           - (dp_ * Pr + dq_ * Qr) * Sr / (S * S))
    E_rr = -(Srr / 2.0) * B - (Sr / 2.0) * Br - C_r
    E_pp = 1.0 / S; E_qq = 1.0 / S
    E_rpp = -Sr / (S * S); E_rqq = -Sr / (S * S)

    ER = E_r / E
    Dfac = Phi_r - Phi * ER
    ek = eps - kk
    E2 = E * E; E3 = E2 * E

    # ---- cosmo metric diagonal + gradient (for Christoffel) ----
    dER_dr = E_rr / E - ER * ER
    dER_dp = E_rp / E - ER * (E_p / E)
    dER_dq = E_rq / E - ER * (E_q / E)
    Dt = Phi_tr - Phi_t * ER
    Dr = Phi_rr - Phi_r * ER - Phi * dER_dr
    Dp = -Phi * dER_dp
    Dq = -Phi * dER_dq
    grr = Dfac * Dfac / ek
    dgrr_dt = 2.0 * Dfac * Dt / ek
    dgrr_dr = 2.0 * Dfac * Dr / ek + Dfac * Dfac * kr / (ek * ek)
    dgrr_dp = 2.0 * Dfac * Dp / ek
    dgrr_dq = 2.0 * Dfac * Dq / ek
    gpp = Phi * Phi / E2
    dgpp_dt = 2.0 * Phi * Phi_t / E2
    dgpp_dr = 2.0 * Phi * Phi_r / E2 - 2.0 * Phi * Phi * E_r / E3
    dgpp_dp = -2.0 * Phi * Phi * E_p / E3
    dgpp_dq = -2.0 * Phi * Phi * E_q / E3
    g_diag = np.array([-cval * cval, grr, gpp, gpp])
    dg = np.zeros((4, 4))
    dg[0, 1] = dgrr_dt; dg[1, 1] = dgrr_dr; dg[2, 1] = dgrr_dp; dg[3, 1] = dgrr_dq
    dg[0, 2] = dgpp_dt; dg[1, 2] = dgpp_dr; dg[2, 2] = dgpp_dp; dg[3, 2] = dgpp_dq
    dg[0, 3] = dgpp_dt; dg[1, 3] = dgpp_dr; dg[2, 3] = dgpp_dp; dg[3, 3] = dgpp_dq
    Gam = _christoffel(g_diag, dg)

    # geodesic acceleration
    du = np.zeros(4)
    for a in range(4):
        s = 0.0
        for b in range(4):
            for cc in range(4):
                s += Gam[a, b, cc] * k[b] * k[cc]
        du[a] = -s

    # ---- Sachs transport ----
    de1 = sachs_transport_rhs(e1, Gam, k)
    de2 = sachs_transport_rhs(e2, Gam, k)

    # ---- geometric H,F derivatives (for the tidal tensor) ----
    Phi_tt = -_G * M / (Phi * Phi) + (Lam * cval * cval / 3.0) * Phi
    Phi_ttr = (-_G * Mr / (Phi * Phi) + 2.0 * _G * M * Phi_r / (Phi ** 3)
               + (Lam * cval * cval / 3.0) * Phi_r)
    sk = np.sqrt(ek)
    ER_pp = E_rpp / E - E_r * E_pp / E2 - 2.0 * E_p * E_rp / E2 + 2.0 * E_r * E_p * E_p / E3
    ER_qq = E_rqq / E - E_r * E_qq / E2 - 2.0 * E_q * E_rq / E2 + 2.0 * E_r * E_q * E_q / E3
    ER_pq = (E_rp * E_q - E_rq * E_p) / E2 - 2.0 * E_q * (E_rp * E - E_r * E_p) / E3
    ER_r = E_rr / E - ER * ER
    Dtt = Phi_ttr - Phi_tt * ER
    Dtp = -Phi_t * dER_dp
    Dtq = -Phi_t * dER_dq
    Dpp = -Phi * ER_pp; Dqq = -Phi * ER_qq; Dpq = -Phi * ER_pq
    H = Dfac / sk
    Ht = Dt / sk; Hr = Dr / sk + Dfac * kr / (2.0 * sk ** 3)
    Hp = Dp / sk; Hq = Dq / sk
    Htt = Dtt / sk; Htp = Dtp / sk; Htq = Dtq / sk
    Hpp = Dpp / sk; Hqq = Dqq / sk; Hpq = Dpq / sk
    F = Phi / E
    Ft = Phi_t / E; Fr = Phi_r / E - Phi * E_r / E2
    Fp = -Phi * E_p / E2; Fq = -Phi * E_q / E2
    Ftt = Phi_tt / E; Ftr = Phi_tr / E - Phi_t * E_r / E2
    Ftp = -Phi_t * E_p / E2; Ftq = -Phi_t * E_q / E2
    Frr = Phi_rr / E - 2.0 * Phi_r * E_r / E2 - Phi * E_rr / E2 + 2.0 * Phi * E_r * E_r / E3
    Frp = -Phi_r * E_p / E2 - Phi * E_rp / E2 + 2.0 * Phi * E_r * E_p / E3
    Frq = -Phi_r * E_q / E2 - Phi * E_rq / E2 + 2.0 * Phi * E_r * E_q / E3
    Fpp = -Phi * E_pp / E2 + 2.0 * Phi * E_p * E_p / E3
    Fqq = -Phi * E_qq / E2 + 2.0 * Phi * E_q * E_q / E3
    # geometric time (t~ = c t): each t-derivative /= c
    Ht /= cval; Htt /= cval * cval; Htp /= cval; Htq /= cval
    Ft /= cval; Ftt /= cval * cval; Ftr /= cval; Ftp /= cval; Ftq /= cval

    # geometric photon vector
    kt = cval * k[0]; krr = k[1]; kp = k[2]; kq = k[3]

    # tidal tensor T_{mu nu} = R_{mu a nu b} k^a k^b  (codegen, geometric basis)
    T = np.zeros((4, 4))
    T[0, 0] = -F * Ftt * kp**2 - F * Ftt * kq**2 - H * Htt * krr**2
    T[0, 1] = (F**2 * (kp**2 + kq**2) * (Fr * Ht - Ftr * H)
               + F * H**2 * Htt * krr * kt
               + H**2 * krr * (kp * (F * Htp - Ft * Hp) + kq * (F * Htq - Ft * Hq))) / (F * H)
    T[0, 2] = (-F**2 * kp * krr * (Fr * Ht - Ftr * H)
               + F * H * (F * Ftt * kp * kt + kp * kq * (F * Ftq - Fq * Ft) + kq**2 * (-F * Ftp + Fp * Ft))
               + H**2 * krr**2 * (-F * Htp + Ft * Hp)) / (F * H)
    T[0, 3] = (-F**2 * kq * krr * (Fr * Ht - Ftr * H)
               + F * H * (F * Ftt * kq * kt + kp**2 * (-F * Ftq + Fq * Ft) + kp * kq * (F * Ftp - Fp * Ft))
               + H**2 * krr**2 * (-F * Htq + Ft * Hq)) / (F * H)
    T[1, 1] = (-F * H**2 * Htt * kt**2
               + 2.0 * H**2 * (kp * kq * (-F * Hpq + Fp * Hq + Fq * Hp)
                               - kp * kt * (F * Htp - Ft * Hp) - kq * kt * (F * Htq - Ft * Hq))
               + kp**2 * (F**2 * Fr * Hr - F * H * (F * Frr - F * Ft * H * Ht + H * Hpp) + H**2 * (Fp * Hp - Fq * Hq))
               + kq**2 * (F**2 * Fr * Hr - F * H * (F * Frr - F * Ft * H * Ht + H * Hqq) + H**2 * (-Fp * Hp + Fq * Hq))) / (F * H)
    T[1, 2] = (F * (-F * kp * kt * (Fr * Ht - Ftr * H)
                    - kp * kq * (F * Fr * Hq - H * (F * Frq - Fq * Fr))
                    + kq**2 * (F * Fr * Hp - H * (F * Frp - Fp * Fr)))
               - H**2 * krr * (kq * (-F * Hpq + Fp * Hq + Fq * Hp) - kt * (F * Htp - Ft * Hp))
               - kp * krr * (F**2 * Fr * Hr - F * H * (F * Frr - F * Ft * H * Ht + H * Hpp) + H**2 * (Fp * Hp - Fq * Hq))) / (F * H)
    T[1, 3] = (F * (-F * kq * kt * (Fr * Ht - Ftr * H)
                    + kp**2 * (F * Fr * Hq - H * (F * Frq - Fq * Fr))
                    - kp * kq * (F * Fr * Hp - H * (F * Frp - Fp * Fr)))
               - H**2 * krr * (kp * (-F * Hpq + Fp * Hq + Fq * Hp) - kt * (F * Htq - Ft * Hq))
               - kq * krr * (F**2 * Fr * Hr - F * H * (F * Frr - F * Ft * H * Ht + H * Hqq) + H**2 * (-Fp * Hp + Fq * Hq))) / (F * H)
    T[2, 2] = (-F * H**2 * kt * (F * Ftt * kt + 2.0 * kq * (F * Ftq - Fq * Ft))
               - 2.0 * F * H * krr * (F * kt * (-Fr * Ht + Ftr * H) - kq * (F * Fr * Hq + H * (-F * Frq + Fq * Fr)))
               - F * kq**2 * (F**2 * Fr**2 + H**2 * (-F**2 * Ft**2 + F * Fpp + F * Fqq - Fp**2 - Fq**2))
               + H * krr**2 * (F**2 * Fr * Hr - F * H * (F * Frr - F * Ft * H * Ht + H * Hpp) + H**2 * (Fp * Hp - Fq * Hq))) / (F * H**2)
    T[2, 3] = (F * H**2 * kt * (kp * (F * Ftq - Fq * Ft) + kq * (F * Ftp - Fp * Ft))
               - F * H * krr * (kp * (F * Fr * Hq - H * (F * Frq - Fq * Fr)) + kq * (F * Fr * Hp - H * (F * Frp - Fp * Fr)))
               + F * kp * kq * (F**2 * Fr**2 + H**2 * (-F**2 * Ft**2 + F * Fpp + F * Fqq - Fp**2 - Fq**2))
               + H**3 * krr**2 * (-F * Hpq + Fp * Hq + Fq * Hp)) / (F * H**2)
    T[3, 3] = (-F * H**2 * kt * (F * Ftt * kt + 2.0 * kp * (F * Ftp - Fp * Ft))
               - 2.0 * F * H * krr * (F * kt * (-Fr * Ht + Ftr * H) - kp * (F * Fr * Hp + H * (-F * Frp + Fp * Fr)))
               - F * kp**2 * (F**2 * Fr**2 + H**2 * (-F**2 * Ft**2 + F * Fpp + F * Fqq - Fp**2 - Fq**2))
               + H * krr**2 * (F**2 * Fr * Hr - F * H * (F * Frr - F * Ft * H * Ht + H * Hqq) + H**2 * (-Fp * Hp + Fq * Hq))) / (F * H**2)
    T[1, 0] = T[0, 1]; T[2, 0] = T[0, 2]; T[3, 0] = T[0, 3]
    T[2, 1] = T[1, 2]; T[3, 1] = T[1, 3]; T[3, 2] = T[2, 3]

    # screen vectors in geometric basis (e^t~ = c e^t)
    e1g = np.array([cval * e1[0], e1[1], e1[2], e1[3]])
    e2g = np.array([cval * e2[0], e2[1], e2[2], e2[3]])
    R_AB = np.zeros((2, 2))
    Te1 = T @ e1g; Te2 = T @ e2g
    R_AB[0, 0] = e1g @ Te1; R_AB[0, 1] = e1g @ Te2
    R_AB[1, 0] = e2g @ Te1; R_AB[1, 1] = e2g @ Te2

    dj = jacobi_rhs(np.concatenate((Dm, Pm)), R_AB)

    out = np.empty(24)
    out[0:4] = k
    out[4:8] = du
    out[8:12] = de1
    out[12:16] = de2
    out[16:20] = dj[0:4]
    out[20:24] = dj[4:8]
    return out


@njit(cache=True, fastmath=True)
def _rk4_distance(state0, ds, n_steps, t_min_stop, r_min_stop, r_max_stop,
                  t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam, kt_o):
    """Integrate the 24-comp state; record [t, r, z, det D] per step."""
    state = state0.copy()
    out = np.empty((n_steps, 4))
    n = 0
    for _ in range(n_steps):
        k1 = _full_rhs(state, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam)
        k2 = _full_rhs(state + 0.5 * ds * k1, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam)
        k3 = _full_rhs(state + 0.5 * ds * k2, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam)
        k4 = _full_rhs(state + ds * k3, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam)
        state = state + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if state[0] < t_min_stop or state[1] < r_min_stop or state[1] > r_max_stop:
            break
        D = state[16:20]
        out[n, 0] = state[0]; out[n, 1] = state[1]
        out[n, 2] = state[4] / kt_o - 1.0
        out[n, 3] = D[0] * D[3] - D[1] * D[2]
        n += 1
    return out[:n]


@njit(cache=True, fastmath=True)
def _dopri5_distance(state0, ds_init, n_steps, ds_min, ds_max, rtol, atol,
                     max_rejected, t_min_stop, r_min_stop, r_max_stop,
                     t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam, kt_o):
    """Adaptive DOPRI5(4) twin of :func:`_rk4_distance` (24-comp distance state)."""
    state = state0.copy()
    out = np.empty((n_steps, 4))
    n = 0
    ds = ds_init
    rejected = 0
    attempts = 0
    max_attempts = n_steps * 20 + 64
    while n < n_steps and attempts < max_attempts:
        ds_eff = _clamp_signed_dt(ds, ds_min, ds_max)
        k1 = _full_rhs(state, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam)
        k2 = _full_rhs(state + ds_eff * (0.2 * k1),
                       t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam)
        k3 = _full_rhs(state + ds_eff * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2),
                       t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam)
        k4 = _full_rhs(state + ds_eff * ((44.0 / 45.0) * k1 - (56.0 / 15.0) * k2
                                         + (32.0 / 9.0) * k3),
                       t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam)
        k5 = _full_rhs(state + ds_eff * ((19372.0 / 6561.0) * k1
                                         - (25360.0 / 2187.0) * k2
                                         + (64448.0 / 6561.0) * k3
                                         - (212.0 / 729.0) * k4),
                       t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam)
        k6 = _full_rhs(state + ds_eff * ((9017.0 / 3168.0) * k1
                                         - (355.0 / 33.0) * k2
                                         + (46732.0 / 5247.0) * k3
                                         + (49.0 / 176.0) * k4
                                         - (5103.0 / 18656.0) * k5),
                       t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam)
        y5 = state + ds_eff * ((35.0 / 384.0) * k1 + (500.0 / 1113.0) * k3
                               + (125.0 / 192.0) * k4 - (2187.0 / 6784.0) * k5
                               + (11.0 / 84.0) * k6)
        k7 = _full_rhs(y5, t0, dt, r0, dr, nt, nr, phi5, rfun, eps, cval, Lam)
        y4 = state + ds_eff * ((5179.0 / 57600.0) * k1 + (7571.0 / 16695.0) * k3
                               + (393.0 / 640.0) * k4 - (92097.0 / 339200.0) * k5
                               + (187.0 / 2100.0) * k6 + (1.0 / 40.0) * k7)

        err_norm = _error_norm(y5 - y4, state, y5, rtol, atol)
        accepted = np.isfinite(err_norm) and err_norm <= 1.0
        ds_new = _proposed_dt(ds_eff, err_norm, accepted)

        if accepted:
            state = y5
            if state[0] < t_min_stop or state[1] < r_min_stop or state[1] > r_max_stop:
                break
            D = state[16:20]
            out[n, 0] = state[0]; out[n, 1] = state[1]
            out[n, 2] = state[4] / kt_o - 1.0
            out[n, 3] = D[0] * D[3] - D[1] * D[2]
            n += 1
            rejected = 0
        else:
            rejected += 1
            if rejected > max_rejected:
                break

        if not np.isfinite(ds_new) or abs(ds_new) < 1e-30:
            break
        ds = ds_new
        attempts += 1
    return out[:n]


# ----------------------------------------------------------------------
#  Python-side builder + driver
# ----------------------------------------------------------------------
class FastSzekeres:
    """Frozen (array-only) Szekeres model for the Numba geodesic backend."""

    def __init__(self, model):
        m = model
        tg = m.t_grid
        rg = m.r_grid
        self.t0 = float(tg[0]); self.dt = float(tg[1] - tg[0]); self.nt = len(tg)
        self.r0 = float(rg[0]); self.dr = float(rg[1] - rg[0]); self.nr = len(rg)
        self.eps = float(m.epsilon)
        self.cval = float(_c)
        self.Lam = float(m.Lambda)

        # Free functions on the r-grid (cheap loop).
        rfun = np.empty((13, self.nr))
        for ir, r in enumerate(rg):
            rfun[0, ir] = m.M(r); rfun[1, ir] = m.dM(r)
            rfun[2, ir] = m.k(r); rfun[3, ir] = m.dk(r)
            rfun[4, ir] = m.S(r); rfun[5, ir] = m.dS(r); rfun[6, ir] = m.ddS(r)
            rfun[7, ir] = m.P(r); rfun[8, ir] = m.dP(r); rfun[9, ir] = m.ddP(r)
            rfun[10, ir] = m.Q(r); rfun[11, ir] = m.dQ(r); rfun[12, ir] = m.ddQ(r)
        self.rfun = rfun

        # Background tables -- vectorised grid evaluation of the splines.
        phi5 = np.empty((5, self.nt, self.nr))
        phi5[0] = m._phi_spline(tg, rg)               # Phi
        phi5[2] = m._phi_r_spline(tg, rg)             # Phi_,r (analytic-based)
        phi5[3] = m._phi_r_spline(tg, rg, dx=1)       # Phi_,tr
        phi5[4] = m._phi_r_spline(tg, rg, dy=1)       # Phi_,rr
        # Phi_,t = W(Phi) straight from Friedmann (exact), broadcast over r.
        M_row = rfun[0][None, :]
        k_row = rfun[2][None, :]
        Phi = phi5[0]
        phi5[1] = np.sqrt(2.0 * _G * M_row / Phi - k_row * _c * _c
                          + (m.Lambda * _c * _c / 3.0) * Phi * Phi)
        self.phi5 = phi5

    def solve_null_kt(self, x, k_spatial, future=False):
        return float(_solve_null_kt(x[0], x[1], x[2], x[3],
                                    k_spatial[0], k_spatial[1], k_spatial[2],
                                    self.t0, self.dt, self.r0, self.dr,
                                    self.nt, self.nr, self.phi5, self.rfun,
                                    self.eps, self.cval, future))

    def _HF(self, x):
        """H, F at position x (from the frozen tables)."""
        t, r, p, q = x
        Phi = _interp2d(t, r, self.t0, self.dt, self.r0, self.dr, self.nt, self.nr, self.phi5[0])
        Phi_r = _interp2d(t, r, self.t0, self.dt, self.r0, self.dr, self.nt, self.nr, self.phi5[2])
        k = _interp1d(r, self.r0, self.dr, self.nr, self.rfun[2])
        S = _interp1d(r, self.r0, self.dr, self.nr, self.rfun[4])
        Sr = _interp1d(r, self.r0, self.dr, self.nr, self.rfun[5])
        P = _interp1d(r, self.r0, self.dr, self.nr, self.rfun[7])
        Pr = _interp1d(r, self.r0, self.dr, self.nr, self.rfun[8])
        Q = _interp1d(r, self.r0, self.dr, self.nr, self.rfun[10])
        Qr = _interp1d(r, self.r0, self.dr, self.nr, self.rfun[11])
        dp_ = p - P; dq_ = q - Q
        E = (dp_ * dp_ + dq_ * dq_) / (2.0 * S) + self.eps * S / 2.0
        E_r = -(Sr / 2.0) * (dp_ * dp_ / (S * S) + dq_ * dq_ / (S * S) - self.eps) \
            - (dp_ * Pr + dq_ * Qr) / S
        H = (Phi_r - Phi * E_r / E) / np.sqrt(self.eps - k)
        F = Phi / E
        return H, F

    def init_screen(self, x, k):
        """Sachs screen e1, e2 at x (Celerier eqs. 85-86, p<->q-corrected)."""
        H, F = self._HF(x)
        kr, kp, kq = k[1], k[2], k[3]
        N2 = kp * kp + kq * kq
        if N2 <= 1e-30 * max(1.0, kr * kr):
            e1 = np.array([0.0, 0.0, 1.0 / F, 0.0])
            e2 = np.array([0.0, 0.0, 0.0, 1.0 / F])
        else:
            N = np.sqrt(N2)
            e1 = np.array([0.0, (F / H) * N, -(H / F) * kr * kp / N, -(H / F) * kr * kq / N])
            e2 = np.array([0.0, 0.0, kq / (F * N), -kp / (F * N)])
        # normalise in the spatial metric (e^t = 0): g_rr = H^2, g_pp = g_qq = F^2
        n1 = np.sqrt(H * H * e1[1] ** 2 + F * F * (e1[2] ** 2 + e1[3] ** 2))
        n2 = np.sqrt(H * H * e2[1] ** 2 + F * F * (e2[2] ** 2 + e2[3] ** 2))
        return e1 / n1, e2 / n2


def _adaptive_ds_bounds(ds_init, ds_min, ds_max):
    """Default affine-step bounds for the DOPRI5 drivers.

    ``ds_max`` is capped at ``20 x`` the RK4-equivalent step so the adaptive
    controller cannot stride over a compact lens / void feature in smooth
    regions (cf. the ``--dt-max-rs`` lesson in the NFW backend); ``ds_min``
    floors it at ``1e-3 x`` to keep step refusals near shell crossings bounded.
    """
    ds_min = ds_init / 1000.0 if ds_min is None else float(ds_min)
    ds_max = ds_init * 20.0 if ds_max is None else float(ds_max)
    return ds_min, ds_max


def integrate_geodesic_fast(fast, x0, k_spatial, *, n_steps=6000, span_t=10.0,
                            t_min_stop=None, r_min_stop=None, r_max_stop=None,
                            scheme="rk4", rtol=1e-8, atol=1e-10,
                            ds_min=None, ds_max=None, max_rejected=50):
    r"""Backward-integrate a null geodesic with the Numba driver.

    ``scheme`` selects the time stepper: ``"rk4"`` (fixed step, exactly
    ``n_steps`` steps) or ``"dopri5"`` (adaptive Dormand--Prince 5(4); ``n_steps``
    is then the *budget* of accepted steps and ``span_t`` only sizes the initial
    step).  ``rtol/atol`` and ``ds_min/ds_max`` tune the adaptive controller.

    Returns ``dict`` with arrays ``t, r, p, q, z`` (``z`` from ``k^t/k^t_o``).
    """
    x0 = np.asarray(x0, dtype=float)
    kt = fast.solve_null_kt(x0, k_spatial)
    state0 = np.array([x0[0], x0[1], x0[2], x0[3], kt,
                       k_spatial[0], k_spatial[1], k_spatial[2]])
    ds = abs(span_t / (kt * n_steps))
    t_min_stop = fast.t0 + 2 * fast.dt if t_min_stop is None else t_min_stop
    r_min_stop = fast.r0 + 2 * fast.dr if r_min_stop is None else r_min_stop
    r_max_stop = (fast.r0 + (fast.nr - 3) * fast.dr) if r_max_stop is None else r_max_stop
    if scheme == "rk4":
        rec = _rk4_geodesic(state0, ds, n_steps, t_min_stop, r_min_stop, r_max_stop,
                            fast.t0, fast.dt, fast.r0, fast.dr, fast.nt, fast.nr,
                            fast.phi5, fast.rfun, fast.eps, fast.cval)
    elif scheme == "dopri5":
        ds_lo, ds_hi = _adaptive_ds_bounds(ds, ds_min, ds_max)
        rec = _dopri5_geodesic(state0, ds, n_steps, ds_lo, ds_hi, rtol, atol,
                               max_rejected, t_min_stop, r_min_stop, r_max_stop,
                               fast.t0, fast.dt, fast.r0, fast.dr, fast.nt, fast.nr,
                               fast.phi5, fast.rfun, fast.eps, fast.cval)
    else:
        raise ValueError(f"Unknown scheme {scheme!r}; use 'rk4' or 'dopri5'.")
    return {"t": rec[:, 0], "r": rec[:, 1], "p": rec[:, 2], "q": rec[:, 3],
            "z": rec[:, 4] / kt - 1.0, "kt_o": kt}


def integrate_distance_fast(fast, x0, k_spatial, *, n_steps=6000, span_t=10.0,
                            t_min_stop=None, r_min_stop=None, r_max_stop=None,
                            scheme="rk4", rtol=1e-8, atol=1e-10,
                            ds_min=None, ds_max=None, max_rejected=50):
    r"""Backward-integrate the 24-comp distance state (Numba).

    ``scheme`` selects ``"rk4"`` (fixed step) or ``"dopri5"`` (adaptive
    Dormand--Prince 5(4); see :func:`integrate_geodesic_fast`).

    Returns ``dict`` with ``t, r, z, D_A, D_L, gamma`` (``D_A = c k^t_o sqrt|det D|``).
    """
    x0 = np.asarray(x0, dtype=float)
    kt = fast.solve_null_kt(x0, k_spatial)
    e1, e2 = fast.init_screen(x0, np.array([kt, *k_spatial]))
    state0 = np.empty(24)
    state0[0:4] = x0
    state0[4:8] = [kt, k_spatial[0], k_spatial[1], k_spatial[2]]
    state0[8:12] = e1
    state0[12:16] = e2
    state0[16:20] = 0.0
    state0[20:24] = [1.0, 0.0, 0.0, 1.0]
    ds = abs(span_t / (kt * n_steps))
    t_min_stop = fast.t0 + 2 * fast.dt if t_min_stop is None else t_min_stop
    r_min_stop = fast.r0 + 2 * fast.dr if r_min_stop is None else r_min_stop
    r_max_stop = (fast.r0 + (fast.nr - 3) * fast.dr) if r_max_stop is None else r_max_stop
    if scheme == "rk4":
        rec = _rk4_distance(state0, ds, n_steps, t_min_stop, r_min_stop, r_max_stop,
                            fast.t0, fast.dt, fast.r0, fast.dr, fast.nt, fast.nr,
                            fast.phi5, fast.rfun, fast.eps, fast.cval, fast.Lam, kt)
    elif scheme == "dopri5":
        ds_lo, ds_hi = _adaptive_ds_bounds(ds, ds_min, ds_max)
        rec = _dopri5_distance(state0, ds, n_steps, ds_lo, ds_hi, rtol, atol,
                               max_rejected, t_min_stop, r_min_stop, r_max_stop,
                               fast.t0, fast.dt, fast.r0, fast.dr, fast.nt, fast.nr,
                               fast.phi5, fast.rfun, fast.eps, fast.cval, fast.Lam, kt)
    else:
        raise ValueError(f"Unknown scheme {scheme!r}; use 'rk4' or 'dopri5'.")
    z = rec[:, 2]
    det_D = rec[:, 3]
    D_A = fast.cval * abs(kt) * np.sqrt(np.abs(det_D))
    return {"t": rec[:, 0], "r": rec[:, 1], "z": z, "D_A": D_A,
            "D_L": (1.0 + z) ** 2 * D_A, "kt_o": kt}
