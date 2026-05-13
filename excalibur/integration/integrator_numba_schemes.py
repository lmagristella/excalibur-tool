"""Generic numba integrator schemes.

This module isolates the algebra of time stepping from the physics-specific
right-hand side used by ``integrator_numba.py``. The compiled scheme kernels
accept a compiled RHS callback plus its arguments and therefore stay agnostic
to the actual geodesic / lensing model.
"""

from __future__ import annotations

import numpy as np
from numba import njit


SCHEME_RK4 = 1
SCHEME_RKF45 = 2
SCHEME_DOPRI54 = 3

STOP_REASON_UNKNOWN = 0
STOP_REASON_LAMBDA_STOP = 1
STOP_REASON_STEP_BUDGET = 2
STOP_REASON_MAX_REJECTED = 3
STOP_REASON_DT_INVALID = 4
STOP_REASON_MAX_ATTEMPTS = 5
STOP_REASON_PYTHON_BACKEND = 6

_STOP_REASON_LABELS = (
    "unknown",
    "lambda_stop_reached",
    "accepted_step_budget_exhausted",
    "max_rejected_exceeded",
    "dt_invalid_or_too_small",
    "max_attempts_exhausted",
    "python_backend_no_reason",
)

_SCHEME_NAME_TO_ID = {
    "rk4": SCHEME_RK4,
    "rk45": SCHEME_RKF45,
    "rkf45": SCHEME_RKF45,
    "dopri5": SCHEME_DOPRI54,
    "dopri54": SCHEME_DOPRI54,
    "dopri5(4)": SCHEME_DOPRI54,
}

_ADAPTIVE_SCHEME_IDS = {SCHEME_RKF45, SCHEME_DOPRI54}


def normalize_numba_integrator_name(name: str) -> str:
    key = name.lower().strip()
    if key not in _SCHEME_NAME_TO_ID:
        valid = ", ".join(sorted(_SCHEME_NAME_TO_ID))
        raise ValueError(f"Unsupported numba integrator '{name}'. Valid choices: {valid}")
    if key in ("rkf45",):
        return "rk45"
    if key in ("dopri54", "dopri5(4)"):
        return "dopri5"
    return key


def numba_integrator_scheme_id(name: str) -> int:
    return _SCHEME_NAME_TO_ID[normalize_numba_integrator_name(name)]


def numba_integrator_is_adaptive(name: str) -> bool:
    return numba_integrator_scheme_id(name) in _ADAPTIVE_SCHEME_IDS


def integrator_stop_reason_labels() -> tuple[str, ...]:
    return _STOP_REASON_LABELS


@njit(cache=True, fastmath=True)
def _error_norm(err, state, candidate, rtol, atol):
    total = 0.0
    n = state.shape[0]
    for i in range(n):
        scale = atol + rtol * max(abs(state[i]), abs(candidate[i]))
        q = err[i] / scale
        total += q * q
    return np.sqrt(total / n)


@njit(cache=True, fastmath=True)
def _proposed_dt(dt, err_norm, accepted):
    safety = 0.9
    min_factor = 0.2
    max_factor = 5.0

    if not np.isfinite(err_norm):
        factor = min_factor
    elif accepted:
        if err_norm < 1e-30:
            factor = max_factor
        else:
            factor = safety * err_norm ** (-0.2)
            factor = min(max_factor, max(min_factor, factor))
    else:
        factor = safety * err_norm ** (-0.25)
        factor = max(min_factor, min(max_factor, factor))

    dt_new = abs(dt) * factor
    return dt_new if dt >= 0.0 else -dt_new


@njit(cache=True, fastmath=True)
def _clamp_signed_dt(dt, dt_min, dt_max):
    abs_dt = abs(dt)
    if abs_dt < dt_min:
        abs_dt = dt_min
    if abs_dt > dt_max:
        abs_dt = dt_max
    return abs_dt if dt >= 0.0 else -abs_dt


@njit(cache=True, fastmath=True)
def rk4_step(rhs_func, state, dt, *rhs_args):
    k1 = rhs_func(state, *rhs_args)
    k2 = rhs_func(state + 0.5 * dt * k1, *rhs_args)
    k3 = rhs_func(state + 0.5 * dt * k2, *rhs_args)
    k4 = rhs_func(state + dt * k3, *rhs_args)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=True, fastmath=True)
def rkf45_step(rhs_func, state, dt, rtol, atol, *rhs_args):
    k1 = rhs_func(state, *rhs_args)
    k2 = rhs_func(state + dt * (0.25 * k1), *rhs_args)
    k3 = rhs_func(state + dt * ((3.0 / 32.0) * k1 + (9.0 / 32.0) * k2), *rhs_args)
    k4 = rhs_func(
        state + dt * ((1932.0 / 2197.0) * k1 - (7200.0 / 2197.0) * k2 + (7296.0 / 2197.0) * k3),
        *rhs_args,
    )
    k5 = rhs_func(
        state + dt * ((439.0 / 216.0) * k1 - 8.0 * k2 + (3680.0 / 513.0) * k3 - (845.0 / 4104.0) * k4),
        *rhs_args,
    )
    k6 = rhs_func(
        state
        + dt
        * (
            -(8.0 / 27.0) * k1
            + 2.0 * k2
            - (3544.0 / 2565.0) * k3
            + (1859.0 / 4104.0) * k4
            - (11.0 / 40.0) * k5
        ),
        *rhs_args,
    )

    y4 = state + dt * (
        (25.0 / 216.0) * k1
        + (1408.0 / 2565.0) * k3
        + (2197.0 / 4104.0) * k4
        - (1.0 / 5.0) * k5
    )
    y5 = state + dt * (
        (16.0 / 135.0) * k1
        + (6656.0 / 12825.0) * k3
        + (28561.0 / 56430.0) * k4
        - (9.0 / 50.0) * k5
        + (2.0 / 55.0) * k6
    )

    err = y5 - y4
    err_norm = _error_norm(err, state, y5, rtol, atol)
    accepted = np.isfinite(err_norm) and err_norm <= 1.0
    dt_new = _proposed_dt(dt, err_norm, accepted)
    if accepted:
        return y5, dt_new, True
    return state, dt_new, False


@njit(cache=True, fastmath=True)
def dopri54_step(rhs_func, state, dt, rtol, atol, *rhs_args):
    k1 = rhs_func(state, *rhs_args)
    k2 = rhs_func(state + dt * (1.0 / 5.0) * k1, *rhs_args)
    k3 = rhs_func(state + dt * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2), *rhs_args)
    k4 = rhs_func(
        state + dt * ((44.0 / 45.0) * k1 - (56.0 / 15.0) * k2 + (32.0 / 9.0) * k3),
        *rhs_args,
    )
    k5 = rhs_func(
        state
        + dt
        * (
            (19372.0 / 6561.0) * k1
            - (25360.0 / 2187.0) * k2
            + (64448.0 / 6561.0) * k3
            - (212.0 / 729.0) * k4
        ),
        *rhs_args,
    )
    k6 = rhs_func(
        state
        + dt
        * (
            (9017.0 / 3168.0) * k1
            - (355.0 / 33.0) * k2
            + (46732.0 / 5247.0) * k3
            + (49.0 / 176.0) * k4
            - (5103.0 / 18656.0) * k5
        ),
        *rhs_args,
    )
    k7 = rhs_func(
        state
        + dt
        * (
            (35.0 / 384.0) * k1
            + (500.0 / 1113.0) * k3
            + (125.0 / 192.0) * k4
            - (2187.0 / 6784.0) * k5
            + (11.0 / 84.0) * k6
        ),
        *rhs_args,
    )

    y5 = state + dt * (
        (35.0 / 384.0) * k1
        + (500.0 / 1113.0) * k3
        + (125.0 / 192.0) * k4
        - (2187.0 / 6784.0) * k5
        + (11.0 / 84.0) * k6
    )
    y4 = state + dt * (
        (5179.0 / 57600.0) * k1
        + (7571.0 / 16695.0) * k3
        + (393.0 / 640.0) * k4
        - (92097.0 / 339200.0) * k5
        + (187.0 / 2100.0) * k6
        + (1.0 / 40.0) * k7
    )

    err = y5 - y4
    err_norm = _error_norm(err, state, y5, rtol, atol)
    accepted = np.isfinite(err_norm) and err_norm <= 1.0
    dt_new = _proposed_dt(dt, err_norm, accepted)
    if accepted:
        return y5, dt_new, True
    return state, dt_new, False


@njit(cache=True, fastmath=True)
def integrate_numba_scheme(
    rhs_func,
    scheme_id,
    state0,
    dt,
    n_steps,
    lambda_stop,
    record_every,
    rtol,
    atol,
    dt_min,
    dt_max,
    max_rejected,
    *rhs_args,
):
    state = state0.copy()
    lam = 0.0
    accepted_steps = 0
    rejected_total = 0
    rejected_consecutive = 0
    rejected_streak_peak = 0
    attempts = 0
    n_state = state.shape[0]
    stop_reason = STOP_REASON_UNKNOWN
    min_abs_dt_attempted = np.inf
    last_dt_attempted = 0.0
    last_dt_suggested = float(dt)

    if record_every <= 0:
        n_rec = 1
    else:
        n_rec = 1 + n_steps // record_every + 2
    traj = np.empty((n_rec, n_state))
    traj[0] = state
    rec_idx = 1

    use_lam_stop = lambda_stop > 0.0
    max_attempts = max(n_steps * 20, n_steps + max_rejected + 16)

    while accepted_steps < n_steps and attempts < max_attempts:
        if use_lam_stop and abs(lam) >= lambda_stop:
            stop_reason = STOP_REASON_LAMBDA_STOP
            break

        dt_eff = _clamp_signed_dt(dt, dt_min, dt_max)
        last_dt_attempted = dt_eff
        abs_dt_eff = abs(dt_eff)
        if abs_dt_eff < min_abs_dt_attempted:
            min_abs_dt_attempted = abs_dt_eff
        if use_lam_stop:
            remaining = lambda_stop - abs(lam)
            if remaining <= 0.0:
                stop_reason = STOP_REASON_LAMBDA_STOP
                break
            if remaining < abs(dt_eff):
                dt_eff = remaining if dt_eff >= 0.0 else -remaining

        if scheme_id == SCHEME_RK4:
            candidate = rk4_step(rhs_func, state, dt_eff, *rhs_args)
            accepted = True
            dt_new = dt_eff
        elif scheme_id == SCHEME_RKF45:
            candidate, dt_new, accepted = rkf45_step(rhs_func, state, dt_eff, rtol, atol, *rhs_args)
        else:
            candidate, dt_new, accepted = dopri54_step(rhs_func, state, dt_eff, rtol, atol, *rhs_args)
        last_dt_suggested = dt_new

        if accepted:
            state = candidate
            lam += dt_eff
            accepted_steps += 1
            rejected_consecutive = 0
            if record_every > 0 and (accepted_steps % record_every) == 0 and rec_idx < n_rec:
                traj[rec_idx] = state
                rec_idx += 1
        else:
            rejected_total += 1
            rejected_consecutive += 1
            if rejected_consecutive > rejected_streak_peak:
                rejected_streak_peak = rejected_consecutive
            if rejected_consecutive > max_rejected:
                stop_reason = STOP_REASON_MAX_REJECTED
                break

        if not np.isfinite(dt_new) or abs(dt_new) < 1e-30:
            stop_reason = STOP_REASON_DT_INVALID
            break

        dt = _clamp_signed_dt(dt_new, dt_min, dt_max)
        attempts += 1

    if rec_idx < n_rec:
        traj[rec_idx] = state
        rec_idx += 1

    if stop_reason == STOP_REASON_UNKNOWN:
        if use_lam_stop and abs(lam) >= lambda_stop:
            stop_reason = STOP_REASON_LAMBDA_STOP
        elif accepted_steps >= n_steps:
            stop_reason = STOP_REASON_STEP_BUDGET
        elif attempts >= max_attempts:
            stop_reason = STOP_REASON_MAX_ATTEMPTS

    if not np.isfinite(min_abs_dt_attempted):
        min_abs_dt_attempted = abs(_clamp_signed_dt(dt, dt_min, dt_max))

    return (
        traj[:rec_idx],
        state,
        rec_idx,
        lam,
        accepted_steps,
        stop_reason,
        rejected_total,
        rejected_streak_peak,
        min_abs_dt_attempted,
        last_dt_attempted,
        last_dt_suggested,
    )