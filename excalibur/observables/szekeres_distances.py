r"""
Observer area distance ``D_A`` (and luminosity distance ``D_L``) along a null
geodesic of the quasi-spherical Szekeres metric, via the Sachs/Jacobi route
(Celerier 2024 eqs. 25-27).

The 24-component state ``[x(4), k(4), e1(4), e2(4), D(4), P(4)]`` is integrated:

* ``x, k``      -- the null geodesic (``SzekeresMetric.geodesic_equations``);
* ``e1, e2``    -- the Sachs screen basis, screen-projected parallel transport;
* ``D, P``      -- the 2x2 Jacobi map, ``dD/dlam = P``, ``dP/dlam = -R_AB D``,
                   with ``R_AB = T_{mu nu} e_A^mu e_B^nu`` and the analytic tidal
                   tensor ``T_{mu nu} = R_{mu a nu b} k^a k^b``
                   (:func:`excalibur.observables.riemann_szekeres.tidal_tensor_analytic`).

The physical area distance is the normalisation-independent combination

    D_A = c * k^t_o * sqrt(|det D|),

with ``D(0)=0``, ``P(0)=I`` (so that ``sqrt|det D| ~ lambda`` near the observer
and the ``c k^t_o`` factor restores proper length).  Etherington reciprocity then
gives ``D_L = (1 + z)^2 D_A``.

Run Szekeres in cosmo units (``EXCALIBUR_UNITS=cosmo``); the curvature is badly
conditioned in SI.
"""
import numpy as np

from excalibur.core.constants import c
from excalibur.observables import riemann_szekeres as rs
from excalibur.observables.sachs_basis import sachs_transport_rhs
from excalibur.observables.optical_tidal_matrix import jacobi_rhs


def _e_geometric(e):
    """Screen 4-vector in the geometric basis (``e^t~ = c e^t``)."""
    return np.array([c * e[0], e[1], e[2], e[3]])


def optical_tidal_matrix_szekeres(model, x, k, e1, e2):
    r"""2x2 optical tidal matrix ``R_{AB} = T_{mu nu} e_A^mu e_B^nu`` (analytic ``T``)."""
    T = rs.tidal_tensor_analytic(model, x, k)        # geometric basis
    e1g, e2g = _e_geometric(e1), _e_geometric(e2)
    e = (e1g, e2g)
    R = np.empty((2, 2))
    for A in range(2):
        for B in range(2):
            R[A, B] = e[A] @ (T @ e[B])
    return R


def init_screen(metric, model, x, k):
    r"""Sachs screen basis ``e1, e2`` at the observer (Celerier 2024 eqs. 85-86).

    .. math::
        E_1^\mu = \Big(0,\; \tfrac{F}{H}N,\;
                       -\tfrac{H}{F}\tfrac{k^r k^p}{N},\;
                       -\tfrac{H}{F}\tfrac{k^r k^q}{N}\Big),\qquad
        E_2^\mu = \Big(0,\;0,\; \tfrac{k^q}{F N},\; -\tfrac{k^p}{F N}\Big),

    with ``N = sqrt((k^p)^2 + (k^q)^2)``.

    .. note::
        The published eq. (86) prints ``E_2 = (0,0, k^p/(FN), -k^q/(FN))``, which
        violates the orthogonality conditions (84) it is meant to satisfy
        (``E_2 . k = F[(k^p)^2-(k^q)^2]/N != 0`` and ``E_1 . E_2 != 0``).  The
        ``p <-> q`` swap above restores ``E_2 . k = E_1 . E_2 = 0`` -- a typo in
        the paper.  For a **radial** ray (``k^p = k^q = 0``, eqs. 85-86 are
        singular) we fall back to the ``(p, q)`` coordinate screen.

    Vectors are normalised in the physical metric (the closed forms are unit only
    when ``k`` is normalised to ``k^t~ = -1``; normalising here is independent of
    the affine scale).
    """
    x = np.asarray(x, dtype=float)
    b = rs.hf_bundle(model, x)
    H, F = b["H"], b["F"]
    kr, kp, kq = k[1], k[2], k[3]
    N2 = kp * kp + kq * kq

    if N2 <= 1e-30 * max(1.0, kr * kr):       # radial: use the (p, q) screen
        e1 = np.array([0.0, 0.0, 1.0 / F, 0.0])
        e2 = np.array([0.0, 0.0, 0.0, 1.0 / F])
    else:
        N = np.sqrt(N2)
        e1 = np.array([0.0, (F / H) * N,
                       -(H / F) * kr * kp / N, -(H / F) * kr * kq / N])
        e2 = np.array([0.0, 0.0, kq / (F * N), -kp / (F * N)])    # (84)-consistent

    g = metric.metric_tensor(x)
    e1 = e1 / np.sqrt(e1 @ (g @ e1))
    e2 = e2 / np.sqrt(e2 @ (g @ e2))
    return e1, e2


def lensing_scalars(D_flat):
    r"""Convergence/shear relative to the ray's own area distance (Celerier eqs. 68-71).

    Returns ``(gamma1, gamma2, gamma, omega)`` normalised by ``D_A = sqrt|det D|``
    of the ray, so they vanish for an isotropic (unsheared) beam regardless of the
    background normalisation.  For the paper's convergence ``kappa`` (eq. 68) one
    rescales by the FLRW background ``D_A^FLRW`` instead; here we expose the
    background-independent shear, the cleanest probe of spurious anisotropy.
    """
    D11, D12, D21, D22 = D_flat
    D_A = np.sqrt(abs(D11 * D22 - D12 * D21))
    if D_A < 1e-300:
        return 0.0, 0.0, 0.0, 0.0
    gamma1 = (D22 - D11) / (2.0 * D_A)
    gamma2 = 0.5 * (D12 + D21) / D_A
    omega = 0.5 * (D12 - D21) / D_A
    gamma = np.hypot(gamma1, gamma2)
    return gamma1, gamma2, gamma, omega


def _rhs(metric, model, state):
    """Full RHS of the 24-component distance state."""
    x = state[0:4]
    k = state[4:8]
    e1 = state[8:12]
    e2 = state[12:16]
    D = state[16:20]
    P = state[20:24]

    dgeo = metric.geodesic_equations(state[0:8])      # [k, dk]
    chris = metric.christoffel(x)
    de1 = sachs_transport_rhs(e1, chris, k)
    de2 = sachs_transport_rhs(e2, chris, k)
    R_AB = optical_tidal_matrix_szekeres(model, x, k, e1, e2)
    dj = jacobi_rhs(np.concatenate([D, P]), R_AB)      # [P, -R D]

    out = np.empty(24)
    out[0:8] = dgeo
    out[8:12] = de1
    out[12:16] = de2
    out[16:20] = dj[0:4]
    out[20:24] = dj[4:8]
    return out


def integrate_area_distance(metric, model, x0, k_spatial, *,
                            n_steps=6000, span_t=10.0, stop=None):
    r"""Backward-integrate a null ray and return distances along it.

    Parameters
    ----------
    metric : SzekeresMetric
    model : SzekeresModel
    x0 : array (4,)         observer position ``(t, r, p, q)``.
    k_spatial : array (3,)  spatial photon direction ``(k^r, k^p, k^q)``.
    n_steps, span_t : int, float
        Fixed-step RK4 budget; ``span_t`` sets the affine step from ``k^t_o``.
    stop : callable(state)->bool, optional
        Early-stop predicate (e.g. leaving the tabulated domain).

    Returns
    -------
    dict with arrays ``lam, t, r, z, D_A, D_L`` (one entry per accepted step).
    """
    x0 = np.asarray(x0, dtype=float)
    kt = metric.solve_null_kt(x0, k_spatial)          # past-pointing (k^t < 0)
    k0 = np.array([kt, *k_spatial], dtype=float)
    kt_o = kt

    e1, e2 = init_screen(metric, model, x0, k0)

    state = np.empty(24)
    state[0:4] = x0
    state[4:8] = k0
    state[8:12] = e1
    state[12:16] = e2
    state[16:20] = 0.0                  # D(0) = 0
    state[20:24] = np.array([1.0, 0.0, 0.0, 1.0])   # P(0) = I

    ds = abs(span_t / (kt_o * n_steps))
    lam = 0.0
    rec = {"lam": [], "t": [], "r": [], "z": [], "D_A": [], "D_L": [],
           "gamma": [], "gamma1": [], "gamma2": [], "omega": []}

    for _ in range(n_steps):
        k1 = _rhs(metric, model, state)
        k2 = _rhs(metric, model, state + 0.5 * ds * k1)
        k3 = _rhs(metric, model, state + 0.5 * ds * k2)
        k4 = _rhs(metric, model, state + ds * k3)
        state = state + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        lam += ds

        if stop is not None and stop(state):
            break

        t = state[0]
        D = state[16:20]
        det_D = D[0] * D[3] - D[1] * D[2]
        D_A = c * abs(kt_o) * np.sqrt(abs(det_D))
        z = state[4] / kt_o - 1.0
        g1, g2, gmag, om = lensing_scalars(D)
        rec["lam"].append(lam)
        rec["t"].append(t)
        rec["r"].append(state[1])
        rec["z"].append(z)
        rec["D_A"].append(D_A)
        rec["D_L"].append((1.0 + z) ** 2 * D_A)
        rec["gamma1"].append(g1)
        rec["gamma2"].append(g2)
        rec["gamma"].append(gmag)
        rec["omega"].append(om)

    return {key: np.asarray(val) for key, val in rec.items()}
