#!/usr/bin/env python3
r"""
Validation of :mod:`excalibur.observables.szekeres_distances` (area distance).

Tests (brief sec. 6, tests 2, 3):
    * In the Einstein--de Sitter limit (flat dust, dipole off) the Sachs/Jacobi
      area distance matches the closed-form FLRW oracle
          D_A(t_e) = a(t_e) |Delta r| = t_e^{2/3} |r_o - r_e|
      along a radial ray, including the characteristic D_A turnover with z.
    * Etherington reciprocity  D_L = (1+z)^2 D_A  holds (definitional).

Curvature is ill-conditioned in SI, so force cosmo units before importing.
Run directly:  ``python _tests/test_szekeres_distances.py``
"""
import os
import sys

os.environ.setdefault("EXCALIBUR_UNITS", "cosmo")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c, unit_system
from excalibur.core.szekeres_model import SzekeresModel
from excalibur.metrics.szekeres_metric import SzekeresMetric
from excalibur.observables.szekeres_distances import integrate_area_distance, init_screen

assert c < 1e3, (
    f"Szekeres distance tests need cosmo units (c~O(1)); got c={c:.3e} "
    f"(unit_system={unit_system!r}).  Run standalone."
)

M0 = 2.0 / (9.0 * G)


def _eds():
    fd = {"dM": lambda r: 3.0 * M0 * r * r, "dk": lambda r: 0.0,
          "dS": lambda r: 0.0, "dP": lambda r: 0.0, "dQ": lambda r: 0.0,
          "dt_B": lambda r: 0.0, "ddS": lambda r: 0.0,
          "ddP": lambda r: 0.0, "ddQ": lambda r: 0.0}
    return SzekeresModel(M=lambda r: M0 * r ** 3, k=lambda r: 0.0,
                         S=lambda r: 1.0, P=lambda r: 0.0, Q=lambda r: 0.0,
                         t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
                         r_min=0.2, r_max=4.5, n_r=200, n_t=400,
                         t_min=1.0, t_max=22.0, free_derivs=fd)


def _dipole():
    return SzekeresModel(
        M=lambda r: M0 * r ** 3 * (1.0 + 0.2 * np.exp(-((r - 2.0) / 0.8) ** 2)),
        k=lambda r: 0.04 * r ** 2 * np.exp(-((r - 2.0) / 0.8) ** 2),
        S=lambda r: 0.9 + 0.1 * r, P=lambda r: 0.15 * np.sin(0.6 * r),
        Q=lambda r: -0.1 * r, t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
        r_min=0.2, r_max=4.5, n_r=220, n_t=440, t_min=1.0, t_max=22.0)


def _lcdm(H0=0.0715, Om=0.3, OL=0.7):
    """Flat LambdaCDM Szekeres background (dipole off, Lambda != 0).

    ``M(r)=M0 r^3``, ``k=0``, ``Lambda=3 H0^2 OL/c^2`` with ``2 G M0=H0^2 Om``
    reduces the Friedmann--LTB equation to the flat-LambdaCDM cosmic-time
    Friedmann equation, so ``Phi=a(t) r`` with the LambdaCDM ``a(t)``.
    """
    M0 = H0 ** 2 * Om / (2.0 * G)
    Lam = 3.0 * H0 ** 2 * OL / (c * c)
    fd = {"dM": lambda r: 3.0 * M0 * r * r, "dk": lambda r: 0.0,
          "dS": lambda r: 0.0, "dP": lambda r: 0.0, "dQ": lambda r: 0.0,
          "dt_B": lambda r: 0.0, "ddS": lambda r: 0.0,
          "ddP": lambda r: 0.0, "ddQ": lambda r: 0.0}
    return SzekeresModel(M=lambda r: M0 * r ** 3, k=lambda r: 0.0,
                         S=lambda r: 1.0, P=lambda r: 0.0, Q=lambda r: 0.0,
                         t_B=lambda r: 0.0, Lambda=Lam, epsilon=1,
                         r_min=0.15, r_max=5.0, n_r=220, n_t=440,
                         t_min=1.0, t_max=30.0, free_derivs=fd)


_STOP = lambda s: s[0] < 1.3 or s[1] < 0.25 or s[1] > 4.4


def test_lcdm_flrw_limit():
    r"""With the dipole off and Lambda != 0, D_A matches the FLRW area distance.

    Universal FLRW oracle (any homogeneous limit, since Phi = a(t) r and the
    comoving radial coordinate r is the comoving distance):

        D_A = a(t_e) |Delta r| = (Phi(t_e, r_e) / r_e) |r_o - r_e| .
    """
    model = _lcdm()
    met = SzekeresMetric(model)
    t_o, r_o = 13.8, 4.5
    stop = lambda s: s[0] < 1.3 or s[1] < 0.2 or s[1] > 4.9

    # homogeneity: a = Phi/r must be r-independent
    for t in (4.0, 8.0, 13.8):
        a = np.array([model.Phi(t, r) / r for r in np.linspace(0.3, 4.5, 6)])
        assert np.ptp(a) / np.mean(a) < 1e-3
    assert abs(model.Phi(13.8, 2.0) / 2.0 - 1.0) < 0.05   # a(t_0) ~ 1

    res = integrate_area_distance(met, model, [t_o, r_o, 0, 0], [-1.0, 0, 0],
                                  n_steps=7000, span_t=14.0, stop=stop)
    t, r, z, D_A = res["t"], res["r"], res["z"], res["D_A"]
    a_e = np.array([model.Phi(te, re) / re for te, re in zip(t, r)])
    oracle = a_e * np.abs(r_o - r)
    m = z > 0.05
    rel = np.max(np.abs(D_A[m] - oracle[m]) / oracle[m])
    assert z[-1] > 1.0 and rel < 1e-3, f"LCDM D_A rel-err {rel:.2e}"
    print(f"[ok] LambdaCDM FLRW limit: D_A = a(t_e)|Delta r| up to z={z[-1]:.2f} "
          f"(max rel-err {rel:.2e})")


def test_eds_area_distance_and_reciprocity():
    model = _eds()
    met = SzekeresMetric(model)
    t_o, r_o = 18.0, 3.8
    stop = lambda s: s[0] < 1.3 or s[1] < 0.25 or s[1] > 4.4
    res = integrate_area_distance(met, model, [t_o, r_o, 0, 0], [-1.0, 0, 0],
                                  n_steps=6000, span_t=11.0, stop=stop)
    t, r, z = res["t"], res["r"], res["z"]
    D_A, D_L = res["D_A"], res["D_L"]

    oracle = t ** (2.0 / 3.0) * np.abs(r_o - r)        # a(t_e) |Delta r|
    m = z > 0.05
    rel_A = np.max(np.abs(D_A[m] - oracle[m]) / oracle[m])
    rel_L = np.max(np.abs(D_L[m] - (1 + z[m]) ** 2 * D_A[m]) / D_L[m])

    assert z[-1] > 1.5, f"ray did not reach cosmological z (z={z[-1]:.2f})"
    assert rel_A < 1e-3, f"D_A vs EdS oracle {rel_A:.2e}"
    assert rel_L < 1e-12, f"D_L=(1+z)^2 D_A {rel_L:.2e}"
    # D_A must turn over (rise then fall) -- the hallmark of angular distance.
    assert D_A.argmax() not in (0, len(D_A) - 1), "no D_A turnover seen"

    print(f"[ok] EdS area distance D_A = a(t_e)|Delta r| up to z={z[-1]:.2f} "
          f"(max rel-err {rel_A:.2e})")
    print(f"[ok] reciprocity D_L=(1+z)^2 D_A  (max rel-err {rel_L:.2e})")
    print(f"[ok] D_A turnover at z={z[D_A.argmax()]:.2f}, D_A,max={D_A.max():.3f}")


def test_paper_screen_is_orthonormal():
    """Sachs screen (eqs. 85-86, p<->q-corrected) satisfies conditions (84)."""
    model = _dipole()
    met = SzekeresMetric(model)
    x = np.array([16.0, 3.0, 0.1, -0.05])
    ks = [-1.0, 0.03, -0.02]
    kt = met.solve_null_kt(x, ks)
    k = np.array([kt, *ks])
    e1, e2 = init_screen(met, model, x, k)
    g = met.metric_tensor(x)
    assert abs(e1 @ g @ e1 - 1) < 1e-12 and abs(e2 @ g @ e2 - 1) < 1e-12
    assert abs(e1 @ g @ e2) < 1e-12
    assert abs(e1 @ g @ k) < 1e-10 and abs(e2 @ g @ k) < 1e-10
    print("[ok] Sachs screen orthonormal & orthogonal to k (eqs. 84-86)")


def test_homogeneous_model_has_no_shear():
    """A non-radial ray in homogeneous EdS must develop NO shear/rotation."""
    model = _eds()
    met = SzekeresMetric(model)
    res = integrate_area_distance(met, model, [18.0, 3.8, 0, 0], [-1.0, 0.03, -0.02],
                                  n_steps=7000, span_t=11.0, stop=_STOP)
    m = res["z"] > 0.1
    assert res["z"][-1] > 1.5
    assert res["gamma"][m].max() < 1e-4, f"spurious shear {res['gamma'][m].max():.2e}"
    assert np.abs(res["omega"][m]).max() < 1e-10
    print(f"[ok] homogeneous (non-radial) ray: max|gamma|={res['gamma'][m].max():.2e}, "
          f"max|omega|={np.abs(res['omega'][m]).max():.2e}")


def test_inhomogeneous_model_shears_the_beam():
    """The dipole model must shear the beam well above the homogeneous floor."""
    model = _dipole()
    met = SzekeresMetric(model)
    res = integrate_area_distance(met, model, [18.0, 3.8, 0, 0], [-1.0, 0.02, -0.015],
                                  n_steps=8000, span_t=11.0, stop=_STOP)
    m = res["z"] > 0.1
    gmax = res["gamma"][m].max()
    assert gmax > 1e-4, f"dipole shear too small: {gmax:.2e}"
    print(f"[ok] inhomogeneous ray shears the beam: max|gamma|={gmax:.2e}")


def _D_A_at(model, kp, z_q=1.0):
    met = SzekeresMetric(model)
    res = integrate_area_distance(met, model, [18.0, 3.8, 0, 0], [-1.0, kp, 0.0],
                                  n_steps=5000, span_t=11.0, stop=_STOP)
    return np.interp(z_q, res["z"], res["D_A"])


def test_dipole_breaks_direction_symmetry():
    r"""Szekeres dipole: D_A(+k^p) != D_A(-k^p); FLRW/EdS stays symmetric.

    The hallmark distinguishing Szekeres from LTB/FLRW -- a dipolar distance
    response across the sky.
    """
    kp = 0.06
    eds = _eds()
    asym_eds = abs(_D_A_at(eds, kp) - _D_A_at(eds, -kp)) / _D_A_at(eds, kp)
    dip = _dipole()
    asym_dip = abs(_D_A_at(dip, kp) - _D_A_at(dip, -kp)) / _D_A_at(dip, kp)

    assert asym_eds < 1e-4, f"EdS should be isotropic, got {asym_eds:.2e}"
    assert asym_dip > 5.0 * max(asym_eds, 1e-5), \
        f"dipole asymmetry {asym_dip:.2e} not above EdS floor {asym_eds:.2e}"
    print(f"[ok] direction symmetry: EdS asym={asym_eds:.2e}, "
          f"dipole asym={asym_dip:.2e} (Szekeres dipole signature)")


if __name__ == "__main__":
    test_eds_area_distance_and_reciprocity()
    test_lcdm_flrw_limit()
    test_paper_screen_is_orthonormal()
    test_homogeneous_model_has_no_shear()
    test_inhomogeneous_model_shears_the_beam()
    test_dipole_breaks_direction_symmetry()
    print("\nAll Szekeres distance tests passed.")
