#!/usr/bin/env python3
r"""
Low-redshift cosmography from the Szekeres luminosity distance (brief sec. 6, test 7).

The luminosity-distance Hubble diagram expands at low z as

    D_L(z) = (c / H0) [ z + 1/2 (1 - q0) z^2 + O(z^3) ],

so a cubic fit of ``D_L(z)`` through the origin recovers the Hubble constant
``H0`` and the deceleration parameter ``q0``.  We check that the Szekeres pipeline
reproduces:

    * Einstein--de Sitter (flat dust):   q0 = +1/2   (decelerating),
    * flat LambdaCDM (Om=0.3, OL=0.7):   q0 = Om/2 - OL = -0.55  (accelerating),

i.e. it correctly captures cosmic deceleration/acceleration -- the observable at
the heart of the supernova program.

Run standalone (cosmo units):  ``python _tests/test_szekeres_cosmography.py``
"""
import os
import sys

os.environ.setdefault("EXCALIBUR_UNITS", "cosmo")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from excalibur.core.constants import G, c, unit_system
from excalibur.core.szekeres_model import SzekeresModel
from excalibur.metrics.szekeres_metric import SzekeresMetric
from excalibur.observables.szekeres_distances import integrate_area_distance

assert c < 1e3, f"cosmography test needs cosmo units; got c={c:.3e} ({unit_system})"

_STOP = lambda s: s[0] < 1.3 or s[1] < 0.2 or s[1] > 4.9


def _flrw(Om, OL, H0):
    M0 = H0 ** 2 * Om / (2.0 * G)
    Lam = 3.0 * H0 ** 2 * OL / (c * c)
    fd = {"dM": lambda r: 3.0 * M0 * r * r, "dk": lambda r: 0.0,
          "dS": lambda r: 0.0, "dP": lambda r: 0.0, "dQ": lambda r: 0.0,
          "dt_B": lambda r: 0.0, "ddS": lambda r: 0.0,
          "ddP": lambda r: 0.0, "ddQ": lambda r: 0.0}
    return SzekeresModel(M=lambda r: M0 * r ** 3, k=lambda r: 0.0,
                         S=lambda r: 1.0, P=lambda r: 0.0, Q=lambda r: 0.0,
                         t_B=lambda r: 0.0, Lambda=Lam, epsilon=1,
                         r_min=0.15, r_max=5.0, n_r=240, n_t=480,
                         t_min=1.0, t_max=30.0, free_derivs=fd)


def _fit_H0_q0(z, D_L, zmax=0.4):
    """Cubic fit of D_L through the origin -> (H0, q0)."""
    m = (z > 1e-3) & (z < zmax)
    zz, y = z[m], D_L[m] / z[m]           # D_L/z = b + c2 z + c3 z^2
    A = np.vstack([np.ones_like(zz), zz, zz ** 2]).T
    b, c2, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return c / b, 1.0 - 2.0 * c2 / b


def _eds():
    M0 = 2.0 / (9.0 * G)
    fd = {"dM": lambda r: 3.0 * M0 * r * r, "dk": lambda r: 0.0,
          "dS": lambda r: 0.0, "dP": lambda r: 0.0, "dQ": lambda r: 0.0,
          "dt_B": lambda r: 0.0, "ddS": lambda r: 0.0,
          "ddP": lambda r: 0.0, "ddQ": lambda r: 0.0}
    return SzekeresModel(M=lambda r: M0 * r ** 3, k=lambda r: 0.0,
                         S=lambda r: 1.0, P=lambda r: 0.0, Q=lambda r: 0.0,
                         t_B=lambda r: 0.0, Lambda=0.0, epsilon=1,
                         r_min=0.15, r_max=5.0, n_r=240, n_t=480,
                         t_min=1.0, t_max=22.0, free_derivs=fd)


def _run(model, t_o):
    met = SzekeresMetric(model)
    return integrate_area_distance(met, model, [t_o, 4.5, 0, 0], [-1.0, 0, 0],
                                   n_steps=9000, span_t=14.0, stop=_STOP)


def test_eds_cosmography_decelerating():
    t_o, H0_exp = 18.0, (2.0 / 3.0) / 18.0     # EdS: a~t^{2/3}, H0=(2/3)/t_o
    res = _run(_eds(), t_o)
    H0, q0 = _fit_H0_q0(res["z"], res["D_L"])
    assert abs(H0 - H0_exp) / H0_exp < 0.03, f"H0 {H0:.4f} vs {H0_exp:.4f}"
    assert abs(q0 - 0.5) < 0.05, f"q0 {q0:.3f} vs 0.5"
    print(f"[ok] EdS cosmography: H0={H0:.4f} (exp {H0_exp:.4f}), "
          f"q0={q0:.3f} (exp +0.500, decelerating)")


def test_lcdm_cosmography_accelerating():
    Om, OL, H0_exp = 0.3, 0.7, 0.0715
    q0_exp = Om / 2.0 - OL                      # = -0.55
    res = _run(_flrw(Om, OL, H0_exp), 13.8)
    H0, q0 = _fit_H0_q0(res["z"], res["D_L"])
    assert abs(H0 - H0_exp) / H0_exp < 0.03, f"H0 {H0:.4f} vs {H0_exp:.4f}"
    assert abs(q0 - q0_exp) < 0.05, f"q0 {q0:.3f} vs {q0_exp:.3f}"
    assert q0 < 0.0, "LambdaCDM must be accelerating (q0 < 0)"
    print(f"[ok] LambdaCDM cosmography: H0={H0:.4f} (exp {H0_exp:.4f}), "
          f"q0={q0:.3f} (exp {q0_exp:.3f}, accelerating)")


if __name__ == "__main__":
    test_eds_cosmography_decelerating()
    test_lcdm_cosmography_accelerating()
    print("\nAll Szekeres cosmography tests passed.")
