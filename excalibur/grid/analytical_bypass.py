"""
Analytical bypass interpolator.

Wraps an existing grid interpolator (InterpolatorFast / AMRInterpolator /
Interpolator4DFast) and transparently returns *analytical* values for the
potential, its gradient and its Hessian inside a user-defined bypass
region around an object of known profile (e.g. an NFW halo).

Motivation
----------
Finite-resolution AMR grids cannot resolve the cusp of spherical profiles
such as NFW: the reconstructed convergence kappa_sim plateaus at
kappa_an(b ~ dx_finest) smoothed ~0.25-0.8x and never reaches the
strong-lensing regime (kappa > 1).  By switching to the exact formula
inside a small sphere of radius ``bypass_radius`` one recovers kappa > 1
and the associated multiple images, without affecting the outer
large-scale trajectory that the grid handles well.

Usage
-----
>>> from excalibur.grid.analytical_bypass import AnalyticalBypassInterpolator
>>> interp = AMRInterpolator(amr_grid)
>>> bypass_interp = AnalyticalBypassInterpolator(
...     base_interp=interp,
...     analytical_source=halo,          # must expose potential/grad/hess
...     bypass_radius=2.0 * dx_finest,
... )
>>> metric = PerturbedFLRWMetricFast(..., interp=bypass_interp)

The ``analytical_source`` object must provide:

- ``potential(x, y, z)        -> float``
- ``potential_gradient(x,y,z) -> (gx, gy, gz)``
- ``potential_hessian(x,y,z)  -> ndarray (3,3)``
- ``center                     ndarray (3,)``  (used as the bypass centre)

Time derivatives are taken to be zero inside the bypass (static source).
If your source evolves, pass ``time_derivative`` as a float or as a
callable ``f(pos, t) -> float``.
"""

import numpy as np


class AnalyticalBypassInterpolator:
    r"""
    Wraps a grid interpolator, replacing its output by analytical values
    for the ``"Phi"`` field (or whatever ``field`` name is passed) inside
    a sphere of radius ``bypass_radius`` around ``analytical_source.center``.

    Parameters
    ----------
    base_interp
        Any interpolator exposing the standard Excalibur API
        (``value_gradient_hessian_and_time_derivative`` etc).
    analytical_source
        Object providing ``potential``, ``potential_gradient``,
        ``potential_hessian`` and a ``center`` attribute.
    bypass_radius : float
        Radius of the bypass sphere [same units as ``center``].
    bypass_fields : iterable of str, optional
        Names of the fields for which the bypass is active (default
        ``{"Phi"}``).  Queries for any other field are passed through
        unchanged to ``base_interp``.
    time_derivative : float or callable, optional
        Time derivative of the analytical potential.  Defaults to 0
        (static source).  If callable, signature ``f(pos, t) -> float``.
    """

    def __init__(self, base_interp, analytical_source, bypass_radius,
                 bypass_fields=("Phi",), time_derivative=0.0):
        self.base = base_interp
        self.src = analytical_source
        self.center = np.asarray(analytical_source.center, dtype=float)
        self.bypass_radius = float(bypass_radius)
        self._bypass_r2 = self.bypass_radius ** 2
        self.bypass_fields = set(bypass_fields)
        self.time_derivative = time_derivative

        # Mirror attributes that downstream code reads directly
        self.grid = getattr(base_interp, "grid", None)
        self.boundary = getattr(base_interp, "boundary", "clamp")

    # ------------------------------------------------------------------
    #  Fast region test
    # ------------------------------------------------------------------
    def _in_bypass(self, pos):
        dx = pos[0] - self.center[0]
        dy = pos[1] - self.center[1]
        dz = pos[2] - self.center[2]
        return dx * dx + dy * dy + dz * dz < self._bypass_r2

    def _dtd(self, pos, t):
        if callable(self.time_derivative):
            return float(self.time_derivative(pos, t))
        return float(self.time_derivative)

    def _active(self, field, pos):
        return (field in self.bypass_fields) and self._in_bypass(pos)

    # ------------------------------------------------------------------
    #  Core API   (matches InterpolatorFast / AMRInterpolator)
    # ------------------------------------------------------------------
    def value_gradient_hessian_and_time_derivative(self, x, field, t=None):
        if not self._active(field, x):
            return self.base.value_gradient_hessian_and_time_derivative(x, field, t)
        phi = float(self.src.potential(x[0], x[1], x[2]))
        gx, gy, gz = self.src.potential_gradient(x[0], x[1], x[2])
        H = self.src.potential_hessian(x[0], x[1], x[2])
        hess3 = (H[0, 0], H[1, 1], H[2, 2], H[0, 1], H[0, 2], H[1, 2])
        return phi, (float(gx), float(gy), float(gz)), hess3, self._dtd(x, t)

    def value_gradient_and_time_derivative(self, x, field, t=None):
        if not self._active(field, x):
            return self.base.value_gradient_and_time_derivative(x, field, t)
        phi = float(self.src.potential(x[0], x[1], x[2]))
        gx, gy, gz = self.src.potential_gradient(x[0], x[1], x[2])
        return phi, (float(gx), float(gy), float(gz)), self._dtd(x, t)

    def interpolate(self, x, field, t=None):
        if not self._active(field, x):
            return self.base.interpolate(x, field, t)
        return float(self.src.potential(x[0], x[1], x[2]))

    def gradient(self, x, field, t=None):
        if not self._active(field, x):
            return self.base.gradient(x, field, t)
        gx, gy, gz = self.src.potential_gradient(x[0], x[1], x[2])
        return float(gx), float(gy), float(gz)

    def value_and_gradient(self, x, field, t=None):
        if not self._active(field, x):
            return self.base.value_and_gradient(x, field, t)
        phi = float(self.src.potential(x[0], x[1], x[2]))
        gx, gy, gz = self.src.potential_gradient(x[0], x[1], x[2])
        return phi, (float(gx), float(gy), float(gz))

    def hessian(self, x, field, t=None):
        if not self._active(field, x):
            return self.base.hessian(x, field, t)
        return self.src.potential_hessian(x[0], x[1], x[2])

    def laplacian(self, x, field, t=None):
        if not self._active(field, x):
            return self.base.laplacian(x, field, t)
        H = self.src.potential_hessian(x[0], x[1], x[2])
        return float(H[0, 0] + H[1, 1] + H[2, 2])
