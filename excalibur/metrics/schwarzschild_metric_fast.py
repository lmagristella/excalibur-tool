# metrics/schwarzschild_metric_fast.py
import numpy as np
from .base_metric import Metric
from excalibur.core.constants import G, c, one_Gpc
from excalibur.core.coordinates import (
    cartesian_to_spherical,
    cartesian_velocity_to_spherical,
)
from numba import njit


@njit(cache=True, fastmath=True)
def _schw_accel_tensorial(
    r, theta, rs,
    u0, u1, u2, u3,   # (dt/dλ, dr/dλ, dθ/dλ, dφ/dλ)
    c_val,
):
    """
    Compute du/dλ for Schwarzschild in spherical coords by inlining Γ^μ_{αβ}.

    Metric (SI-consistent):
      ds² = -(1 - rs/r) c² dt² + (1 - rs/r)^(-1) dr² + r² dθ² + r² sin²θ dφ²
    with rs = 2GM/c².

    Returns: (du0, du1, du2, du3)
    """
    # Guards (caller should already enforce r > rs and r >= radius)
    # Still keep minimal numerical safety here.
    if r <= rs:
        # Hard stop signal via NaNs (lets caller raise if it wants)
        return np.nan, np.nan, np.nan, np.nan

    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    # Common factors
    inv_r = 1.0 / r
    r_minus_rs = r - rs
    inv_r_minus_rs = 1.0 / r_minus_rs

    # Useful combos
    u0u1 = u0 * u1
    u2u2 = u2 * u2
    u3u3 = u3 * u3

    # Γ^0_{10} = Γ^0_{01} = rs / (2 r (r-rs))
    Gamma_010 = 0.5 * rs * inv_r * inv_r_minus_rs

    # Γ^1_{00} = (c² rs (r-rs)) / (2 r^3)
    # (derived from your existing Γ[1,0,0] expression)
    Gamma_100 = 0.5 * (c_val * c_val) * rs * (r_minus_rs) / (r * r * r)

    # Γ^1_{11} = -rs / (2 r (r-rs))
    Gamma_111 = -Gamma_010

    # Γ^1_{22} = -(r-rs)
    Gamma_122 = -(r_minus_rs)

    # Γ^1_{33} = -(r-rs) sin²θ
    Gamma_133 = -(r_minus_rs) * (sin_th * sin_th)

    # Γ^2_{12} = Γ^2_{21} = 1/r
    Gamma_212 = inv_r

    # Γ^2_{33} = -sinθ cosθ
    Gamma_233 = -sin_th * cos_th

    # Γ^3_{13} = Γ^3_{31} = 1/r
    Gamma_313 = inv_r

    # Γ^3_{23} = Γ^3_{32} = cosθ/sinθ
    # (avoid blow-up at poles; caller should avoid exactly sin=0)
    Gamma_323 = cos_th / sin_th

    # du^μ = - Γ^μ_{αβ} u^α u^β  (sum over αβ)
    # μ=0: only terms (0,1,0) and (0,0,1) => 2 Γ^0_{10} u0 u1
    du0 = -(2.0 * Gamma_010 * u0u1)

    # μ=1: terms 00, 11, 22, 33
    du1 = -(
        Gamma_100 * (u0 * u0) +
        Gamma_111 * (u1 * u1) +
        Gamma_122 * u2u2 +
        Gamma_133 * u3u3
    )

    # μ=2: terms 12/21 and 33
    # 2 * Γ^2_{12} u1 u2 + Γ^2_{33} u3^2
    du2 = -(
        2.0 * Gamma_212 * u1 * u2 +
        Gamma_233 * u3u3
    )

    # μ=3: terms 13/31 and 23/32
    # 2 * Γ^3_{13} u1 u3 + 2 * Γ^3_{23} u2 u3
    du3 = -(
        2.0 * Gamma_313 * u1 * u3 +
        2.0 * Gamma_323 * u2 * u3
    )

    return du0, du1, du2, du3


@njit(cache=True, fastmath=True)
def _schw_accel_analytical(
    r, theta, rs,
    dtdl, drdl, dthetadl, dphidl,
    c_val,
    free_time_geodesic,
):
    """
    Your existing analytical form, but Numba-compiled.
    If free_time_geodesic=False, we enforce null normalization for dt/dλ and set du0=0.
    Returns updated (dtdl_eff, du0, du1, du2, du3).
    """
    if r <= rs:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    # du0
    if free_time_geodesic:
        du0 = -(rs / (r * (r - rs))) * dtdl * drdl
        dtdl_eff = dtdl
    else:
        # Enforce null condition:
        # dtdl = -(1/c) * sqrt( (1/f)^2 dr^2 + r^2/f dθ^2 + r^2 sin^2θ/f dφ^2 )
        f = 1.0 - rs / r
        inv_f = 1.0 / f
        inv_f2 = inv_f * inv_f

        v = inv_f2 * (drdl * drdl) + (r * r) * inv_f * (dthetadl * dthetadl) + (r * r) * (sin_th * sin_th) * inv_f * (dphidl * dphidl)
        dtdl_eff = -(1.0 / c_val) * np.sqrt(v)
        du0 = 0.0

    # du1 (same as in your current analytical)
    du1 = (
        rs / (2.0 * r * (r - rs)) * drdl * drdl
        - (r - rs) / (2.0 * r**3)
        * ((c_val * c_val) * rs * dtdl_eff * dtdl_eff
           - 2.0 * r**3 * (dthetadl * dthetadl + (sin_th * sin_th) * dphidl * dphidl))
    )

    # du2
    du2 = (
        -2.0 / r * drdl * dthetadl
        + sin_th * cos_th * dphidl * dphidl
    )

    # du3
    du3 = (
        -2.0 / r * drdl * dphidl
        - 2.0 * (cos_th / sin_th) * dthetadl * dphidl
    )

    return dtdl_eff, du0, du1, du2, du3


class SchwarzschildMetricFast(Metric):
    """
    Fast Schwarzschild metric:
    - Internal coords: spherical (t, r, theta, phi)
    - Optional input coords: cartesian (t, x, y, z)

    Focus: fast geodesic_equations() for RK loops.
    """

    def __init__(
        self,
        mass,
        radius,
        center=np.array([0.5, 0.5, 0.5]) * one_Gpc,
        analytical_geodesics=False,
        free_time_geodesic=True,
        coords="cartesian",
    ):
        self.mass = float(mass)
        self.radius = float(radius)
        self.center = np.asarray(center, dtype=float)

        self.r_s = 2.0 * G * self.mass / (c * c)

        self.analytical_geodesics = bool(analytical_geodesics)
        self.free_time_geodesic = bool(free_time_geodesic)

        self.input_coords = coords
        self.internal_coords = "spherical"

    def metric_tensor(self, x):
        # Kept for compatibility/debug; not used in fast RK loop.
        t, pos = x[0], x[1:]
        r = float(pos[0])
        f = 1.0 - self.r_s / r

        g = np.zeros((4, 4))
        g[0, 0] = -f * (c * c)
        g[1, 1] = 1.0 / f
        g[2, 2] = r * r
        g[3, 3] = (r * r) * (np.sin(pos[1]) ** 2)
        return g
    
    def christoffel(self, x):
        """
        IMPORTANT: x is expected to be (t, r, theta, phi) with r measured FROM THE MASS CENTER.
        If you pass cartesian states, geodesic_equations will convert them using cartesian_state_to_spherical(),
        which already subtracts self.center.
        """
        t, r, theta, phi = x
        r = float(r)
        theta = float(theta)

        if (not np.isfinite(r)) or (not np.isfinite(theta)) or (not np.isfinite(phi)):
            raise ValueError("Non-finite (r,theta,phi) in SchwarzschildMetric")

        if r <= self.radius:
            r = self.radius
        if r <= self.r_s:
            raise ValueError("Photon inside the Schwarzschild radius, stopping integration")

        M = self.mass
        rs = self.r_s

        # Avoid division by extremely small denominators (numerical guard)
        denom = c**2 * r - 2 * G * M
        if abs(denom) < 1e-300:
            raise ValueError("Denominator too small in Christoffel computation (near horizon)")

        Γ = np.zeros((4, 4, 4), dtype=float)

        # Non-zero Christoffels for Schwarzschild (standard form)
        Γ[0, 1, 0] = G * M / (r * denom)
        Γ[0, 0, 1] = Γ[0, 1, 0]

        Γ[1, 0, 0] = G * M * (c**2 * r - 2 * G * M) / (c**2 * r**3)
        Γ[1, 1, 1] = -G * M / (r * denom)

        Γ[1, 2, 2] = -(r - rs)
        Γ[1, 3, 3] = -(r - rs) * (np.sin(theta) ** 2)

        Γ[2, 1, 2] = 1.0 / r
        Γ[2, 2, 1] = 1.0 / r
        Γ[2, 3, 3] = -np.sin(theta) * np.cos(theta)

        Γ[3, 1, 3] = 1.0 / r
        Γ[3, 3, 1] = 1.0 / r
        Γ[3, 2, 3] = np.cos(theta) / np.sin(theta)
        Γ[3, 3, 2] = Γ[3, 2, 3]

        return Γ


    def geodesic_equations(self, state):
        # Convert to spherical if needed (same convention as your current SchwarzschildMetric). :contentReference[oaicite:2]{index=2}
        if self.input_coords == "cartesian":
            state = self.cartesian_state_to_spherical(state)
        elif self.input_coords != "spherical":
            raise ValueError("coords must be 'cartesian' or 'spherical'")

        t, r, theta, phi = state[0], float(state[1]), float(state[2]), float(state[3])
        u0, u1, u2, u3 = float(state[4]), float(state[5]), float(state[6]), float(state[7])

        # Physical guards (same intent as your current implementation). :contentReference[oaicite:3]{index=3}
        if (not np.isfinite(r)) or (not np.isfinite(theta)) or (not np.isfinite(phi)):
            raise ValueError("Non-finite (r,theta,phi) in SchwarzschildMetricFast")

        if r <= self.radius:
            r = self.radius
        if r <= self.r_s:
            raise ValueError("Photon inside the Schwarzschild radius, stopping integration")

        # Compute accelerations
        if self.analytical_geodesics:
            dtdl_eff, du0, du1, du2, du3 = _schw_accel_analytical(
                r, theta, self.r_s,
                u0, u1, u2, u3,
                c,
                self.free_time_geodesic,
            )
            # If we enforce dt/dλ, overwrite u0 for consistent output derivative of t
            u0_out = dtdl_eff
        else:
            du0, du1, du2, du3 = _schw_accel_tensorial(
                r, theta, self.r_s,
                u0, u1, u2, u3,
                c,
            )
            u0_out = u0

        if not np.isfinite(du0):
            raise ValueError("Numerical failure in Schwarzschild fast acceleration (near horizon / poles)")

        # Build output without concatenation (fast path like FLRW fast). :contentReference[oaicite:4]{index=4}
        out = np.empty(8, dtype=float)
        out[0] = u0_out
        out[1] = u1
        out[2] = u2
        out[3] = u3
        out[4] = du0
        out[5] = du1
        out[6] = du2
        out[7] = du3
        return out

    def metric_physical_quantities(self, state):
        # Expect spherical (t,r,theta,phi) if called here.
        r = float(state[1])
        return np.array([r, self.mass, self.radius, self.r_s], dtype=float)

    def cartesian_state_to_spherical(self, state):
        """
        (t, x, y, z, dt, dx, dy, dz) -> (t, r, theta, phi, dt, dr, dtheta, dphi)
        """
        t, x, y, z, dtdl, dxdl, dydl, dzdl = state

        X = x - self.center[0]
        Y = y - self.center[1]
        Z = z - self.center[2]

        r, theta, phi = cartesian_to_spherical(X, Y, Z)
        drdl, dthetadl, dphidl = cartesian_velocity_to_spherical(
            X, Y, Z, dxdl, dydl, dzdl
        )

        return np.array([t, r, theta, phi, dtdl, drdl, dthetadl, dphidl], dtype=float)
