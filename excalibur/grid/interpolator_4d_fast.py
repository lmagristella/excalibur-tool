# excalibur/grid/interpolator_4d_fast.py
"""
Fast grid interpolator with pluggable interpolation schemes.

Supported schemes
-----------------
  "trilinear"  : 2-point stencil per axis, C0 continuous, analytic gradient,
                 cross-Hessian only (diagonal Hessian = 0).
  "tricubic"   : 4-point Catmull-Rom stencil per axis, C1 continuous,
                 analytic gradient, full Hessian (including diagonal).

Dimensions
----------
  3D grids: field(x, y, z)      -> 3-gradient, 3x3 Hessian
  4D grids: field(x, y, z, t)   -> 4-gradient (gx,gy,gz,gt), 4x4 Hessian

Supported boundary modes: "error", "clamp", "periodic".

All hot-path functions are Numba-compiled (@njit).

Performance
-----------
  Tensor-product kernels use *separable 1D contractions* (no nested loops).
  3D tricubic: 3 passes over 4 nodes  (vs 64-iteration triple loop).
  4D tricubic: 4 passes over 4 nodes  (vs 256-iteration quadruple loop).
"""
from __future__ import annotations

import numpy as np
from numba import njit


# ================================================================
#  UTILITY HELPERS
# ================================================================

@njit(cache=True, fastmath=True)
def _wrap_index(i, n):
    """Modulo that works for negative indices."""
    return i % n


@njit(cache=True, fastmath=True)
def _wrap_pos(p, origin, L):
    """Map *p* into [origin, origin + L)."""
    return origin + ((p - origin) % L)


@njit(cache=True, fastmath=True)
def _clamp_pos(p, lo, hi):
    if p < lo:
        return lo
    if p > hi:
        return hi
    return p


# ================================================================
#  1D CATMULL-ROM WEIGHTS
# ================================================================

@njit(cache=True, fastmath=True)
def _cr_weights(t):
    t2 = t * t; t3 = t2 * t
    w0 = -0.5 * t + t2 - 0.5 * t3
    w1 = 1.0 - 2.5 * t2 + 1.5 * t3
    w2 = 0.5 * t + 2.0 * t2 - 1.5 * t3
    w3 = -0.5 * t2 + 0.5 * t3
    return w0, w1, w2, w3

@njit(cache=True, fastmath=True)
def _cr_dweights(t):
    t2 = t * t
    d0 = -0.5 + 2.0 * t - 1.5 * t2
    d1 = -5.0 * t + 4.5 * t2
    d2 = 0.5 + 4.0 * t - 4.5 * t2
    d3 = -t + 1.5 * t2
    return d0, d1, d2, d3

@njit(cache=True, fastmath=True)
def _cr_d2weights(t):
    d0 = 2.0 - 3.0 * t
    d1 = -5.0 + 9.0 * t
    d2 = 4.0 - 9.0 * t
    d3 = -1.0 + 3.0 * t
    return d0, d1, d2, d3

@njit(cache=True, fastmath=True)
def _lin_weights(t):
    return 1.0 - t, t

@njit(cache=True, fastmath=True)
def _lin_dweights(t):
    return -1.0, 1.0

@njit(cache=True, fastmath=True)
def _lin_d2weights(t):
    return 0.0, 0.0


# ================================================================
#  TRILINEAR KERNEL  (2x2x2 stencil, 3D)  — hand-unrolled
# ================================================================

@njit(cache=True, fastmath=True)
def _trilinear_val_grad_hess(
    c000, c100, c010, c001,
    c110, c101, c011, c111,
    dx, dy, dz,
    inv_sx, inv_sy, inv_sz,
):
    """
    Returns (val, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz).
    For trilinear: hxx = hyy = hzz = 0.
    """
    mx = 1.0 - dx; my = 1.0 - dy; mz = 1.0 - dz

    val = (c000*mx*my*mz + c100*dx*my*mz + c010*mx*dy*mz + c001*mx*my*dz
         + c110*dx*dy*mz + c101*dx*my*dz + c011*mx*dy*dz + c111*dx*dy*dz)

    dfdx = ((c100-c000)*my*mz + (c110-c010)*dy*mz
          + (c101-c001)*my*dz + (c111-c011)*dy*dz)
    dfdy = ((c010-c000)*mx*mz + (c110-c100)*dx*mz
          + (c011-c001)*mx*dz + (c111-c101)*dx*dz)
    dfdz = ((c001-c000)*mx*my + (c101-c100)*dx*my
          + (c011-c010)*mx*dy + (c111-c110)*dx*dy)

    gx = dfdx * inv_sx; gy = dfdy * inv_sy; gz = dfdz * inv_sz

    d2fdxdy = (c110-c010-c100+c000)*mz + (c111-c011-c101+c001)*dz
    d2fdxdz = (c101-c001-c100+c000)*my + (c111-c011-c110+c010)*dy
    d2fdydz = (c011-c001-c010+c000)*mx + (c111-c101-c110+c100)*dx

    return (val, gx, gy, gz,
            0.0, 0.0, 0.0,
            d2fdxdy*inv_sx*inv_sy, d2fdxdz*inv_sx*inv_sz, d2fdydz*inv_sy*inv_sz)


# ================================================================
#  TRICUBIC KERNEL  (4x4x4, 3D)  — SEPARABLE (no triple loop)
#
#  Strategy: contract axis-by-axis.
#  For value + gradient + Hessian we need to track 3 "channels"
#  per already-contracted axis:  W (weight), D (1st deriv), D2 (2nd deriv).
#
#  Pass 1 — contract z:  stencil(4,4,4) → buf_z(4,4)
#    For each (a,b): sum over g of corners[a,b,g] * { wz, dwz, d2wz }
#    -> 3 channels:  V, dZ, d2Z   (4x4 each)
#
#  Pass 2 — contract y:  buf_z(4,4) → buf_y(4)
#    For each a: sum over b of buf_z[a,b] * { wy, dwy, d2wy }
#    Combinatorics: V*wy, V*dwy, V*d2wy, dZ*wy, dZ*dwy, d2Z*wy
#    -> 6 channels: V, dY, d2Y, dZ, dYdZ, d2Z
#
#  Pass 3 — contract x:  buf_y(4) → scalars
#    For each a: sum over a of buf_y[a] * { wx, dwx, d2wx }
#    -> 10 outputs: val, gx,gy,gz, hxx,hyy,hzz, hxy,hxz,hyz
# ================================================================

@njit(cache=True, fastmath=True)
def _tricubic_val_grad_hess(
    corners,
    dx, dy, dz,
    inv_sx, inv_sy, inv_sz,
):
    """
    Catmull-Rom tricubic interpolant — separable implementation.
    corners[a, b, c]: field at stencil offset (a-1, b-1, c-1).
    Returns (val, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz).
    """
    wz0, wz1, wz2, wz3 = _cr_weights(dz)
    dz0, dz1, dz2, dz3 = _cr_dweights(dz)
    ddz0, ddz1, ddz2, ddz3 = _cr_d2weights(dz)
    wz = (wz0, wz1, wz2, wz3)
    dzw = (dz0, dz1, dz2, dz3)
    d2zw = (ddz0, ddz1, ddz2, ddz3)

    wy0, wy1, wy2, wy3 = _cr_weights(dy)
    dy0, dy1, dy2, dy3 = _cr_dweights(dy)
    ddy0, ddy1, ddy2, ddy3 = _cr_d2weights(dy)
    wy = (wy0, wy1, wy2, wy3)
    dyw = (dy0, dy1, dy2, dy3)
    d2yw = (ddy0, ddy1, ddy2, ddy3)

    wx0, wx1, wx2, wx3 = _cr_weights(dx)
    dx0, dx1, dx2, dx3 = _cr_dweights(dx)
    ddx0, ddx1, ddx2, ddx3 = _cr_d2weights(dx)
    wxw = (wx0, wx1, wx2, wx3)
    dxw = (dx0, dx1, dx2, dx3)
    d2xw = (ddx0, ddx1, ddx2, ddx3)

    # -- Pass 1: contract z --  (4,4,4) -> 3 channels of (4,4)
    # V[a,b]   = sum_g corners[a,b,g]*wz[g]
    # dZ[a,b]  = sum_g corners[a,b,g]*dz[g]
    # d2Z[a,b] = sum_g corners[a,b,g]*d2z[g]
    V00 = corners[0,0,0]*wz[0] + corners[0,0,1]*wz[1] + corners[0,0,2]*wz[2] + corners[0,0,3]*wz[3]
    V01 = corners[0,1,0]*wz[0] + corners[0,1,1]*wz[1] + corners[0,1,2]*wz[2] + corners[0,1,3]*wz[3]
    V02 = corners[0,2,0]*wz[0] + corners[0,2,1]*wz[1] + corners[0,2,2]*wz[2] + corners[0,2,3]*wz[3]
    V03 = corners[0,3,0]*wz[0] + corners[0,3,1]*wz[1] + corners[0,3,2]*wz[2] + corners[0,3,3]*wz[3]
    V10 = corners[1,0,0]*wz[0] + corners[1,0,1]*wz[1] + corners[1,0,2]*wz[2] + corners[1,0,3]*wz[3]
    V11 = corners[1,1,0]*wz[0] + corners[1,1,1]*wz[1] + corners[1,1,2]*wz[2] + corners[1,1,3]*wz[3]
    V12 = corners[1,2,0]*wz[0] + corners[1,2,1]*wz[1] + corners[1,2,2]*wz[2] + corners[1,2,3]*wz[3]
    V13 = corners[1,3,0]*wz[0] + corners[1,3,1]*wz[1] + corners[1,3,2]*wz[2] + corners[1,3,3]*wz[3]
    V20 = corners[2,0,0]*wz[0] + corners[2,0,1]*wz[1] + corners[2,0,2]*wz[2] + corners[2,0,3]*wz[3]
    V21 = corners[2,1,0]*wz[0] + corners[2,1,1]*wz[1] + corners[2,1,2]*wz[2] + corners[2,1,3]*wz[3]
    V22 = corners[2,2,0]*wz[0] + corners[2,2,1]*wz[1] + corners[2,2,2]*wz[2] + corners[2,2,3]*wz[3]
    V23 = corners[2,3,0]*wz[0] + corners[2,3,1]*wz[1] + corners[2,3,2]*wz[2] + corners[2,3,3]*wz[3]
    V30 = corners[3,0,0]*wz[0] + corners[3,0,1]*wz[1] + corners[3,0,2]*wz[2] + corners[3,0,3]*wz[3]
    V31 = corners[3,1,0]*wz[0] + corners[3,1,1]*wz[1] + corners[3,1,2]*wz[2] + corners[3,1,3]*wz[3]
    V32 = corners[3,2,0]*wz[0] + corners[3,2,1]*wz[1] + corners[3,2,2]*wz[2] + corners[3,2,3]*wz[3]
    V33 = corners[3,3,0]*wz[0] + corners[3,3,1]*wz[1] + corners[3,3,2]*wz[2] + corners[3,3,3]*wz[3]

    Z00 = corners[0,0,0]*dzw[0] + corners[0,0,1]*dzw[1] + corners[0,0,2]*dzw[2] + corners[0,0,3]*dzw[3]
    Z01 = corners[0,1,0]*dzw[0] + corners[0,1,1]*dzw[1] + corners[0,1,2]*dzw[2] + corners[0,1,3]*dzw[3]
    Z02 = corners[0,2,0]*dzw[0] + corners[0,2,1]*dzw[1] + corners[0,2,2]*dzw[2] + corners[0,2,3]*dzw[3]
    Z03 = corners[0,3,0]*dzw[0] + corners[0,3,1]*dzw[1] + corners[0,3,2]*dzw[2] + corners[0,3,3]*dzw[3]
    Z10 = corners[1,0,0]*dzw[0] + corners[1,0,1]*dzw[1] + corners[1,0,2]*dzw[2] + corners[1,0,3]*dzw[3]
    Z11 = corners[1,1,0]*dzw[0] + corners[1,1,1]*dzw[1] + corners[1,1,2]*dzw[2] + corners[1,1,3]*dzw[3]
    Z12 = corners[1,2,0]*dzw[0] + corners[1,2,1]*dzw[1] + corners[1,2,2]*dzw[2] + corners[1,2,3]*dzw[3]
    Z13 = corners[1,3,0]*dzw[0] + corners[1,3,1]*dzw[1] + corners[1,3,2]*dzw[2] + corners[1,3,3]*dzw[3]
    Z20 = corners[2,0,0]*dzw[0] + corners[2,0,1]*dzw[1] + corners[2,0,2]*dzw[2] + corners[2,0,3]*dzw[3]
    Z21 = corners[2,1,0]*dzw[0] + corners[2,1,1]*dzw[1] + corners[2,1,2]*dzw[2] + corners[2,1,3]*dzw[3]
    Z22 = corners[2,2,0]*dzw[0] + corners[2,2,1]*dzw[1] + corners[2,2,2]*dzw[2] + corners[2,2,3]*dzw[3]
    Z23 = corners[2,3,0]*dzw[0] + corners[2,3,1]*dzw[1] + corners[2,3,2]*dzw[2] + corners[2,3,3]*dzw[3]
    Z30 = corners[3,0,0]*dzw[0] + corners[3,0,1]*dzw[1] + corners[3,0,2]*dzw[2] + corners[3,0,3]*dzw[3]
    Z31 = corners[3,1,0]*dzw[0] + corners[3,1,1]*dzw[1] + corners[3,1,2]*dzw[2] + corners[3,1,3]*dzw[3]
    Z32 = corners[3,2,0]*dzw[0] + corners[3,2,1]*dzw[1] + corners[3,2,2]*dzw[2] + corners[3,2,3]*dzw[3]
    Z33 = corners[3,3,0]*dzw[0] + corners[3,3,1]*dzw[1] + corners[3,3,2]*dzw[2] + corners[3,3,3]*dzw[3]

    ZZ00 = corners[0,0,0]*d2zw[0] + corners[0,0,1]*d2zw[1] + corners[0,0,2]*d2zw[2] + corners[0,0,3]*d2zw[3]
    ZZ01 = corners[0,1,0]*d2zw[0] + corners[0,1,1]*d2zw[1] + corners[0,1,2]*d2zw[2] + corners[0,1,3]*d2zw[3]
    ZZ02 = corners[0,2,0]*d2zw[0] + corners[0,2,1]*d2zw[1] + corners[0,2,2]*d2zw[2] + corners[0,2,3]*d2zw[3]
    ZZ03 = corners[0,3,0]*d2zw[0] + corners[0,3,1]*d2zw[1] + corners[0,3,2]*d2zw[2] + corners[0,3,3]*d2zw[3]
    ZZ10 = corners[1,0,0]*d2zw[0] + corners[1,0,1]*d2zw[1] + corners[1,0,2]*d2zw[2] + corners[1,0,3]*d2zw[3]
    ZZ11 = corners[1,1,0]*d2zw[0] + corners[1,1,1]*d2zw[1] + corners[1,1,2]*d2zw[2] + corners[1,1,3]*d2zw[3]
    ZZ12 = corners[1,2,0]*d2zw[0] + corners[1,2,1]*d2zw[1] + corners[1,2,2]*d2zw[2] + corners[1,2,3]*d2zw[3]
    ZZ13 = corners[1,3,0]*d2zw[0] + corners[1,3,1]*d2zw[1] + corners[1,3,2]*d2zw[2] + corners[1,3,3]*d2zw[3]
    ZZ20 = corners[2,0,0]*d2zw[0] + corners[2,0,1]*d2zw[1] + corners[2,0,2]*d2zw[2] + corners[2,0,3]*d2zw[3]
    ZZ21 = corners[2,1,0]*d2zw[0] + corners[2,1,1]*d2zw[1] + corners[2,1,2]*d2zw[2] + corners[2,1,3]*d2zw[3]
    ZZ22 = corners[2,2,0]*d2zw[0] + corners[2,2,1]*d2zw[1] + corners[2,2,2]*d2zw[2] + corners[2,2,3]*d2zw[3]
    ZZ23 = corners[2,3,0]*d2zw[0] + corners[2,3,1]*d2zw[1] + corners[2,3,2]*d2zw[2] + corners[2,3,3]*d2zw[3]
    ZZ30 = corners[3,0,0]*d2zw[0] + corners[3,0,1]*d2zw[1] + corners[3,0,2]*d2zw[2] + corners[3,0,3]*d2zw[3]
    ZZ31 = corners[3,1,0]*d2zw[0] + corners[3,1,1]*d2zw[1] + corners[3,1,2]*d2zw[2] + corners[3,1,3]*d2zw[3]
    ZZ32 = corners[3,2,0]*d2zw[0] + corners[3,2,1]*d2zw[1] + corners[3,2,2]*d2zw[2] + corners[3,2,3]*d2zw[3]
    ZZ33 = corners[3,3,0]*d2zw[0] + corners[3,3,1]*d2zw[1] + corners[3,3,2]*d2zw[2] + corners[3,3,3]*d2zw[3]

    # -- Pass 2: contract y --  (4,4) -> 6 channels of (4,)
    # channels after y-contraction:
    #   V_a   = sum_b V[a,b]*wy[b]           (value)
    #   Y_a   = sum_b V[a,b]*dyw[b]          (dY)
    #   YY_a  = sum_b V[a,b]*d2yw[b]         (d2Y)
    #   Zv_a  = sum_b Z[a,b]*wy[b]           (dZ)
    #   YZ_a  = sum_b Z[a,b]*dyw[b]          (dYdZ)
    #   ZZv_a = sum_b ZZ[a,b]*wy[b]          (d2Z)
    Va0  = V00*wy[0]  + V01*wy[1]  + V02*wy[2]  + V03*wy[3]
    Va1  = V10*wy[0]  + V11*wy[1]  + V12*wy[2]  + V13*wy[3]
    Va2  = V20*wy[0]  + V21*wy[1]  + V22*wy[2]  + V23*wy[3]
    Va3  = V30*wy[0]  + V31*wy[1]  + V32*wy[2]  + V33*wy[3]

    Ya0  = V00*dyw[0] + V01*dyw[1] + V02*dyw[2] + V03*dyw[3]
    Ya1  = V10*dyw[0] + V11*dyw[1] + V12*dyw[2] + V13*dyw[3]
    Ya2  = V20*dyw[0] + V21*dyw[1] + V22*dyw[2] + V23*dyw[3]
    Ya3  = V30*dyw[0] + V31*dyw[1] + V32*dyw[2] + V33*dyw[3]

    YYa0 = V00*d2yw[0]+ V01*d2yw[1]+ V02*d2yw[2]+ V03*d2yw[3]
    YYa1 = V10*d2yw[0]+ V11*d2yw[1]+ V12*d2yw[2]+ V13*d2yw[3]
    YYa2 = V20*d2yw[0]+ V21*d2yw[1]+ V22*d2yw[2]+ V23*d2yw[3]
    YYa3 = V30*d2yw[0]+ V31*d2yw[1]+ V32*d2yw[2]+ V33*d2yw[3]

    Zva0 = Z00*wy[0]  + Z01*wy[1]  + Z02*wy[2]  + Z03*wy[3]
    Zva1 = Z10*wy[0]  + Z11*wy[1]  + Z12*wy[2]  + Z13*wy[3]
    Zva2 = Z20*wy[0]  + Z21*wy[1]  + Z22*wy[2]  + Z23*wy[3]
    Zva3 = Z30*wy[0]  + Z31*wy[1]  + Z32*wy[2]  + Z33*wy[3]

    YZa0 = Z00*dyw[0] + Z01*dyw[1] + Z02*dyw[2] + Z03*dyw[3]
    YZa1 = Z10*dyw[0] + Z11*dyw[1] + Z12*dyw[2] + Z13*dyw[3]
    YZa2 = Z20*dyw[0] + Z21*dyw[1] + Z22*dyw[2] + Z23*dyw[3]
    YZa3 = Z30*dyw[0] + Z31*dyw[1] + Z32*dyw[2] + Z33*dyw[3]

    ZZa0 = ZZ00*wy[0] + ZZ01*wy[1] + ZZ02*wy[2] + ZZ03*wy[3]
    ZZa1 = ZZ10*wy[0] + ZZ11*wy[1] + ZZ12*wy[2] + ZZ13*wy[3]
    ZZa2 = ZZ20*wy[0] + ZZ21*wy[1] + ZZ22*wy[2] + ZZ23*wy[3]
    ZZa3 = ZZ30*wy[0] + ZZ31*wy[1] + ZZ32*wy[2] + ZZ33*wy[3]

    # -- Pass 3: contract x --  (4,) -> 10 scalars
    val_  = Va0*wxw[0]  + Va1*wxw[1]  + Va2*wxw[2]  + Va3*wxw[3]
    gx_   = Va0*dxw[0]  + Va1*dxw[1]  + Va2*dxw[2]  + Va3*dxw[3]
    d2x_  = Va0*d2xw[0] + Va1*d2xw[1] + Va2*d2xw[2] + Va3*d2xw[3]

    gy_   = Ya0*wxw[0]  + Ya1*wxw[1]  + Ya2*wxw[2]  + Ya3*wxw[3]
    d2y_  = YYa0*wxw[0] + YYa1*wxw[1] + YYa2*wxw[2] + YYa3*wxw[3]

    gz_   = Zva0*wxw[0]  + Zva1*wxw[1]  + Zva2*wxw[2]  + Zva3*wxw[3]
    d2z_  = ZZa0*wxw[0]  + ZZa1*wxw[1]  + ZZa2*wxw[2]  + ZZa3*wxw[3]

    hxy_  = Ya0*dxw[0]  + Ya1*dxw[1]  + Ya2*dxw[2]  + Ya3*dxw[3]
    hxz_  = Zva0*dxw[0] + Zva1*dxw[1] + Zva2*dxw[2] + Zva3*dxw[3]
    hyz_  = YZa0*wxw[0] + YZa1*wxw[1] + YZa2*wxw[2] + YZa3*wxw[3]

    gx = gx_ * inv_sx; gy = gy_ * inv_sy; gz = gz_ * inv_sz
    hxx = d2x_ * inv_sx * inv_sx
    hyy = d2y_ * inv_sy * inv_sy
    hzz = d2z_ * inv_sz * inv_sz
    hxy = hxy_ * inv_sx * inv_sy
    hxz = hxz_ * inv_sx * inv_sz
    hyz = hyz_ * inv_sy * inv_sz

    return val_, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz


# ================================================================
#  4D TRILINEAR KERNEL (2^4=16) — SEPARABLE (no quadruple loop)
#
#  For trilinear, d2w ≡ 0, so diagonal Hessians vanish.
#  We still need cross-derivatives (6 off-diagonal terms).
#
#  4 passes: t -> z -> y -> x
# ================================================================

@njit(cache=True, fastmath=True)
def _interp_4d_linear_kernel(
    stencil,
    dx, dy, dz, dt,
    inv_sx, inv_sy, inv_sz, inv_st,
):
    """
    4D trilinear (2x2x2x2) interpolation — separable.
    Returns 15 values:
      val, gx, gy, gz, gt,
      hxx, hyy, hzz, htt,
      hxy, hxz, hxt, hyz, hyt, hzt
    """
    wt0, wt1 = _lin_weights(dt)
    dt0, dt1 = _lin_dweights(dt)

    # Pass 1: contract t — (2,2,2,2) -> 2 channels of (2,2,2)
    # V[a,b,g] = S[a,b,g,0]*wt0 + S[a,b,g,1]*wt1
    # T[a,b,g] = S[a,b,g,0]*dt0 + S[a,b,g,1]*dt1
    V000 = stencil[0,0,0,0]*wt0 + stencil[0,0,0,1]*wt1
    V001 = stencil[0,0,1,0]*wt0 + stencil[0,0,1,1]*wt1
    V010 = stencil[0,1,0,0]*wt0 + stencil[0,1,0,1]*wt1
    V011 = stencil[0,1,1,0]*wt0 + stencil[0,1,1,1]*wt1
    V100 = stencil[1,0,0,0]*wt0 + stencil[1,0,0,1]*wt1
    V101 = stencil[1,0,1,0]*wt0 + stencil[1,0,1,1]*wt1
    V110 = stencil[1,1,0,0]*wt0 + stencil[1,1,0,1]*wt1
    V111 = stencil[1,1,1,0]*wt0 + stencil[1,1,1,1]*wt1

    T000 = stencil[0,0,0,0]*dt0 + stencil[0,0,0,1]*dt1
    T001 = stencil[0,0,1,0]*dt0 + stencil[0,0,1,1]*dt1
    T010 = stencil[0,1,0,0]*dt0 + stencil[0,1,0,1]*dt1
    T011 = stencil[0,1,1,0]*dt0 + stencil[0,1,1,1]*dt1
    T100 = stencil[1,0,0,0]*dt0 + stencil[1,0,0,1]*dt1
    T101 = stencil[1,0,1,0]*dt0 + stencil[1,0,1,1]*dt1
    T110 = stencil[1,1,0,0]*dt0 + stencil[1,1,0,1]*dt1
    T111 = stencil[1,1,1,0]*dt0 + stencil[1,1,1,1]*dt1

    # Now we have a 3D trilinear in V for val/grad_xyz/hess_xyz,
    # and T for gt/hxt/hyt/hzt.
    # Use the hand-unrolled trilinear for V:
    mx = 1.0 - dx; my = 1.0 - dy; mz = 1.0 - dz

    val = (V000*mx*my*mz + V100*dx*my*mz + V010*mx*dy*mz + V001*mx*my*dz
         + V110*dx*dy*mz + V101*dx*my*dz + V011*mx*dy*dz + V111*dx*dy*dz)

    dfdx = ((V100-V000)*my*mz + (V110-V010)*dy*mz
          + (V101-V001)*my*dz + (V111-V011)*dy*dz)
    dfdy = ((V010-V000)*mx*mz + (V110-V100)*dx*mz
          + (V011-V001)*mx*dz + (V111-V101)*dx*dz)
    dfdz = ((V001-V000)*mx*my + (V101-V100)*dx*my
          + (V011-V010)*mx*dy + (V111-V110)*dx*dy)

    # gt from T stencil (same trilinear interpolation of T values)
    gt_ = (T000*mx*my*mz + T100*dx*my*mz + T010*mx*dy*mz + T001*mx*my*dz
         + T110*dx*dy*mz + T101*dx*my*dz + T011*mx*dy*dz + T111*dx*dy*dz)

    # cross Hessian xy, xz, yz from V
    d2xy = (V110-V010-V100+V000)*mz + (V111-V011-V101+V001)*dz
    d2xz = (V101-V001-V100+V000)*my + (V111-V011-V110+V010)*dy
    d2yz = (V011-V001-V010+V000)*mx + (V111-V101-V110+V100)*dx

    # cross Hessian xt, yt, zt from T
    dTdx = ((T100-T000)*my*mz + (T110-T010)*dy*mz
          + (T101-T001)*my*dz + (T111-T011)*dy*dz)
    dTdy = ((T010-T000)*mx*mz + (T110-T100)*dx*mz
          + (T011-T001)*mx*dz + (T111-T101)*dx*dz)
    dTdz = ((T001-T000)*mx*my + (T101-T100)*dx*my
          + (T011-T010)*mx*dy + (T111-T110)*dx*dy)

    gx = dfdx * inv_sx; gy = dfdy * inv_sy; gz = dfdz * inv_sz
    gt = gt_ * inv_st

    return (val, gx, gy, gz, gt,
            0.0, 0.0, 0.0, 0.0,                          # hxx, hyy, hzz, htt (all 0 for linear)
            d2xy*inv_sx*inv_sy, d2xz*inv_sx*inv_sz,
            dTdx*inv_sx*inv_st,
            d2yz*inv_sy*inv_sz,
            dTdy*inv_sy*inv_st, dTdz*inv_sz*inv_st)


# ================================================================
#  4D TRICUBIC (CATMULL-ROM) KERNEL — SEPARABLE
#
#  4 passes: t -> z -> y -> x
#
#  Pass 1 (contract t): stencil(4,4,4,4) -> 3 channels of (4,4,4)
#    V, dT, d2T
#
#  Pass 2 (contract z): (4,4,4) -> 6 channels of (4,4)
#    V, dZ, d2Z   (from V-channel)
#    T, TZ        (from dT-channel: dT*wz, dT*dz — need T alone + T*dz for hzt)
#    TT           (from d2T-channel: d2T*wz — just value)
#
#  Actually, for the 15 outputs we need these channel combinations
#  accumulated through the passes.  Let me enumerate what we need
#  at the final x-pass:
#
#  val = Wx · V_a
#  gx  = dWx · V_a        gy = Wx · Y_a          gz = Wx · Z_a          gt = Wx · T_a
#  hxx = d2Wx · V_a        hyy = Wx · YY_a        hzz = Wx · ZZ_a       htt = Wx · TT_a
#  hxy = dWx · Y_a         hxz = dWx · Z_a        hxt = dWx · T_a
#  hyz = Wx · YZ_a         hyt = Wx · YT_a        hzt = Wx · ZT_a
#
#  So we need 10 channels at the x-pass level:
#    V, Y, YY, Z, ZZ, YZ, T, TT, YT, ZT
#
#  Building these bottom-up through the passes is straightforward.
# ================================================================

@njit(cache=True, fastmath=True)
def _interp_4d_cubic_kernel(
    stencil,
    dx, dy, dz, dt,
    inv_sx, inv_sy, inv_sz, inv_st,
):
    """
    4D Catmull-Rom (4^4) interpolation — separable.
    Returns 15 values: val, gx, gy, gz, gt, hxx, hyy, hzz, htt,
                       hxy, hxz, hxt, hyz, hyt, hzt.
    """
    # --- weights ---
    Wt = _cr_weights(dt);   Dt = _cr_dweights(dt);   D2t = _cr_d2weights(dt)
    Wz = _cr_weights(dz);   Dz = _cr_dweights(dz);   D2z = _cr_d2weights(dz)
    Wy = _cr_weights(dy);   Dy = _cr_dweights(dy);   D2y = _cr_d2weights(dy)
    Wx = _cr_weights(dx);   DDx = _cr_dweights(dx);  D2x = _cr_d2weights(dx)

    # ---- Pass 1: contract t ----
    #   V [a,b,g] = sum_h stencil[a,b,g,h] * Wt[h]
    #   Tv[a,b,g] = sum_h stencil[a,b,g,h] * Dt[h]
    #   TT[a,b,g] = sum_h stencil[a,b,g,h] * D2t[h]
    #
    # We store these in flat arrays (64 each).  Using a single loop over
    # the 64 spatial indices.
    V  = np.empty(64)
    Tv = np.empty(64)
    TT = np.empty(64)
    idx = 0
    for a in range(4):
        for b in range(4):
            for g in range(4):
                s0 = stencil[a, b, g, 0]
                s1 = stencil[a, b, g, 1]
                s2 = stencil[a, b, g, 2]
                s3 = stencil[a, b, g, 3]
                V[idx]  = s0*Wt[0]  + s1*Wt[1]  + s2*Wt[2]  + s3*Wt[3]
                Tv[idx] = s0*Dt[0]  + s1*Dt[1]  + s2*Dt[2]  + s3*Dt[3]
                TT[idx] = s0*D2t[0] + s1*D2t[1] + s2*D2t[2] + s3*D2t[3]
                idx += 1

    # ---- Pass 2: contract z ----
    #   From V:   Vyz[a,b] = sum_g V[a,b,g]*Wz[g]
    #             Zv[a,b]  = sum_g V[a,b,g]*Dz[g]
    #             ZZ[a,b]  = sum_g V[a,b,g]*D2z[g]
    #   From Tv:  Tyz[a,b] = sum_g Tv[a,b,g]*Wz[g]
    #             TZv[a,b] = sum_g Tv[a,b,g]*Dz[g]   (for hzt)
    #   From TT:  TTyz[a,b]= sum_g TT[a,b,g]*Wz[g]
    Vyz  = np.empty(16)
    Zv   = np.empty(16)
    ZZv  = np.empty(16)
    Tyz  = np.empty(16)
    TZv  = np.empty(16)
    TTyz = np.empty(16)
    idx2 = 0
    for a in range(4):
        for b in range(4):
            base = a * 16 + b * 4
            v0 = V[base]; v1 = V[base+1]; v2 = V[base+2]; v3 = V[base+3]
            Vyz[idx2]  = v0*Wz[0]  + v1*Wz[1]  + v2*Wz[2]  + v3*Wz[3]
            Zv[idx2]   = v0*Dz[0]  + v1*Dz[1]  + v2*Dz[2]  + v3*Dz[3]
            ZZv[idx2]  = v0*D2z[0] + v1*D2z[1] + v2*D2z[2] + v3*D2z[3]
            t0 = Tv[base]; t1 = Tv[base+1]; t2 = Tv[base+2]; t3 = Tv[base+3]
            Tyz[idx2]  = t0*Wz[0]  + t1*Wz[1]  + t2*Wz[2]  + t3*Wz[3]
            TZv[idx2]  = t0*Dz[0]  + t1*Dz[1]  + t2*Dz[2]  + t3*Dz[3]
            tt0 = TT[base]; tt1 = TT[base+1]; tt2 = TT[base+2]; tt3 = TT[base+3]
            TTyz[idx2] = tt0*Wz[0] + tt1*Wz[1] + tt2*Wz[2] + tt3*Wz[3]
            idx2 += 1

    # ---- Pass 3: contract y ----
    # From Vyz:  Va[a]  = sum_b Vyz[a,b]*Wy[b]
    #            Ya[a]  = sum_b Vyz[a,b]*Dy[b]
    #            YYa[a] = sum_b Vyz[a,b]*D2y[b]
    # From Zv:   Za[a]  = sum_b Zv[a,b]*Wy[b]
    #            YZa[a] = sum_b Zv[a,b]*Dy[b]
    # From ZZv:  ZZa[a] = sum_b ZZv[a,b]*Wy[b]
    # From Tyz:  Ta[a]  = sum_b Tyz[a,b]*Wy[b]
    #            YTa[a] = sum_b Tyz[a,b]*Dy[b]
    # From TZv:  ZTa[a] = sum_b TZv[a,b]*Wy[b]
    # From TTyz: TTa[a] = sum_b TTyz[a,b]*Wy[b]
    Va  = np.empty(4); Ya = np.empty(4);  YYa = np.empty(4)
    Za  = np.empty(4); YZa = np.empty(4); ZZa = np.empty(4)
    Ta  = np.empty(4); YTa = np.empty(4); ZTa = np.empty(4)
    TTa = np.empty(4)
    for a in range(4):
        base = a * 4
        v0 = Vyz[base]; v1 = Vyz[base+1]; v2 = Vyz[base+2]; v3 = Vyz[base+3]
        Va[a]  = v0*Wy[0]  + v1*Wy[1]  + v2*Wy[2]  + v3*Wy[3]
        Ya[a]  = v0*Dy[0]  + v1*Dy[1]  + v2*Dy[2]  + v3*Dy[3]
        YYa[a] = v0*D2y[0] + v1*D2y[1] + v2*D2y[2] + v3*D2y[3]

        z0 = Zv[base]; z1 = Zv[base+1]; z2 = Zv[base+2]; z3 = Zv[base+3]
        Za[a]  = z0*Wy[0]  + z1*Wy[1]  + z2*Wy[2]  + z3*Wy[3]
        YZa[a] = z0*Dy[0]  + z1*Dy[1]  + z2*Dy[2]  + z3*Dy[3]

        zz0 = ZZv[base]; zz1 = ZZv[base+1]; zz2 = ZZv[base+2]; zz3 = ZZv[base+3]
        ZZa[a] = zz0*Wy[0] + zz1*Wy[1] + zz2*Wy[2] + zz3*Wy[3]

        t0 = Tyz[base]; t1 = Tyz[base+1]; t2 = Tyz[base+2]; t3 = Tyz[base+3]
        Ta[a]  = t0*Wy[0]  + t1*Wy[1]  + t2*Wy[2]  + t3*Wy[3]
        YTa[a] = t0*Dy[0]  + t1*Dy[1]  + t2*Dy[2]  + t3*Dy[3]

        zt0 = TZv[base]; zt1 = TZv[base+1]; zt2 = TZv[base+2]; zt3 = TZv[base+3]
        ZTa[a] = zt0*Wy[0] + zt1*Wy[1] + zt2*Wy[2] + zt3*Wy[3]

        tt0 = TTyz[base]; tt1 = TTyz[base+1]; tt2 = TTyz[base+2]; tt3 = TTyz[base+3]
        TTa[a] = tt0*Wy[0] + tt1*Wy[1] + tt2*Wy[2] + tt3*Wy[3]

    # ---- Pass 4: contract x ----  (4,) -> 15 scalars
    val_ = Va[0]*Wx[0]  + Va[1]*Wx[1]  + Va[2]*Wx[2]  + Va[3]*Wx[3]
    gx_  = Va[0]*DDx[0] + Va[1]*DDx[1] + Va[2]*DDx[2] + Va[3]*DDx[3]
    hxx_ = Va[0]*D2x[0] + Va[1]*D2x[1] + Va[2]*D2x[2] + Va[3]*D2x[3]

    gy_  = Ya[0]*Wx[0]  + Ya[1]*Wx[1]  + Ya[2]*Wx[2]  + Ya[3]*Wx[3]
    hyy_ = YYa[0]*Wx[0] + YYa[1]*Wx[1] + YYa[2]*Wx[2] + YYa[3]*Wx[3]
    hxy_ = Ya[0]*DDx[0] + Ya[1]*DDx[1] + Ya[2]*DDx[2] + Ya[3]*DDx[3]

    gz_  = Za[0]*Wx[0]  + Za[1]*Wx[1]  + Za[2]*Wx[2]  + Za[3]*Wx[3]
    hzz_ = ZZa[0]*Wx[0] + ZZa[1]*Wx[1] + ZZa[2]*Wx[2] + ZZa[3]*Wx[3]
    hxz_ = Za[0]*DDx[0] + Za[1]*DDx[1] + Za[2]*DDx[2] + Za[3]*DDx[3]
    hyz_ = YZa[0]*Wx[0] + YZa[1]*Wx[1] + YZa[2]*Wx[2] + YZa[3]*Wx[3]

    gt_  = Ta[0]*Wx[0]  + Ta[1]*Wx[1]  + Ta[2]*Wx[2]  + Ta[3]*Wx[3]
    htt_ = TTa[0]*Wx[0] + TTa[1]*Wx[1] + TTa[2]*Wx[2] + TTa[3]*Wx[3]
    hxt_ = Ta[0]*DDx[0] + Ta[1]*DDx[1] + Ta[2]*DDx[2] + Ta[3]*DDx[3]
    hyt_ = YTa[0]*Wx[0] + YTa[1]*Wx[1] + YTa[2]*Wx[2] + YTa[3]*Wx[3]
    hzt_ = ZTa[0]*Wx[0] + ZTa[1]*Wx[1] + ZTa[2]*Wx[2] + ZTa[3]*Wx[3]

    gx = gx_ * inv_sx; gy = gy_ * inv_sy
    gz = gz_ * inv_sz; gt = gt_ * inv_st
    hxx = hxx_ * inv_sx * inv_sx
    hyy = hyy_ * inv_sy * inv_sy
    hzz = hzz_ * inv_sz * inv_sz
    htt = htt_ * inv_st * inv_st
    hxy = hxy_ * inv_sx * inv_sy
    hxz = hxz_ * inv_sx * inv_sz
    hxt = hxt_ * inv_sx * inv_st
    hyz = hyz_ * inv_sy * inv_sz
    hyt = hyt_ * inv_sy * inv_st
    hzt = hzt_ * inv_sz * inv_st

    return val_, gx, gy, gz, gt, hxx, hyy, hzz, htt, hxy, hxz, hxt, hyz, hyt, hzt


# ================================================================
#  LOCATION HELPERS
# ================================================================

@njit(cache=True, fastmath=True)
def _locate_error(x, y, z,
                  origin_x, origin_y, origin_z,
                  spacing_x, spacing_y, spacing_z,
                  nx, ny, nz, margin):
    rel_x = (x - origin_x) / spacing_x
    rel_y = (y - origin_y) / spacing_y
    rel_z = (z - origin_z) / spacing_z
    lo = float(margin)
    hix = float(nx - 1 - margin)
    hiy = float(ny - 1 - margin)
    hiz = float(nz - 1 - margin)
    if (rel_x < lo or rel_x >= hix or
            rel_y < lo or rel_y >= hiy or
            rel_z < lo or rel_z >= hiz):
        return False, 0, 0, 0, 0.0, 0.0, 0.0
    i0 = int(np.floor(rel_x)); j0 = int(np.floor(rel_y)); k0 = int(np.floor(rel_z))
    hi_i = nx - 2 - margin; hi_j = ny - 2 - margin; hi_k = nz - 2 - margin
    if i0 > hi_i: i0 = hi_i
    if j0 > hi_j: j0 = hi_j
    if k0 > hi_k: k0 = hi_k
    dx = rel_x - i0; dy = rel_y - j0; dz = rel_z - k0
    return True, i0, j0, k0, dx, dy, dz


@njit(cache=True, fastmath=True)
def _locate_clamp(x, y, z,
                  origin_x, origin_y, origin_z,
                  spacing_x, spacing_y, spacing_z,
                  nx, ny, nz, margin):
    epsx = 1e-12 * spacing_x; epsy = 1e-12 * spacing_y; epsz = 1e-12 * spacing_z
    x = _clamp_pos(x, origin_x + margin * spacing_x,
                   origin_x + (nx - 1 - margin) * spacing_x - epsx)
    y = _clamp_pos(y, origin_y + margin * spacing_y,
                   origin_y + (ny - 1 - margin) * spacing_y - epsy)
    z = _clamp_pos(z, origin_z + margin * spacing_z,
                   origin_z + (nz - 1 - margin) * spacing_z - epsz)
    rel_x = (x - origin_x) / spacing_x
    rel_y = (y - origin_y) / spacing_y
    rel_z = (z - origin_z) / spacing_z
    i0 = int(np.floor(rel_x)); j0 = int(np.floor(rel_y)); k0 = int(np.floor(rel_z))
    lo = margin; hi_i = nx - 2 - margin; hi_j = ny - 2 - margin; hi_k = nz - 2 - margin
    if i0 < lo: i0 = lo
    if j0 < lo: j0 = lo
    if k0 < lo: k0 = lo
    if i0 > hi_i: i0 = hi_i
    if j0 > hi_j: j0 = hi_j
    if k0 > hi_k: k0 = hi_k
    dx = rel_x - i0; dy = rel_y - j0; dz = rel_z - k0
    return True, i0, j0, k0, dx, dy, dz


@njit(cache=True, fastmath=True)
def _locate_periodic(x, y, z,
                     origin_x, origin_y, origin_z,
                     spacing_x, spacing_y, spacing_z,
                     nx, ny, nz, margin):
    Lx = nx * spacing_x; Ly = ny * spacing_y; Lz = nz * spacing_z
    x = _wrap_pos(x, origin_x, Lx)
    y = _wrap_pos(y, origin_y, Ly)
    z = _wrap_pos(z, origin_z, Lz)
    rel_x = (x - origin_x) / spacing_x
    rel_y = (y - origin_y) / spacing_y
    rel_z = (z - origin_z) / spacing_z
    i0 = int(np.floor(rel_x)); j0 = int(np.floor(rel_y)); k0 = int(np.floor(rel_z))
    dx = rel_x - i0; dy = rel_y - j0; dz = rel_z - k0
    i0 = _wrap_index(i0, nx); j0 = _wrap_index(j0, ny); k0 = _wrap_index(k0, nz)
    return True, i0, j0, k0, dx, dy, dz


@njit(cache=True, fastmath=True)
def _time_locate_error(t, origin_t, spacing_t, nt, margin):
    rel_t = (t - origin_t) / spacing_t
    lo = float(margin); hi = float(nt - 1 - margin)
    if rel_t < lo or rel_t >= hi:
        return False, 0, 0.0
    it0 = int(np.floor(rel_t))
    hi_i = nt - 2 - margin
    if it0 > hi_i: it0 = hi_i
    return True, it0, rel_t - it0


@njit(cache=True, fastmath=True)
def _time_locate_clamp(t, origin_t, spacing_t, nt, margin):
    eps_t = 1e-12 * spacing_t
    t = _clamp_pos(t, origin_t + margin * spacing_t,
                   origin_t + (nt - 1 - margin) * spacing_t - eps_t)
    rel_t = (t - origin_t) / spacing_t
    it0 = int(np.floor(rel_t))
    lo = margin; hi_i = nt - 2 - margin
    if it0 < lo: it0 = lo
    if it0 > hi_i: it0 = hi_i
    return True, it0, rel_t - it0


# ================================================================
#  STENCIL READERS
# ================================================================

@njit(cache=True, fastmath=True)
def _read_stencil_2_direct(field, i0, j0, k0):
    c000 = field[i0, j0, k0]
    c100 = field[i0+1, j0, k0]
    c010 = field[i0, j0+1, k0]
    c001 = field[i0, j0, k0+1]
    c110 = field[i0+1, j0+1, k0]
    c101 = field[i0+1, j0, k0+1]
    c011 = field[i0, j0+1, k0+1]
    c111 = field[i0+1, j0+1, k0+1]
    return c000, c100, c010, c001, c110, c101, c011, c111

@njit(cache=True, fastmath=True)
def _read_stencil_2_periodic(field, i0, j0, k0, nx, ny, nz):
    i1 = _wrap_index(i0+1, nx); j1 = _wrap_index(j0+1, ny); k1 = _wrap_index(k0+1, nz)
    return (field[i0,j0,k0], field[i1,j0,k0], field[i0,j1,k0], field[i0,j0,k1],
            field[i1,j1,k0], field[i1,j0,k1], field[i0,j1,k1], field[i1,j1,k1])

@njit(cache=True, fastmath=True)
def _read_stencil_4_direct(field, i0, j0, k0):
    out = np.empty((4, 4, 4), dtype=np.float64)
    for a in range(4):
        ia = i0 - 1 + a
        for b in range(4):
            jb = j0 - 1 + b
            for g in range(4):
                out[a, b, g] = field[ia, jb, k0 - 1 + g]
    return out

@njit(cache=True, fastmath=True)
def _read_stencil_4_periodic(field, i0, j0, k0, nx, ny, nz):
    out = np.empty((4, 4, 4), dtype=np.float64)
    for a in range(4):
        ia = _wrap_index(i0-1+a, nx)
        for b in range(4):
            jb = _wrap_index(j0-1+b, ny)
            for g in range(4):
                out[a, b, g] = field[ia, jb, _wrap_index(k0-1+g, nz)]
    return out


# ================================================================
#  4D STENCIL READERS
# ================================================================

@njit(cache=True, fastmath=True)
def _read_stencil_4d_2x2_direct(field4d, i0, j0, k0, it0):
    out = np.empty((2, 2, 2, 2), dtype=np.float64)
    for a in range(2):
        for b in range(2):
            for g in range(2):
                for d in range(2):
                    out[a, b, g, d] = field4d[i0+a, j0+b, k0+g, it0+d]
    return out

@njit(cache=True, fastmath=True)
def _read_stencil_4d_2x2_periodic(field4d, i0, j0, k0, it0, nx, ny, nz, nt):
    out = np.empty((2, 2, 2, 2), dtype=np.float64)
    for a in range(2):
        ia = _wrap_index(i0+a, nx)
        for b in range(2):
            jb = _wrap_index(j0+b, ny)
            for g in range(2):
                kg = _wrap_index(k0+g, nz)
                for d in range(2):
                    td = it0 + d
                    if td < 0: td = 0
                    if td >= nt: td = nt - 1
                    out[a, b, g, d] = field4d[ia, jb, kg, td]
    return out

@njit(cache=True, fastmath=True)
def _read_stencil_4d_4x4_direct(field4d, i0, j0, k0, it0):
    out = np.empty((4, 4, 4, 4), dtype=np.float64)
    for a in range(4):
        ia = i0 - 1 + a
        for b in range(4):
            jb = j0 - 1 + b
            for g in range(4):
                kg = k0 - 1 + g
                for d in range(4):
                    out[a, b, g, d] = field4d[ia, jb, kg, it0 - 1 + d]
    return out

@njit(cache=True, fastmath=True)
def _read_stencil_4d_4x4_periodic(field4d, i0, j0, k0, it0, nx, ny, nz, nt):
    out = np.empty((4, 4, 4, 4), dtype=np.float64)
    for a in range(4):
        ia = _wrap_index(i0-1+a, nx)
        for b in range(4):
            jb = _wrap_index(j0-1+b, ny)
            for g in range(4):
                kg = _wrap_index(k0-1+g, nz)
                for d in range(4):
                    td = it0 - 1 + d
                    if td < 0: td = 0
                    if td >= nt: td = nt - 1
                    out[a, b, g, d] = field4d[ia, jb, kg, td]
    return out


# ================================================================
#  FUSED 3D FUNCTIONS
#  Each returns: (val, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz)
# ================================================================

@njit(cache=True, fastmath=True)
def _fused_3d_trilinear_error(x, y, z, field,
    origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz):
    ok, i0, j0, k0, dx, dy, dz = _locate_error(
        x, y, z, origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz, 0)
    if not ok: return np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    c = _read_stencil_2_direct(field, i0, j0, k0)
    return _trilinear_val_grad_hess(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],
        dx, dy, dz, 1.0/spacing_x, 1.0/spacing_y, 1.0/spacing_z)

@njit(cache=True, fastmath=True)
def _fused_3d_trilinear_clamp(x, y, z, field,
    origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz):
    ok, i0, j0, k0, dx, dy, dz = _locate_clamp(
        x, y, z, origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz, 0)
    c = _read_stencil_2_direct(field, i0, j0, k0)
    return _trilinear_val_grad_hess(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],
        dx, dy, dz, 1.0/spacing_x, 1.0/spacing_y, 1.0/spacing_z)

@njit(cache=True, fastmath=True)
def _fused_3d_trilinear_periodic(x, y, z, field,
    origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz):
    ok, i0, j0, k0, dx, dy, dz = _locate_periodic(
        x, y, z, origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz, 0)
    c = _read_stencil_2_periodic(field, i0, j0, k0, nx, ny, nz)
    return _trilinear_val_grad_hess(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],
        dx, dy, dz, 1.0/spacing_x, 1.0/spacing_y, 1.0/spacing_z)

@njit(cache=True, fastmath=True)
def _fused_3d_tricubic_error(x, y, z, field,
    origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz):
    ok, i0, j0, k0, dx, dy, dz = _locate_error(
        x, y, z, origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz, 1)
    if not ok: return np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    corners = _read_stencil_4_direct(field, i0, j0, k0)
    return _tricubic_val_grad_hess(corners, dx, dy, dz,
        1.0/spacing_x, 1.0/spacing_y, 1.0/spacing_z)

@njit(cache=True, fastmath=True)
def _fused_3d_tricubic_clamp(x, y, z, field,
    origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz):
    ok, i0, j0, k0, dx, dy, dz = _locate_clamp(
        x, y, z, origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz, 1)
    corners = _read_stencil_4_direct(field, i0, j0, k0)
    return _tricubic_val_grad_hess(corners, dx, dy, dz,
        1.0/spacing_x, 1.0/spacing_y, 1.0/spacing_z)

@njit(cache=True, fastmath=True)
def _fused_3d_tricubic_periodic(x, y, z, field,
    origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz):
    ok, i0, j0, k0, dx, dy, dz = _locate_periodic(
        x, y, z, origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz, 0)
    corners = _read_stencil_4_periodic(field, i0, j0, k0, nx, ny, nz)
    return _tricubic_val_grad_hess(corners, dx, dy, dz,
        1.0/spacing_x, 1.0/spacing_y, 1.0/spacing_z)


# ================================================================
#  FUSED TRUE-4D FUNCTIONS  (15-tuple return)
# ================================================================

_NAN15 = (np.nan, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0)

@njit(cache=True, fastmath=True)
def _fused_true4d_trilinear_error(x, y, z, t, field4d,
    ox, oy, oz, ot, sx, sy, sz, st, nx, ny, nz, nt):
    ok_s, i0, j0, k0, ddx, ddy, ddz = _locate_error(x,y,z,ox,oy,oz,sx,sy,sz,nx,ny,nz,0)
    if not ok_s: return (np.nan,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
    ok_t, it0, ddt = _time_locate_error(t, ot, st, nt, 0)
    if not ok_t: return (np.nan,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
    s = _read_stencil_4d_2x2_direct(field4d, i0, j0, k0, it0)
    return _interp_4d_linear_kernel(s, ddx, ddy, ddz, ddt, 1./sx, 1./sy, 1./sz, 1./st)

@njit(cache=True, fastmath=True)
def _fused_true4d_trilinear_clamp(x, y, z, t, field4d,
    ox, oy, oz, ot, sx, sy, sz, st, nx, ny, nz, nt):
    ok_s, i0, j0, k0, ddx, ddy, ddz = _locate_clamp(x,y,z,ox,oy,oz,sx,sy,sz,nx,ny,nz,0)
    ok_t, it0, ddt = _time_locate_clamp(t, ot, st, nt, 0)
    s = _read_stencil_4d_2x2_direct(field4d, i0, j0, k0, it0)
    return _interp_4d_linear_kernel(s, ddx, ddy, ddz, ddt, 1./sx, 1./sy, 1./sz, 1./st)

@njit(cache=True, fastmath=True)
def _fused_true4d_trilinear_periodic(x, y, z, t, field4d,
    ox, oy, oz, ot, sx, sy, sz, st, nx, ny, nz, nt):
    ok_s, i0, j0, k0, ddx, ddy, ddz = _locate_periodic(x,y,z,ox,oy,oz,sx,sy,sz,nx,ny,nz,0)
    ok_t, it0, ddt = _time_locate_error(t, ot, st, nt, 0)
    if not ok_t: return (np.nan,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
    s = _read_stencil_4d_2x2_periodic(field4d, i0, j0, k0, it0, nx, ny, nz, nt)
    return _interp_4d_linear_kernel(s, ddx, ddy, ddz, ddt, 1./sx, 1./sy, 1./sz, 1./st)

@njit(cache=True, fastmath=True)
def _fused_true4d_tricubic_error(x, y, z, t, field4d,
    ox, oy, oz, ot, sx, sy, sz, st, nx, ny, nz, nt):
    ok_s, i0, j0, k0, ddx, ddy, ddz = _locate_error(x,y,z,ox,oy,oz,sx,sy,sz,nx,ny,nz,1)
    if not ok_s: return (np.nan,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
    ok_t, it0, ddt = _time_locate_error(t, ot, st, nt, 1)
    if not ok_t: return (np.nan,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
    s = _read_stencil_4d_4x4_direct(field4d, i0, j0, k0, it0)
    return _interp_4d_cubic_kernel(s, ddx, ddy, ddz, ddt, 1./sx, 1./sy, 1./sz, 1./st)

@njit(cache=True, fastmath=True)
def _fused_true4d_tricubic_clamp(x, y, z, t, field4d,
    ox, oy, oz, ot, sx, sy, sz, st, nx, ny, nz, nt):
    ok_s, i0, j0, k0, ddx, ddy, ddz = _locate_clamp(x,y,z,ox,oy,oz,sx,sy,sz,nx,ny,nz,1)
    ok_t, it0, ddt = _time_locate_clamp(t, ot, st, nt, 1)
    s = _read_stencil_4d_4x4_direct(field4d, i0, j0, k0, it0)
    return _interp_4d_cubic_kernel(s, ddx, ddy, ddz, ddt, 1./sx, 1./sy, 1./sz, 1./st)

@njit(cache=True, fastmath=True)
def _fused_true4d_tricubic_periodic(x, y, z, t, field4d,
    ox, oy, oz, ot, sx, sy, sz, st, nx, ny, nz, nt):
    ok_s, i0, j0, k0, ddx, ddy, ddz = _locate_periodic(x,y,z,ox,oy,oz,sx,sy,sz,nx,ny,nz,0)
    ok_t, it0, ddt = _time_locate_error(t, ot, st, nt, 1)
    if not ok_t: return (np.nan,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
    s = _read_stencil_4d_4x4_periodic(field4d, i0, j0, k0, it0, nx, ny, nz, nt)
    return _interp_4d_cubic_kernel(s, ddx, ddy, ddz, ddt, 1./sx, 1./sy, 1./sz, 1./st)


# ================================================================
#  BACKWARD-COMPATIBLE ALIASES (old 5-tuple return)
# ================================================================

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_3d_error(x, y, z, field,
    origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz):
    r = _fused_3d_trilinear_error(x,y,z,field,origin_x,origin_y,origin_z,
        spacing_x,spacing_y,spacing_z,nx,ny,nz)
    return r[0], r[1], r[2], r[3], 0.0

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_3d_clamp(x, y, z, field,
    origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz):
    r = _fused_3d_trilinear_clamp(x,y,z,field,origin_x,origin_y,origin_z,
        spacing_x,spacing_y,spacing_z,nx,ny,nz)
    return r[0], r[1], r[2], r[3], 0.0

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_3d_periodic(x, y, z, field,
    origin_x, origin_y, origin_z, spacing_x, spacing_y, spacing_z, nx, ny, nz):
    r = _fused_3d_trilinear_periodic(x,y,z,field,origin_x,origin_y,origin_z,
        spacing_x,spacing_y,spacing_z,nx,ny,nz)
    return r[0], r[1], r[2], r[3], 0.0

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_4d_error(x,y,z,t,field4d,
    ox,oy,oz,ot,sx,sy,sz,st,nx,ny,nz,nt):
    r = _fused_true4d_trilinear_error(x,y,z,t,field4d,ox,oy,oz,ot,sx,sy,sz,st,nx,ny,nz,nt)
    return r[0], r[1], r[2], r[3], r[4]

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_4d_clamp(x,y,z,t,field4d,
    ox,oy,oz,ot,sx,sy,sz,st,nx,ny,nz,nt):
    r = _fused_true4d_trilinear_clamp(x,y,z,t,field4d,ox,oy,oz,ot,sx,sy,sz,st,nx,ny,nz,nt)
    return r[0], r[1], r[2], r[3], r[4]

@njit(cache=True, fastmath=True)
def fused_interpolation_gradient_4d_periodic(x,y,z,t,field4d,
    ox,oy,oz,ot,sx,sy,sz,st,nx,ny,nz,nt):
    r = _fused_true4d_trilinear_periodic(x,y,z,t,field4d,ox,oy,oz,ot,sx,sy,sz,st,nx,ny,nz,nt)
    return r[0], r[1], r[2], r[3], r[4]


# ================================================================
#  PUBLIC CLASS
# ================================================================

class InterpolatorFast:
    """
    Fast grid interpolator with pluggable scheme and boundary mode.

    Kernel performance:
      3D tricubic uses separable contractions (no triple loop).
      4D tricubic uses separable 4-pass contractions (no quadruple loop).
    """

    _FN_3D = {
        ("trilinear", "error"): _fused_3d_trilinear_error,
        ("trilinear", "clamp"): _fused_3d_trilinear_clamp,
        ("trilinear", "periodic"): _fused_3d_trilinear_periodic,
        ("tricubic", "error"): _fused_3d_tricubic_error,
        ("tricubic", "clamp"): _fused_3d_tricubic_clamp,
        ("tricubic", "periodic"): _fused_3d_tricubic_periodic,
    }
    _FN_4D = {
        ("trilinear", "error"): _fused_true4d_trilinear_error,
        ("trilinear", "clamp"): _fused_true4d_trilinear_clamp,
        ("trilinear", "periodic"): _fused_true4d_trilinear_periodic,
        ("tricubic", "error"): _fused_true4d_tricubic_error,
        ("tricubic", "clamp"): _fused_true4d_tricubic_clamp,
        ("tricubic", "periodic"): _fused_true4d_tricubic_periodic,
    }

    SUPPORTED_SCHEMES = ("trilinear", "tricubic")
    SUPPORTED_BOUNDARIES = ("error", "clamp", "periodic")

    def __init__(self, grid, boundary="periodic", scheme="trilinear"):
        self.grid = grid
        self.is_4d = len(grid.shape) == 4
        self.scheme = scheme
        self.boundary = boundary

        if boundary not in self.SUPPORTED_BOUNDARIES:
            raise ValueError(f"boundary must be one of {self.SUPPORTED_BOUNDARIES}")
        if scheme not in self.SUPPORTED_SCHEMES:
            raise ValueError(f"scheme must be one of {self.SUPPORTED_SCHEMES}")

        self.origin = np.array(grid.origin, dtype=np.float64)
        self.spacing = np.array(grid.spacing, dtype=np.float64)
        self.shape = np.array(grid.shape, dtype=np.int64)

        ndim = 4 if self.is_4d else 3
        if self.origin.shape[0] < ndim:
            raise ValueError(f"{ndim}D grid requires origin/spacing/shape of length >= {ndim}")

        min_stencil = 4 if scheme == "tricubic" else 2
        for d in range(3):
            if self.shape[d] < min_stencil:
                raise ValueError(
                    f"{scheme} needs grid size >= {min_stencil} in each spatial "
                    f"dim (dim {d} has {self.shape[d]})")
        if self.is_4d and self.shape[3] < min_stencil:
            raise ValueError(f"{scheme} needs >= {min_stencil} time slices (got {self.shape[3]})")

        key = (scheme, boundary)
        self._fn_3d = self._FN_3D[key]
        self._fn_4d = self._FN_4D[key]

    # ----------------------------------------------------------
    #  Full 4D: val + 4-gradient + 4x4 Hessian
    # ----------------------------------------------------------

    def full_4d(self, x, field, t=None):
        f = self.grid.fields[field]
        if self.is_4d:
            if t is None: raise ValueError("4D grid requires t")
            r = self._fn_4d(
                x[0], x[1], x[2], t, f,
                self.origin[0], self.origin[1], self.origin[2], self.origin[3],
                self.spacing[0], self.spacing[1], self.spacing[2], self.spacing[3],
                int(self.shape[0]), int(self.shape[1]), int(self.shape[2]), int(self.shape[3]))
            val = r[0]
            grad4 = (r[1], r[2], r[3], r[4])
            hess4 = (r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14])
        else:
            r = self._fn_3d(
                x[0], x[1], x[2], f,
                self.origin[0], self.origin[1], self.origin[2],
                self.spacing[0], self.spacing[1], self.spacing[2],
                int(self.shape[0]), int(self.shape[1]), int(self.shape[2]))
            val = r[0]
            grad4 = (r[1], r[2], r[3], 0.0)
            hess4 = (r[4], r[5], r[6], 0.0, r[7], r[8], 0.0, r[9], 0.0, 0.0)

        if np.isnan(val):
            raise ValueError(f"Point {x} (t={t}) outside grid bounds (boundary='{self.boundary}')")
        return val, grad4, hess4

    # ----------------------------------------------------------
    #  Backward-compatible 3D spatial accessors
    # ----------------------------------------------------------

    def value_gradient_hessian_and_time_derivative(self, x, field, t=None):
        val, grad4, hess4 = self.full_4d(x, field, t)
        grad3 = (grad4[0], grad4[1], grad4[2])
        hess3 = (hess4[0], hess4[1], hess4[2], hess4[4], hess4[5], hess4[7])
        return val, grad3, hess3, grad4[3]

    def value_gradient_and_time_derivative(self, x, field, t=None):
        val, grad4, _h = self.full_4d(x, field, t)
        return val, (grad4[0], grad4[1], grad4[2]), grad4[3]

    def interpolate(self, x, field, t=None):
        val, _, _ = self.full_4d(x, field, t)
        return val

    def gradient(self, x, field, t=None):
        _, grad4, _ = self.full_4d(x, field, t)
        return (grad4[0], grad4[1], grad4[2])

    def value_and_gradient(self, x, field, t=None):
        val, grad4, _ = self.full_4d(x, field, t)
        return val, (grad4[0], grad4[1], grad4[2])

    # ----------------------------------------------------------
    #  Full 4D accessors
    # ----------------------------------------------------------

    def gradient4d(self, x, field, t=None):
        _, grad4, _ = self.full_4d(x, field, t)
        return grad4

    def hessian4d(self, x, field, t=None):
        _, _, hess4 = self.full_4d(x, field, t)
        hxx, hyy, hzz, htt, hxy, hxz, hxt, hyz, hyt, hzt = hess4
        return np.array([
            [hxx, hxy, hxz, hxt],
            [hxy, hyy, hyz, hyt],
            [hxz, hyz, hzz, hzt],
            [hxt, hyt, hzt, htt]])

    def hessian(self, x, field, t=None):
        _, _, hess4 = self.full_4d(x, field, t)
        hxx, hyy, hzz, _htt, hxy, hxz, _hxt, hyz, _hyt, _hzt = hess4
        return np.array([
            [hxx, hxy, hxz],
            [hxy, hyy, hyz],
            [hxz, hyz, hzz]])

    def laplacian(self, x, field, t=None):
        _, _, hess4 = self.full_4d(x, field, t)
        return hess4[0] + hess4[1] + hess4[2]

    def laplacian4d(self, x, field, t=None):
        _, _, hess4 = self.full_4d(x, field, t)
        return hess4[0] + hess4[1] + hess4[2] + hess4[3]
