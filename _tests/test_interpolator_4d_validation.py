"""
Comprehensive validation tests for excalibur.grid.interpolator_4d_fast

Tests cover:
  1. 3D trilinear & tricubic: exact polynomial reproduction
  2. 4D trilinear & tricubic: exact polynomial reproduction (true 4D)
  3. 4-gradient vs finite differences
  4. 4x4 Hessian vs finite differences
  5. Boundary modes (clamp, periodic, error)
  6. Backward-compatible API: value_gradient_and_time_derivative
"""
import numpy as np
import pytest

from excalibur.grid.interpolator_4d_fast import InterpolatorFast


# ================================================================
#  Mock Grid classes
# ================================================================

class MockGrid3D:
    """Simple 3D grid for testing."""

    def __init__(self, field_func, nx=16, ny=16, nz=16,
                 origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
        self.shape = (nx, ny, nz)
        self.origin = origin
        self.spacing = spacing
        xs = origin[0] + np.arange(nx) * spacing[0]
        ys = origin[1] + np.arange(ny) * spacing[1]
        zs = origin[2] + np.arange(nz) * spacing[2]
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        self.fields = {"Phi": field_func(X, Y, Z)}


class MockGrid4D:
    """Simple 4D grid (x, y, z, t) for testing."""

    def __init__(self, field_func, nx=16, ny=16, nz=16, nt=16,
                 origin=(0.0, 0.0, 0.0, 0.0),
                 spacing=(1.0, 1.0, 1.0, 1.0)):
        self.shape = (nx, ny, nz, nt)
        self.origin = origin
        self.spacing = spacing
        xs = origin[0] + np.arange(nx) * spacing[0]
        ys = origin[1] + np.arange(ny) * spacing[1]
        zs = origin[2] + np.arange(nz) * spacing[2]
        ts = origin[3] + np.arange(nt) * spacing[3]
        X, Y, Z, T = np.meshgrid(xs, ys, zs, ts, indexing='ij')
        self.fields = {"Phi": field_func(X, Y, Z, T)}


# ================================================================
#  Analytic test functions and their derivatives
# ================================================================

def f_quadratic_3d(X, Y, Z):
    """f = x^2 + 2*y^2 + 3*z^2"""
    return X**2 + 2.0 * Y**2 + 3.0 * Z**2

def grad_quadratic_3d(x, y, z):
    """Gradient of f = x^2 + 2y^2 + 3z^2"""
    return 2.0 * x, 4.0 * y, 6.0 * z

def hess_quadratic_3d():
    """Hessian of f = x^2 + 2y^2 + 3z^2"""
    return np.array([[2.0, 0.0, 0.0],
                     [0.0, 4.0, 0.0],
                     [0.0, 0.0, 6.0]])


def f_cubic_3d(X, Y, Z):
    """f = x^3 + y^3 + z^3 + x*y*z"""
    return X**3 + Y**3 + Z**3 + X * Y * Z

def grad_cubic_3d(x, y, z):
    return 3*x**2 + y*z, 3*y**2 + x*z, 3*z**2 + x*y

def hess_cubic_3d(x, y, z):
    return np.array([[6*x, z, y],
                     [z, 6*y, x],
                     [y, x, 6*z]])


def f_quadratic_4d(X, Y, Z, T):
    """f = x^2 + 2*y^2 + 3*z^2 + 4*t^2 + x*t"""
    return X**2 + 2.0 * Y**2 + 3.0 * Z**2 + 4.0 * T**2 + X * T

def grad_quadratic_4d(x, y, z, t):
    """(gx, gy, gz, gt)"""
    return 2*x + t, 4*y, 6*z, 8*t + x

def hess_quadratic_4d():
    """Full 4x4 Hessian:
    [[2, 0, 0, 1],
     [0, 4, 0, 0],
     [0, 0, 6, 0],
     [1, 0, 0, 8]]
    """
    return np.array([[2.0, 0.0, 0.0, 1.0],
                     [0.0, 4.0, 0.0, 0.0],
                     [0.0, 0.0, 6.0, 0.0],
                     [1.0, 0.0, 0.0, 8.0]])


def f_cubic_4d(X, Y, Z, T):
    """f = x^3 + y^3 + z^3 + t^3 + x*y*t"""
    return X**3 + Y**3 + Z**3 + T**3 + X * Y * T


def grad_cubic_4d(x, y, z, t):
    return 3*x**2 + y*t, 3*y**2 + x*t, 3*z**2, 3*t**2 + x*y


def hess_cubic_4d(x, y, z, t):
    return np.array([
        [6*x, t, 0, y],
        [t, 6*y, 0, x],
        [0, 0, 6*z, 0],
        [y, x, 0, 6*t],
    ])


# ================================================================
#  3D TESTS
# ================================================================

class Test3DTrilinear:
    """Tests for 3D trilinear interpolation."""

    @pytest.fixture
    def grid_linear(self):
        """Linear field: f = 2x + 3y + 5z. Trilinear should be exact."""
        def f(X, Y, Z): return 2*X + 3*Y + 5*Z
        return MockGrid3D(f, nx=8, ny=8, nz=8,
                          origin=(0, 0, 0), spacing=(1, 1, 1))

    def test_value_exact_linear(self, grid_linear):
        interp = InterpolatorFast(grid_linear, boundary="clamp",
                                  scheme="trilinear")
        pos = np.array([2.5, 3.3, 1.7])
        val = interp.interpolate(pos, "Phi")
        expected = 2*2.5 + 3*3.3 + 5*1.7
        assert abs(val - expected) < 1e-12

    def test_gradient_exact_linear(self, grid_linear):
        interp = InterpolatorFast(grid_linear, boundary="clamp",
                                  scheme="trilinear")
        pos = np.array([2.5, 3.3, 1.7])
        gx, gy, gz = interp.gradient(pos, "Phi")
        assert abs(gx - 2.0) < 1e-12
        assert abs(gy - 3.0) < 1e-12
        assert abs(gz - 5.0) < 1e-12


class Test3DTricubic:
    """Tests for 3D tricubic (Catmull-Rom) interpolation."""

    @pytest.fixture
    def grid_quad(self):
        return MockGrid3D(f_quadratic_3d, nx=16, ny=16, nz=16,
                          origin=(0, 0, 0), spacing=(1, 1, 1))

    @pytest.fixture
    def grid_cubic(self):
        return MockGrid3D(f_cubic_3d, nx=16, ny=16, nz=16,
                          origin=(0, 0, 0), spacing=(1, 1, 1))

    def test_value_exact_quadratic(self, grid_quad):
        """Catmull-Rom reproduces quadratics exactly."""
        interp = InterpolatorFast(grid_quad, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        val = interp.interpolate(pos, "Phi")
        expected = 3**2 + 2*4**2 + 3*5**2
        assert abs(val - expected) < 1e-10, f"val={val}, expected={expected}"

    def test_gradient_exact_quadratic(self, grid_quad):
        interp = InterpolatorFast(grid_quad, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        gx, gy, gz = interp.gradient(pos, "Phi")
        ex, ey, ez = grad_quadratic_3d(3, 4, 5)
        assert abs(gx - ex) < 1e-10, f"gx={gx}, expected={ex}"
        assert abs(gy - ey) < 1e-10
        assert abs(gz - ez) < 1e-10

    def test_hessian_exact_quadratic(self, grid_quad):
        interp = InterpolatorFast(grid_quad, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        H = interp.hessian(pos, "Phi")
        H_exact = hess_quadratic_3d()
        np.testing.assert_allclose(H, H_exact, atol=1e-10)

    def test_laplacian_exact_quadratic(self, grid_quad):
        interp = InterpolatorFast(grid_quad, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        lap = interp.laplacian(pos, "Phi")
        assert abs(lap - 12.0) < 1e-10

    def test_gradient_cubic_field(self, grid_cubic):
        """Catmull-Rom value+gradient exact for quadratics.
        For cubics, value is approximate -> gradient approximate.
        Use fine grid for better accuracy."""
        fine_grid = MockGrid3D(f_cubic_3d, nx=64, ny=64, nz=64,
                               origin=(0, 0, 0), spacing=(0.25, 0.25, 0.25))
        interp = InterpolatorFast(fine_grid, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([5.3, 4.7, 3.2])
        gx, gy, gz = interp.gradient(pos, "Phi")
        ex, ey, ez = grad_cubic_3d(*pos)
        assert abs(gx - ex) < 0.05, f"gx={gx}, expected={ex}"
        assert abs(gy - ey) < 0.05, f"gy={gy}, expected={ey}"
        assert abs(gz - ez) < 0.05, f"gz={gz}, expected={ez}"

    def test_hessian_cubic_field(self, grid_cubic):
        """Catmull-Rom Hessian is exact for quadratics but has O(h^2)
        error for cubics.  Use a fine grid to get good agreement."""
        # Use fine grid for better accuracy
        fine_grid = MockGrid3D(f_cubic_3d, nx=64, ny=64, nz=64,
                               origin=(0, 0, 0), spacing=(0.25, 0.25, 0.25))
        interp = InterpolatorFast(fine_grid, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([5.3, 4.7, 3.2])
        H = interp.hessian(pos, "Phi")
        H_exact = hess_cubic_3d(*pos)
        np.testing.assert_allclose(H, H_exact, atol=1.0)


# ================================================================
#  4D TESTS  (True 4-dimensional interpolation)
# ================================================================

class Test4DTrilinear:
    """Tests for 4D trilinear (2^4) interpolation."""

    @pytest.fixture
    def grid_linear_4d(self):
        """f = 2x + 3y + 5z + 7t. Linear -> trilinear exact."""
        def f(X, Y, Z, T): return 2*X + 3*Y + 5*Z + 7*T
        return MockGrid4D(f, nx=8, ny=8, nz=8, nt=8,
                          origin=(0, 0, 0, 0), spacing=(1, 1, 1, 1))

    def test_value_exact_linear(self, grid_linear_4d):
        interp = InterpolatorFast(grid_linear_4d, boundary="clamp",
                                  scheme="trilinear")
        pos = np.array([2.5, 3.3, 1.7])
        val = interp.interpolate(pos, "Phi", t=4.2)
        expected = 2*2.5 + 3*3.3 + 5*1.7 + 7*4.2
        assert abs(val - expected) < 1e-10

    def test_gradient4d_exact_linear(self, grid_linear_4d):
        interp = InterpolatorFast(grid_linear_4d, boundary="clamp",
                                  scheme="trilinear")
        pos = np.array([2.5, 3.3, 1.7])
        gx, gy, gz, gt = interp.gradient4d(pos, "Phi", t=4.2)
        assert abs(gx - 2.0) < 1e-10
        assert abs(gy - 3.0) < 1e-10
        assert abs(gz - 5.0) < 1e-10
        assert abs(gt - 7.0) < 1e-10


class Test4DTricubic:
    """Tests for 4D tricubic (Catmull-Rom) interpolation."""

    @pytest.fixture
    def grid_quad_4d(self):
        return MockGrid4D(f_quadratic_4d, nx=16, ny=16, nz=16, nt=16,
                          origin=(0, 0, 0, 0), spacing=(1, 1, 1, 1))

    @pytest.fixture
    def grid_cubic_4d(self):
        return MockGrid4D(f_cubic_4d, nx=16, ny=16, nz=16, nt=16,
                          origin=(0, 0, 0, 0), spacing=(1, 1, 1, 1))

    def test_value_exact_quadratic(self, grid_quad_4d):
        interp = InterpolatorFast(grid_quad_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        t = 6.0
        val = interp.interpolate(pos, "Phi", t=t)
        expected = f_quadratic_4d(3, 4, 5, 6)
        assert abs(val - expected) < 1e-8, f"val={val}, expected={expected}"

    def test_gradient4d_exact_quadratic(self, grid_quad_4d):
        interp = InterpolatorFast(grid_quad_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        t = 6.0
        gx, gy, gz, gt = interp.gradient4d(pos, "Phi", t=t)
        ex, ey, ez, et_ = grad_quadratic_4d(3, 4, 5, 6)
        assert abs(gx - ex) < 1e-8, f"gx={gx}, expected={ex}"
        assert abs(gy - ey) < 1e-8, f"gy={gy}, expected={ey}"
        assert abs(gz - ez) < 1e-8, f"gz={gz}, expected={ez}"
        assert abs(gt - et_) < 1e-8, f"gt={gt}, expected={et_}"

    def test_hessian4d_exact_quadratic(self, grid_quad_4d):
        interp = InterpolatorFast(grid_quad_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        t = 6.0
        H = interp.hessian4d(pos, "Phi", t=t)
        H_exact = hess_quadratic_4d()
        np.testing.assert_allclose(H, H_exact, atol=1e-8,
                                   err_msg=f"H=\n{H}\nexpected=\n{H_exact}")

    def test_laplacian4d_quadratic(self, grid_quad_4d):
        """Laplacian4d = d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2 + d^2f/dt^2
        = 2 + 4 + 6 + 8 = 20"""
        interp = InterpolatorFast(grid_quad_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        t = 6.0
        lap4 = interp.laplacian4d(pos, "Phi", t=t)
        assert abs(lap4 - 20.0) < 1e-8

    def test_spatial_laplacian_quadratic(self, grid_quad_4d):
        """Spatial laplacian = 2 + 4 + 6 = 12."""
        interp = InterpolatorFast(grid_quad_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        t = 6.0
        lap = interp.laplacian(pos, "Phi", t=t)
        assert abs(lap - 12.0) < 1e-8

    def test_gradient4d_cubic_field(self, grid_cubic_4d):
        """Catmull-Rom value + gradient exact for quadratics, approximate
        for cubics.  Use fine grid for better accuracy."""
        fine_grid = MockGrid4D(f_cubic_4d, nx=32, ny=32, nz=32, nt=32,
                               origin=(0, 0, 0, 0),
                               spacing=(0.5, 0.5, 0.5, 0.5))
        interp = InterpolatorFast(fine_grid, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([5.3, 4.7, 3.2])
        t = 6.4
        gx, gy, gz, gt = interp.gradient4d(pos, "Phi", t=t)
        ex, ey, ez, et_ = grad_cubic_4d(*pos, t)
        # With spacing=0.5, CR cubic error is O(h^2)~0.25, so ~1% relative
        assert abs(gx - ex) < 0.15, f"gx={gx}, expected={ex}"
        assert abs(gy - ey) < 0.15, f"gy={gy}, expected={ey}"
        assert abs(gz - ez) < 0.15, f"gz={gz}, expected={ez}"
        assert abs(gt - et_) < 0.15, f"gt={gt}, expected={et_}"

    def test_hessian4d_cubic_field(self, grid_cubic_4d):
        """Catmull-Rom Hessian has O(h^2) error for cubics."""
        fine_grid = MockGrid4D(f_cubic_4d, nx=32, ny=32, nz=32, nt=32,
                               origin=(0, 0, 0, 0),
                               spacing=(0.5, 0.5, 0.5, 0.5))
        interp = InterpolatorFast(fine_grid, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([5.3, 4.7, 3.2])
        t = 6.4
        H = interp.hessian4d(pos, "Phi", t=t)
        H_exact = hess_cubic_4d(*pos, t)
        np.testing.assert_allclose(H, H_exact, atol=2.0,
                                   err_msg=f"H=\n{H}\nexpected=\n{H_exact}")

    def test_mixed_derivative_xt(self, grid_quad_4d):
        """The xt cross-derivative of x^2 + 4t^2 + x*t is 1."""
        interp = InterpolatorFast(grid_quad_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        t = 6.0
        H = interp.hessian4d(pos, "Phi", t=t)
        # H[0,3] = d^2f / (dx dt) = 1
        assert abs(H[0, 3] - 1.0) < 1e-8, f"H[0,3]={H[0,3]}"


# ================================================================
#  GRADIENT VS FINITE DIFFERENCES
# ================================================================

class TestGradientFiniteDiff:
    """Validate gradient against finite differences for a non-polynomial."""

    @pytest.fixture
    def grid_sin_3d(self):
        def f(X, Y, Z): return np.sin(X) * np.cos(Y) * np.exp(0.1 * Z)
        return MockGrid3D(f, nx=32, ny=32, nz=32,
                          origin=(0, 0, 0), spacing=(0.5, 0.5, 0.5))

    @pytest.fixture
    def grid_sin_4d(self):
        def f(X, Y, Z, T):
            return np.sin(X) * np.cos(Y) * np.exp(0.1 * Z) * np.sin(0.5 * T)
        return MockGrid4D(f, nx=20, ny=20, nz=20, nt=20,
                          origin=(0, 0, 0, 0),
                          spacing=(0.5, 0.5, 0.5, 0.5))

    def test_gradient_3d_vs_fd(self, grid_sin_3d):
        interp = InterpolatorFast(grid_sin_3d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 3.0])
        gx, gy, gz = interp.gradient(pos, "Phi")

        eps = 1e-5
        vxp = interp.interpolate(pos + np.array([eps, 0, 0]), "Phi")
        vxm = interp.interpolate(pos - np.array([eps, 0, 0]), "Phi")
        vyp = interp.interpolate(pos + np.array([0, eps, 0]), "Phi")
        vym = interp.interpolate(pos - np.array([0, eps, 0]), "Phi")
        vzp = interp.interpolate(pos + np.array([0, 0, eps]), "Phi")
        vzm = interp.interpolate(pos - np.array([0, 0, eps]), "Phi")

        fd_gx = (vxp - vxm) / (2 * eps)
        fd_gy = (vyp - vym) / (2 * eps)
        fd_gz = (vzp - vzm) / (2 * eps)

        assert abs(gx - fd_gx) < 1e-4, f"gx={gx}, fd={fd_gx}"
        assert abs(gy - fd_gy) < 1e-4, f"gy={gy}, fd={fd_gy}"
        assert abs(gz - fd_gz) < 1e-4, f"gz={gz}, fd={fd_gz}"

    def test_gradient_4d_vs_fd(self, grid_sin_4d):
        interp = InterpolatorFast(grid_sin_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 3.0, 3.0])
        t = 3.0
        gx, gy, gz, gt = interp.gradient4d(pos, "Phi", t=t)

        eps = 1e-5
        vxp = interp.interpolate(pos + np.array([eps, 0, 0]), "Phi", t=t)
        vxm = interp.interpolate(pos - np.array([eps, 0, 0]), "Phi", t=t)
        vyp = interp.interpolate(pos + np.array([0, eps, 0]), "Phi", t=t)
        vym = interp.interpolate(pos - np.array([0, eps, 0]), "Phi", t=t)
        vzp = interp.interpolate(pos + np.array([0, 0, eps]), "Phi", t=t)
        vzm = interp.interpolate(pos - np.array([0, 0, eps]), "Phi", t=t)
        vtp = interp.interpolate(pos, "Phi", t=t + eps)
        vtm = interp.interpolate(pos, "Phi", t=t - eps)

        fd_gx = (vxp - vxm) / (2 * eps)
        fd_gy = (vyp - vym) / (2 * eps)
        fd_gz = (vzp - vzm) / (2 * eps)
        fd_gt = (vtp - vtm) / (2 * eps)

        assert abs(gx - fd_gx) < 1e-4, f"gx={gx}, fd={fd_gx}"
        assert abs(gy - fd_gy) < 1e-4, f"gy={gy}, fd={fd_gy}"
        assert abs(gz - fd_gz) < 1e-4, f"gz={gz}, fd={fd_gz}"
        assert abs(gt - fd_gt) < 1e-4, f"gt={gt}, fd={fd_gt}"


# ================================================================
#  HESSIAN VS FINITE DIFFERENCES
# ================================================================

class TestHessianFiniteDiff:
    """Validate Hessian against 2nd-order finite differences."""

    @pytest.fixture
    def grid_sin_4d(self):
        def f(X, Y, Z, T):
            return np.sin(X) * np.cos(Y) * np.exp(0.1 * Z) * np.sin(0.5 * T)
        return MockGrid4D(f, nx=32, ny=32, nz=32, nt=32,
                          origin=(0, 0, 0, 0),
                          spacing=(0.3, 0.3, 0.3, 0.3))

    def test_hessian4d_diagonal_vs_fd(self, grid_sin_4d):
        interp = InterpolatorFast(grid_sin_4d, boundary="clamp",
                                  scheme="tricubic")
        # Use fractional positions (avoid grid nodes)
        pos = np.array([2.75, 2.85, 2.65])
        t = 2.95
        H = interp.hessian4d(pos, "Phi", t=t)
        v0 = interp.interpolate(pos, "Phi", t=t)

        eps = 1e-4
        dirs = [np.array([eps, 0, 0]), np.array([0, eps, 0]),
                np.array([0, 0, eps])]
        # Diagonal spatial: d^2f/dxi^2 = (f(+eps) - 2f(0) + f(-eps)) / eps^2
        for i, d in enumerate(dirs):
            vp = interp.interpolate(pos + d, "Phi", t=t)
            vm = interp.interpolate(pos - d, "Phi", t=t)
            fd_hii = (vp - 2 * v0 + vm) / eps**2
            assert abs(H[i, i] - fd_hii) < 1e-3, \
                f"H[{i},{i}]={H[i,i]}, fd={fd_hii}"

        # Time diagonal
        vtp = interp.interpolate(pos, "Phi", t=t + eps)
        vtm = interp.interpolate(pos, "Phi", t=t - eps)
        fd_htt = (vtp - 2 * v0 + vtm) / eps**2
        assert abs(H[3, 3] - fd_htt) < 1e-3, \
            f"H[3,3]={H[3,3]}, fd={fd_htt}"

    def test_hessian4d_off_diagonal_vs_fd(self, grid_sin_4d):
        interp = InterpolatorFast(grid_sin_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([2.75, 2.85, 2.65])
        t = 2.95
        H = interp.hessian4d(pos, "Phi", t=t)

        eps = 1e-4
        # d^2f/(dxi dxj) = (f(+i+j) - f(+i-j) - f(-i+j) + f(-i-j))/(4 eps^2)
        labels = ['x', 'y', 'z', 't']
        for i in range(4):
            for j in range(i + 1, 4):
                def shift_val(si, sj):
                    p = pos.copy()
                    tt = t
                    if i < 3: p[i] += si * eps
                    else: tt += si * eps
                    if j < 3: p[j] += sj * eps
                    else: tt += sj * eps
                    return interp.interpolate(p, "Phi", t=tt)

                fd = (shift_val(1, 1) - shift_val(1, -1)
                      - shift_val(-1, 1) + shift_val(-1, -1)) / (4 * eps**2)
                assert abs(H[i, j] - fd) < 0.01, \
                    f"H[{labels[i]},{labels[j]}]={H[i,j]}, fd={fd}"


# ================================================================
#  BOUNDARY MODE TESTS
# ================================================================

class TestBoundaryModes:

    @pytest.fixture
    def grid_3d(self):
        return MockGrid3D(f_quadratic_3d, nx=8, ny=8, nz=8,
                          origin=(0, 0, 0), spacing=(1, 1, 1))

    @pytest.fixture
    def grid_4d(self):
        return MockGrid4D(f_quadratic_4d, nx=8, ny=8, nz=8, nt=8,
                          origin=(0, 0, 0, 0), spacing=(1, 1, 1, 1))

    def test_error_mode_raises_3d(self, grid_3d):
        interp = InterpolatorFast(grid_3d, boundary="error",
                                  scheme="trilinear")
        with pytest.raises(ValueError, match="outside grid"):
            interp.interpolate(np.array([-1.0, 3.0, 3.0]), "Phi")

    def test_error_mode_raises_4d(self, grid_4d):
        interp = InterpolatorFast(grid_4d, boundary="error",
                                  scheme="trilinear")
        with pytest.raises(ValueError, match="outside grid"):
            interp.interpolate(np.array([3.0, 3.0, 3.0]), "Phi", t=-1.0)

    def test_clamp_mode_3d(self, grid_3d):
        interp = InterpolatorFast(grid_3d, boundary="clamp",
                                  scheme="trilinear")
        # Should not raise even outside
        val = interp.interpolate(np.array([-1.0, 3.0, 3.0]), "Phi")
        # Clamped to edge -> should equal f(0, 3, 3)
        assert abs(val - f_quadratic_3d(0, 3, 3)) < 1.0  # approximate

    def test_clamp_mode_4d(self, grid_4d):
        interp = InterpolatorFast(grid_4d, boundary="clamp",
                                  scheme="trilinear")
        val = interp.interpolate(np.array([3.0, 3.0, 3.0]), "Phi",
                                 t=-1.0)
        assert np.isfinite(val)

    def test_periodic_3d(self, grid_3d):
        interp = InterpolatorFast(grid_3d, boundary="periodic",
                                  scheme="trilinear")
        # Value at x and x+L should be the same
        pos1 = np.array([2.5, 3.5, 4.5])
        pos2 = np.array([2.5 + 8, 3.5 + 8, 4.5 + 8])
        v1 = interp.interpolate(pos1, "Phi")
        v2 = interp.interpolate(pos2, "Phi")
        assert abs(v1 - v2) < 1e-10


# ================================================================
#  BACKWARD-COMPATIBLE API TESTS
# ================================================================

class TestBackwardCompat:
    """Ensure old callers' API signature still works."""

    @pytest.fixture
    def grid_4d(self):
        return MockGrid4D(f_quadratic_4d, nx=12, ny=12, nz=12, nt=12,
                          origin=(0, 0, 0, 0), spacing=(1, 1, 1, 1))

    def test_value_gradient_and_time_derivative_signature(self, grid_4d):
        """Callers do: val, grad, dtd = interp.value_gradient_and_time_derivative(pos, field, t)
        where grad = (gx, gy, gz) and dtd = df/dt."""
        interp = InterpolatorFast(grid_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        t = 6.0
        val, grad_phi, phi_dot = interp.value_gradient_and_time_derivative(
            pos, "Phi", t)

        # Check types
        assert isinstance(val, float)
        assert isinstance(grad_phi, tuple) and len(grad_phi) == 3
        assert isinstance(phi_dot, float)

        # Check values (quadratic field -> exact)
        expected_val = f_quadratic_4d(3, 4, 5, 6)
        ex, ey, ez, et_ = grad_quadratic_4d(3, 4, 5, 6)

        assert abs(val - expected_val) < 1e-8
        gx, gy, gz = grad_phi
        assert abs(gx - ex) < 1e-8
        assert abs(gy - ey) < 1e-8
        assert abs(gz - ez) < 1e-8
        assert abs(phi_dot - et_) < 1e-8

    def test_caller_pattern_phi_SI(self, grid_4d):
        """Simulates: phi_SI, _, _ = interp.value_gradient_and_time_derivative(pos, "Phi", eta)"""
        interp = InterpolatorFast(grid_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        phi_SI, _, _ = interp.value_gradient_and_time_derivative(
            pos, "Phi", 6.0)
        assert np.isfinite(phi_SI)

    def test_caller_pattern_unpack(self, grid_4d):
        """Simulates: phi, grad_phi, phi_dot = interp.value_gradient_and_time_derivative(...)"""
        interp = InterpolatorFast(grid_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        phi, grad_phi, phi_dot = interp.value_gradient_and_time_derivative(
            pos, "Phi", 6.0)
        gx, gy, gz = grad_phi
        assert np.isfinite(gx) and np.isfinite(gy) and np.isfinite(gz)
        assert np.isfinite(phi_dot)

    def test_value_gradient_hessian_and_time_derivative(self, grid_4d):
        interp = InterpolatorFast(grid_4d, boundary="clamp",
                                  scheme="tricubic")
        pos = np.array([3.0, 4.0, 5.0])
        val, grad3, hess3, dtd = \
            interp.value_gradient_hessian_and_time_derivative(pos, "Phi", 6.0)
        assert len(grad3) == 3
        assert len(hess3) == 6  # hxx, hyy, hzz, hxy, hxz, hyz
        assert np.isfinite(dtd)


# ================================================================
#  SCHEME CONSISTENCY TESTS
# ================================================================

class TestSchemeConsistency:
    """Both schemes should agree on a linear field (where both are exact)."""

    @pytest.fixture
    def grid_linear_4d(self):
        def f(X, Y, Z, T): return 2*X + 3*Y + 5*Z + 7*T
        return MockGrid4D(f, nx=12, ny=12, nz=12, nt=12,
                          origin=(0, 0, 0, 0), spacing=(1, 1, 1, 1))

    def test_trilinear_and_tricubic_agree_on_linear(self, grid_linear_4d):
        interp_lin = InterpolatorFast(grid_linear_4d, boundary="clamp",
                                      scheme="trilinear")
        interp_cub = InterpolatorFast(grid_linear_4d, boundary="clamp",
                                      scheme="tricubic")
        pos = np.array([4.3, 5.7, 3.1])
        t = 6.2

        v1 = interp_lin.interpolate(pos, "Phi", t=t)
        v2 = interp_cub.interpolate(pos, "Phi", t=t)
        expected = 2*4.3 + 3*5.7 + 5*3.1 + 7*6.2
        assert abs(v1 - expected) < 1e-10
        assert abs(v2 - expected) < 1e-10

        g1 = interp_lin.gradient4d(pos, "Phi", t=t)
        g2 = interp_cub.gradient4d(pos, "Phi", t=t)
        for i, expected_gi in enumerate([2, 3, 5, 7]):
            assert abs(g1[i] - expected_gi) < 1e-10
            assert abs(g2[i] - expected_gi) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
