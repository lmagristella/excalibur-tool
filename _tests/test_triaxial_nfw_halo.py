import numpy as np
import pytest
from numba.typed import List

from excalibur.core.constants import one_Msun
from excalibur.integration.integrator_numba import (
    _BYPASS_MODE_TRIAXIAL_NFW,
    _value_gradient_hessian_with_optional_bypass,
)
from excalibur.objects.nfw_halo import NFWHalo, TriaxialNFWHalo


@pytest.fixture(scope="module")
def spherical_pair():
    center = np.zeros(3, dtype=float)
    mass = 2.0e15 * one_Msun
    spherical = NFWHalo(M_200=mass, c_NFW=7.0, center=center)
    triaxial = TriaxialNFWHalo(
        M_200=mass,
        c_NFW=7.0,
        center=center,
        axis_ratios=(1.0, 1.0),
        orientation_euler_deg=(19.0, 31.0, 47.0),
    )
    return spherical, triaxial


@pytest.fixture(scope="module")
def triaxial_halo():
    center = np.array([0.05, -0.03, 0.02]) * 1.0e22
    return TriaxialNFWHalo(
        M_200=1.6e15 * one_Msun,
        c_NFW=5.5,
        center=center,
        axis_ratios=(0.78, 0.56),
        orientation_euler_deg=(23.0, -17.0, 41.0),
    )


def test_triaxial_nfw_matches_spherical_limit(spherical_pair):
    spherical, triaxial = spherical_pair
    rs = spherical.r_s
    points = [
        np.array([0.3 * rs, 0.0, 0.0]),
        np.array([1.2 * rs, -0.4 * rs, 0.7 * rs]),
        np.array([2.0 * rs, 0.5 * rs, -0.8 * rs]),
    ]

    for point in points:
        np.testing.assert_allclose(
            triaxial.potential(*point),
            spherical.potential(*point),
            rtol=2e-6,
            atol=0.0,
        )
        np.testing.assert_allclose(
            triaxial.potential_gradient(*point),
            spherical.potential_gradient(*point),
            rtol=2e-6,
            atol=1e-16,
        )
        np.testing.assert_allclose(
            triaxial.potential_hessian(*point),
            spherical.potential_hessian(*point),
            rtol=1e-12,
            atol=1e-30,
        )


def test_triaxial_gradient_matches_finite_difference(triaxial_halo):
    rs = triaxial_halo.r_s
    point = triaxial_halo.center + np.array([1.1, -0.7, 0.45]) * rs
    step = 2e-4 * rs

    gradient = np.array(triaxial_halo.potential_gradient(*point))
    gradient_fd = np.empty(3, dtype=float)
    for axis_index in range(3):
        plus = point.copy()
        minus = point.copy()
        plus[axis_index] += step
        minus[axis_index] -= step
        gradient_fd[axis_index] = (
            float(triaxial_halo.potential(*plus))
            - float(triaxial_halo.potential(*minus))
        ) / (2.0 * step)

    scale = np.max(np.abs(gradient_fd))
    np.testing.assert_allclose(gradient, gradient_fd, rtol=0.0, atol=5e-4 * scale)


def test_triaxial_hessian_matches_finite_difference(triaxial_halo):
    rs = triaxial_halo.r_s
    point = triaxial_halo.center + np.array([1.35, -0.55, 0.8]) * rs
    step = 2e-4 * rs

    hessian = triaxial_halo.potential_hessian(*point)
    hessian_fd = np.empty((3, 3), dtype=float)
    for axis_index in range(3):
        plus = point.copy()
        minus = point.copy()
        plus[axis_index] += step
        minus[axis_index] -= step
        grad_plus = np.array(triaxial_halo.potential_gradient(*plus))
        grad_minus = np.array(triaxial_halo.potential_gradient(*minus))
        hessian_fd[:, axis_index] = (grad_plus - grad_minus) / (2.0 * step)

    np.testing.assert_allclose(hessian, hessian_fd, rtol=5e-4, atol=0.0)


def test_numba_triaxial_bypass_matches_python_object():
    halo = TriaxialNFWHalo(
        M_200=1.2e15 * one_Msun,
        c_NFW=6.2,
        center=np.array([0.1, -0.2, 0.3]) * 1.0e22,
        axis_ratios=(0.67, 0.49),
        orientation_euler_deg=(17.0, -28.0, 39.0),
        quadrature_order=32,
    )

    point = halo.center + np.array([0.42, -0.31, 0.27]) * halo.r_s
    hessian = halo.potential_hessian(*point)
    python_eval = np.array(
        [
            halo.potential(*point),
            *halo.potential_gradient(*point),
            hessian[0, 0],
            hessian[1, 1],
            hessian[2, 2],
            hessian[0, 1],
            hessian[0, 2],
            hessian[1, 2],
        ],
        dtype=float,
    )

    origins = np.zeros((1, 3), dtype=np.float64)
    uppers = np.ones((1, 3), dtype=np.float64)
    spacings = np.ones((1, 3), dtype=np.float64)
    shapes = np.full((1, 3), 4, dtype=np.int64)
    fields = List()
    fields.append(np.zeros((4, 4, 4), dtype=np.float64))

    packed_matrix = np.vstack(
        [
            np.asarray(halo.rotation_matrix, dtype=np.float64),
            np.asarray(halo._axis_scales_sq, dtype=np.float64).reshape(1, 3),
        ]
    )
    packed_nodes = np.concatenate(
        [
            np.array(
                [
                    halo._xi_center,
                    halo._inv_delta_integral,
                    halo._m_soft,
                    halo._axis_product,
                    halo.r_s,
                    halo.rho_s,
                ],
                dtype=np.float64,
            ),
            np.asarray(halo._tau_nodes, dtype=np.float64),
        ]
    )
    packed_weights = np.asarray(halo._tau_weights, dtype=np.float64)

    numba_eval = np.array(
        _value_gradient_hessian_with_optional_bypass(
            point[0],
            point[1],
            point[2],
            origins,
            uppers,
            spacings,
            shapes,
            fields,
            1,
            _BYPASS_MODE_TRIAXIAL_NFW,
            np.asarray(halo.center, dtype=np.float64),
            1.0e99,
            packed_matrix,
            packed_nodes,
            packed_weights,
        ),
        dtype=float,
    )

    np.testing.assert_allclose(numba_eval, python_eval, rtol=5e-15, atol=1e-30)