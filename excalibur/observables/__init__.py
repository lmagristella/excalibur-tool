"""
Observable quantities module.

This module contains functions for computing physical observables
from photon trajectories, such as redshift, time delays, lensing effects, etc.
"""

from .redshift import (
    compute_redshift_components,
    compute_total_redshift,
    RedshiftCalculator
)

from .redshift_plots import (
    plot_redshift_evolution,
    plot_redshift_statistics,
    plot_redshift_vs_quantity
)

from .optical_tidal_matrix import (
    optical_tidal_matrix_from_blocks,
    optical_tidal_matrix_optimized,
    jacobi_rhs,
    optical_scalars_from_tidal,
    lensing_from_jacobi,
    angular_diameter_distance_from_jacobi,
    distance_comparison,
)

__all__ = [
    'compute_redshift_components',
    'compute_total_redshift',
    'RedshiftCalculator',
    'plot_redshift_evolution',
    'plot_redshift_statistics',
    'plot_redshift_vs_quantity',
    'optical_tidal_matrix_from_blocks',
    'optical_tidal_matrix_optimized',
    'jacobi_rhs',
    'optical_scalars_from_tidal',
    'lensing_from_jacobi',
    'angular_diameter_distance_from_jacobi',
    'distance_comparison',
]
