import numpy as np
import importlib.util
from pathlib import Path

from excalibur.core.constants import one_Msun
from excalibur.objects.equivalent_mass_profiles import (
    available_equivalent_mass_presets,
    build_equivalent_mass_profile,
    science_ready_triaxial_preset_suite,
)
from excalibur.objects.nfw_halo import TriaxialNFWHalo


_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_lensing_equivalent_mass_profiles",
    Path(__file__).resolve().parents[1] / "_excalibur_runs" / "run_lensing_equivalent_mass_profiles.py",
)
_RUNNER_MODULE = importlib.util.module_from_spec(_RUNNER_SPEC)
assert _RUNNER_SPEC.loader is not None
_RUNNER_SPEC.loader.exec_module(_RUNNER_MODULE)


def _standard_basis():
    return {
        "center": np.zeros(3, dtype=float),
        "los_dir": np.array([0.0, 0.0, 1.0], dtype=float),
        "e_perp1": np.array([1.0, 0.0, 0.0], dtype=float),
        "e_perp2": np.array([0.0, 1.0, 0.0], dtype=float),
    }


def test_available_presets_include_direct_triaxial_profiles():
    presets = available_equivalent_mass_presets()
    assert "triaxial_nfw_los_minor" in presets
    assert "triaxial_nfw_los_major" in presets
    assert "triaxial_nfw_oblique" in presets
    assert "triaxial_nfw_science_mild_los_minor" in presets
    assert "triaxial_nfw_science_mild_los_major" in presets
    assert "triaxial_nfw_science_fiducial_oblique" in presets
    assert "triaxial_nfw_science_strong_los_minor" in presets
    assert "triaxial_nfw_science_strong_los_major" in presets
    assert "triaxial_nfw_science_cigar_los_major" in presets


def test_science_ready_triaxial_suite_is_curated_and_stable():
    assert science_ready_triaxial_preset_suite() == [
        "single_nfw",
        "triaxial_nfw_science_mild_los_minor",
        "triaxial_nfw_science_mild_los_major",
        "triaxial_nfw_science_fiducial_oblique",
        "triaxial_nfw_science_strong_los_minor",
        "triaxial_nfw_science_strong_los_major",
        "triaxial_nfw_science_cigar_los_major",
    ]


def test_build_direct_triaxial_profile_delegates_to_triaxial_halo():
    basis = _standard_basis()
    source = build_equivalent_mass_profile(
        "triaxial_nfw_los_major",
        total_mass_msun=1.4e15,
        c_nfw=6.5,
        center=basis["center"],
        los_dir=basis["los_dir"],
        e_perp1=basis["e_perp1"],
        e_perp2=basis["e_perp2"],
        shape_scale_mpc=1.0,
    )

    components = source.component_sources()
    assert len(components) == 1
    assert isinstance(components[0], TriaxialNFWHalo)
    assert source.supports_numba_nfw_bypass is False
    assert source.supports_numba_specialized_bypass is True

    major_axis_world = components[0].rotation_matrix[:, 0]
    assert abs(np.dot(major_axis_world, basis["los_dir"])) > 0.99

    point = np.array([0.2, -0.1, 0.3]) * components[0].r_s
    phi = source.potential(*point)
    grad = np.array(source.potential_gradient(*point))
    hess = source.potential_hessian(*point)

    assert np.isfinite(phi)
    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))
    np.testing.assert_allclose(hess, hess.T, rtol=0.0, atol=1e-30)


def test_direct_triaxial_profile_accepts_axis_ratio_override():
    basis = _standard_basis()
    source = build_equivalent_mass_profile(
        "triaxial_nfw_oblique",
        total_mass_msun=1.1e15,
        c_nfw=5.8,
        center=basis["center"],
        los_dir=basis["los_dir"],
        e_perp1=basis["e_perp1"],
        e_perp2=basis["e_perp2"],
        shape_scale_mpc=1.0,
        triaxial_axis_ratios=(0.83, 0.67),
    )

    halo = source.component_sources()[0]
    assert isinstance(halo, TriaxialNFWHalo)
    np.testing.assert_allclose(halo.axis_ratios, (0.83, 0.67), rtol=0.0, atol=1e-12)
    assert np.isclose(source.M_200 / one_Msun, 1.1e15)


def test_runner_numba_guard_allows_direct_triaxial_on_specialized_dopri5_only():
    basis = _standard_basis()
    direct_source = build_equivalent_mass_profile(
        "triaxial_nfw_los_minor",
        total_mass_msun=1.0e15,
        c_nfw=6.0,
        center=basis["center"],
        los_dir=basis["los_dir"],
        e_perp1=basis["e_perp1"],
        e_perp2=basis["e_perp2"],
        shape_scale_mpc=1.0,
    )
    composite_source = build_equivalent_mass_profile(
        "single_nfw",
        total_mass_msun=1.0e15,
        c_nfw=6.0,
        center=basis["center"],
        los_dir=basis["los_dir"],
        e_perp1=basis["e_perp1"],
        e_perp2=basis["e_perp2"],
        shape_scale_mpc=1.0,
    )

    assert _RUNNER_MODULE._source_supports_numba_bypass(direct_source) is False
    assert _RUNNER_MODULE._source_supports_numba_bypass(composite_source) is True
    assert _RUNNER_MODULE._source_supports_numba_path(
        direct_source,
        kernel_name="specialized",
        integrator_name="dopri5",
    ) is True
    assert _RUNNER_MODULE._source_supports_numba_path(
        direct_source,
        kernel_name="specialized",
        integrator_name="rk4",
    ) is False
    assert _RUNNER_MODULE._source_supports_numba_path(
        direct_source,
        kernel_name="standard",
        integrator_name="dopri5",
    ) is False
    assert _RUNNER_MODULE._source_supports_numba_path(
        composite_source,
        kernel_name="standard",
        integrator_name="rk4",
    ) is True
    assert len(_RUNNER_MODULE._source_components_for_metadata(direct_source)) == 1


def test_runner_expands_science_ready_suite_alias():
    expanded = _RUNNER_MODULE._expand_profile_presets(["science_ready_triaxial"])
    assert expanded == science_ready_triaxial_preset_suite()


def test_science_ready_major_axis_preset_aligns_with_line_of_sight():
    basis = _standard_basis()
    source = build_equivalent_mass_profile(
        "triaxial_nfw_science_strong_los_major",
        total_mass_msun=1.3e15,
        c_nfw=6.0,
        center=basis["center"],
        los_dir=basis["los_dir"],
        e_perp1=basis["e_perp1"],
        e_perp2=basis["e_perp2"],
        shape_scale_mpc=1.0,
    )

    halo = source.component_sources()[0]
    assert isinstance(halo, TriaxialNFWHalo)
    np.testing.assert_allclose(halo.axis_ratios, (0.70, 0.46), rtol=0.0, atol=1e-12)
    major_axis_world = halo.rotation_matrix[:, 0]
    assert abs(np.dot(major_axis_world, basis["los_dir"])) > 0.99


def test_science_ready_cigar_preset_is_axisymmetric_and_los_aligned():
    basis = _standard_basis()
    source = build_equivalent_mass_profile(
        "triaxial_nfw_science_cigar_los_major",
        total_mass_msun=1.3e15,
        c_nfw=6.0,
        center=basis["center"],
        los_dir=basis["los_dir"],
        e_perp1=basis["e_perp1"],
        e_perp2=basis["e_perp2"],
        shape_scale_mpc=1.0,
    )

    halo = source.component_sources()[0]
    assert isinstance(halo, TriaxialNFWHalo)
    np.testing.assert_allclose(halo.axis_ratios, (0.55, 0.55), rtol=0.0, atol=1e-12)
    major_axis_world = halo.rotation_matrix[:, 0]
    assert abs(np.dot(major_axis_world, basis["los_dir"])) > 0.99