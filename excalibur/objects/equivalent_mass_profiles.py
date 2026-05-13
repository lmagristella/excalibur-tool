"""Analytical equivalent-mass lens profiles.

This module provides analytical sources that are compatible with
``AnalyticalBypassInterpolator``. The goal is to compare how the same
total mass redistributes convergence and shear once the halo shape is no
longer restricted to a single spherical NFW component.

The shapes implemented here are exact superpositions of spherical NFW
halos, so the total mass is conserved by construction when the component
masses sum to the requested ``M_200``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from excalibur.core.constants import one_Mpc, one_Msun
from excalibur.objects.nfw_halo import NFWHalo


@dataclass(frozen=True)
class ComponentSpec:
    """Mass-fraction and offset for one NFW component."""

    mass_fraction: float
    offset_xyz_mpc: tuple[float, float, float]
    concentration_scale: float = 1.0


class CompositeAnalyticalSource:
    """Analytical source formed by summing several spherical NFW halos."""

    def __init__(self, halos: Sequence[NFWHalo], *, label: str, description: str):
        if not halos:
            raise ValueError("CompositeAnalyticalSource requires at least one halo")
        self.halos = list(halos)
        self.label = label
        self.description = description

        weights = np.array([halo.M_200 for halo in self.halos], dtype=float)
        centers = np.array([halo.center for halo in self.halos], dtype=float)
        self.center = np.average(centers, axis=0, weights=weights)
        self.M_200 = float(np.sum(weights))
        self.component_masses = np.array(weights, dtype=float)
        self.component_centers = np.array(centers, dtype=float)

        # The runner uses these scales to choose a stable integration step and
        # a bypass radius that encloses the whole composite object.
        self.r_s_min = float(min(halo.r_s for halo in self.halos))
        self.r_s = float(max(halo.r_s for halo in self.halos))
        self.R_200 = float(max(np.linalg.norm(halo.center - self.center) + halo.R_200 for halo in self.halos))

    def potential(self, x, y, z):
        return float(sum(halo.potential(x, y, z) for halo in self.halos))

    def potential_gradient(self, x, y, z):
        grad = np.zeros(3, dtype=float)
        for halo in self.halos:
            grad += np.asarray(halo.potential_gradient(x, y, z), dtype=float)
        return tuple(grad)

    def potential_hessian(self, x, y, z):
        hess = np.zeros((3, 3), dtype=float)
        for halo in self.halos:
            hess += np.asarray(halo.potential_hessian(x, y, z), dtype=float)
        return hess

    def mass_weighted_axis_lengths(self):
        deltas = self.component_centers - self.center[None, :]
        weights = self.component_masses / np.sum(self.component_masses)
        cov = np.einsum("n,ni,nj->ij", weights, deltas, deltas)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        return np.sqrt(eigvals)


def parse_component_specs(spec_strings: Sequence[str]) -> list[ComponentSpec]:
    """Parse component specs from CLI strings.

    Format per component:

    ``mass_fraction:dx:dy:dz[:concentration_scale]``

    Offsets are expressed in the local ``(perp1, perp2, los)`` frame and are
    scaled afterward by ``shape_scale_mpc``. Commas are accepted as separators
    too, so both ``0.2:1:0:0:1.1`` and ``0.2,1,0,0,1.1`` are valid.
    """

    specs = []
    for raw_spec in spec_strings:
        fields = raw_spec.replace(",", ":").split(":")
        if len(fields) not in (4, 5):
            raise ValueError(
                "Each custom component must use mass_fraction:dx:dy:dz[:concentration_scale]"
            )
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"Could not parse custom component '{raw_spec}'") from exc

        mass_fraction, dx, dy, dz = values[:4]
        concentration_scale = values[4] if len(values) == 5 else 1.0
        specs.append(
            ComponentSpec(
                mass_fraction=mass_fraction,
                offset_xyz_mpc=(dx, dy, dz),
                concentration_scale=concentration_scale,
            )
        )
    return specs


def _normalized(vector):
    vec = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return vec / norm


def _profile_vectors(los_dir, e_perp1, e_perp2):
    return {
        "los": _normalized(los_dir),
        "perp1": _normalized(e_perp1),
        "perp2": _normalized(e_perp2),
    }


def _validate_axis_scale(axis_scale_xyz):
    axis_scale = np.asarray(axis_scale_xyz, dtype=float)
    if axis_scale.shape != (3,):
        raise ValueError("axis_scale_xyz must contain exactly three values")
    if np.any(axis_scale <= 0.0):
        raise ValueError("axis_scale_xyz must be strictly positive")
    return axis_scale


def _validate_orientation(orientation_euler_deg):
    orientation = np.asarray(orientation_euler_deg, dtype=float)
    if orientation.shape != (3,):
        raise ValueError("orientation_euler_deg must contain exactly three values")
    return orientation


def _rotation_matrix_xyz(orientation_euler_deg):
    ax, ay, az = np.deg2rad(_validate_orientation(orientation_euler_deg))

    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(ax), -np.sin(ax)],
            [0.0, np.sin(ax), np.cos(ax)],
        ],
        dtype=float,
    )
    rot_y = np.array(
        [
            [np.cos(ay), 0.0, np.sin(ay)],
            [0.0, 1.0, 0.0],
            [-np.sin(ay), 0.0, np.cos(ay)],
        ],
        dtype=float,
    )
    rot_z = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rot_z @ rot_y @ rot_x


def _local_basis_matrix(axes):
    return np.column_stack([axes["perp1"], axes["perp2"], axes["los"]])


def _cigar_specs(cigar_axis, cigar_core_fraction, cigar_satellite_concentration_scale):
    axis_to_index = {"perp1": 0, "perp2": 1, "los": 2}
    if cigar_axis not in axis_to_index:
        raise ValueError("cigar_axis must be one of 'perp1', 'perp2', or 'los'")
    if not (0.0 < cigar_core_fraction < 1.0):
        raise ValueError("cigar_core_fraction must lie strictly between 0 and 1")

    side_fraction = 0.5 * (1.0 - cigar_core_fraction)
    negative = np.zeros(3, dtype=float)
    positive = np.zeros(3, dtype=float)
    negative[axis_to_index[cigar_axis]] = -1.0
    positive[axis_to_index[cigar_axis]] = +1.0

    return [
        ComponentSpec(cigar_core_fraction, (0.0, 0.0, 0.0), 1.0),
        ComponentSpec(side_fraction, tuple(negative), cigar_satellite_concentration_scale),
        ComponentSpec(side_fraction, tuple(positive), cigar_satellite_concentration_scale),
    ]


def _preset_specs(
    preset_name,
    *,
    cigar_axis,
    cigar_core_fraction,
    cigar_satellite_concentration_scale,
    custom_component_specs,
):
    base_presets: dict[str, tuple[str, Sequence[ComponentSpec]]] = {
        "single_nfw": (
            "Reference spherical NFW halo.",
            [ComponentSpec(1.0, (0.0, 0.0, 0.0), 1.0)],
        ),
        "los_cigar": (
            "Three aligned NFW components stretched along the line of sight.",
            [
                ComponentSpec(0.50, (0.0, 0.0, 0.0), 1.00),
                ComponentSpec(0.25, (0.0, 0.0, -1.00), 1.15),
                ComponentSpec(0.25, (0.0, 0.0, +1.00), 1.15),
            ],
        ),
        "sky_cigar": (
            "Three aligned NFW components stretched in the image plane.",
            [
                ComponentSpec(0.50, (0.0, 0.0, 0.0), 1.00),
                ComponentSpec(0.25, (-1.00, 0.0, 0.0), 1.15),
                ComponentSpec(0.25, (+1.00, 0.0, 0.0), 1.15),
            ],
        ),
        "triaxial_tilted": (
            "Five-component triaxial cluster tilted with respect to the line of sight.",
            [
                ComponentSpec(0.40, (0.0, 0.0, 0.0), 1.05),
                ComponentSpec(0.18, (+0.90, +0.30, +0.60), 1.15),
                ComponentSpec(0.18, (-0.90, -0.30, -0.60), 1.15),
                ComponentSpec(0.12, (-0.35, +0.95, -0.25), 0.90),
                ComponentSpec(0.12, (+0.35, -0.95, +0.25), 0.90),
            ],
        ),
        "disturbed_cluster": (
            "Dominant core plus asymmetric satellites, mimicking a disturbed cluster.",
            [
                ComponentSpec(0.60, (0.0, 0.0, 0.0), 1.05),
                ComponentSpec(0.15, (+1.10, +0.15, +0.10), 0.95),
                ComponentSpec(0.10, (-0.75, +0.90, -0.20), 1.20),
                ComponentSpec(0.08, (+0.20, -1.05, +0.35), 0.85),
                ComponentSpec(0.07, (-0.25, -0.35, +1.10), 1.35),
            ],
        ),
        "triaxial_parametric": (
            "Symmetric triaxial template. Stretch it with --profile-axis-scale and rotate it with --profile-orientation-deg.",
            [
                ComponentSpec(0.28, (0.0, 0.0, 0.0), 1.05),
                ComponentSpec(0.18, (-1.0, 0.0, 0.0), 1.10),
                ComponentSpec(0.18, (+1.0, 0.0, 0.0), 1.10),
                ComponentSpec(0.14, (0.0, -0.7, 0.0), 0.95),
                ComponentSpec(0.14, (0.0, +0.7, 0.0), 0.95),
                ComponentSpec(0.04, (0.0, 0.0, -0.5), 0.90),
                ComponentSpec(0.04, (0.0, 0.0, +0.5), 0.90),
            ],
        ),
    }

    if preset_name == "cigar_parametric":
        description = (
            f"Parametric three-component cigar aligned with {cigar_axis}; "
            f"core mass fraction = {cigar_core_fraction:.3f}, "
            f"satellite concentration scale = {cigar_satellite_concentration_scale:.3f}."
        )
        return f"cigar_parametric_{cigar_axis}", description, _cigar_specs(
            cigar_axis,
            cigar_core_fraction,
            cigar_satellite_concentration_scale,
        )

    if preset_name == "custom_components":
        if not custom_component_specs:
            raise ValueError(
                "preset 'custom_components' requires at least one --custom-components entry"
            )
        description = (
            f"User-defined composite profile with {len(custom_component_specs)} components in the local "
            "(perp1, perp2, los) frame."
        )
        return preset_name, description, list(custom_component_specs)

    if preset_name not in base_presets:
        valid = ", ".join(sorted(available_equivalent_mass_presets()))
        raise ValueError(f"Unknown profile preset '{preset_name}'. Valid choices: {valid}")

    description, specs = base_presets[preset_name]
    return preset_name, description, specs


def _build_component_halos(
    component_specs: Iterable[ComponentSpec],
    *,
    total_mass_msun: float,
    c_nfw: float,
    center,
    axes,
    axis_scale_xyz,
    orientation_euler_deg,
):
    center = np.asarray(center, dtype=float)
    component_specs = list(component_specs)
    if not component_specs:
        raise ValueError("At least one component specification is required")

    mass_fractions = np.array([spec.mass_fraction for spec in component_specs], dtype=float)
    if np.any(mass_fractions <= 0.0):
        raise ValueError("All component mass fractions must be strictly positive")
    offsets_mpc = np.array([spec.offset_xyz_mpc for spec in component_specs], dtype=float)
    offsets_mpc -= np.average(offsets_mpc, axis=0, weights=mass_fractions)

    axis_scale = _validate_axis_scale(axis_scale_xyz)
    rotation = _rotation_matrix_xyz(orientation_euler_deg)
    basis = _local_basis_matrix(axes)

    halos = []
    for spec, offset_mpc in zip(component_specs, offsets_mpc):
        transformed_offset_mpc = rotation @ (axis_scale * offset_mpc)
        offset = (basis @ transformed_offset_mpc) * one_Mpc
        halos.append(
            NFWHalo(
                spec.mass_fraction * total_mass_msun * one_Msun,
                c_nfw * spec.concentration_scale,
                center + offset,
            )
        )

    mass_sum = sum(halo.M_200 for halo in halos) / (total_mass_msun * one_Msun)
    if not np.isclose(mass_sum, 1.0, atol=1e-10):
        raise ValueError("Component mass fractions must sum to one")
    return halos


def build_equivalent_mass_profile(
    preset_name: str,
    *,
    total_mass_msun: float,
    c_nfw: float,
    center,
    los_dir,
    e_perp1,
    e_perp2,
    shape_scale_mpc: float,
    axis_scale_xyz=(1.0, 1.0, 1.0),
    orientation_euler_deg=(0.0, 0.0, 0.0),
    cigar_axis="los",
    cigar_core_fraction=0.5,
    cigar_satellite_concentration_scale=1.15,
    custom_component_specs: Sequence[ComponentSpec] | None = None,
):
    """Build one analytical source of total mass ``total_mass_msun``.

    Parameters
    ----------
    preset_name
        Name of the mass-shape preset.
    shape_scale_mpc
        Global offset scale used for the non-spherical presets.
    """

    axes = _profile_vectors(los_dir, e_perp1, e_perp2)

    label, description, specs = _preset_specs(
        preset_name,
        cigar_axis=cigar_axis,
        cigar_core_fraction=cigar_core_fraction,
        cigar_satellite_concentration_scale=cigar_satellite_concentration_scale,
        custom_component_specs=custom_component_specs,
    )

    axis_scale = _validate_axis_scale(axis_scale_xyz)
    orientation = _validate_orientation(orientation_euler_deg)
    scaled_specs = [
        ComponentSpec(
            mass_fraction=spec.mass_fraction,
            offset_xyz_mpc=tuple(shape_scale_mpc * np.asarray(spec.offset_xyz_mpc, dtype=float)),
            concentration_scale=spec.concentration_scale,
        )
        for spec in specs
    ]
    halos = _build_component_halos(
        scaled_specs,
        total_mass_msun=total_mass_msun,
        c_nfw=c_nfw,
        center=center,
        axes=axes,
        axis_scale_xyz=axis_scale,
        orientation_euler_deg=orientation,
    )

    transform_notes = []
    if not np.allclose(axis_scale, 1.0):
        transform_notes.append(
            f"axis_scale(local perp1, perp2, los) = ({axis_scale[0]:.3f}, {axis_scale[1]:.3f}, {axis_scale[2]:.3f})"
        )
    if not np.allclose(orientation, 0.0):
        transform_notes.append(
            f"orientation_deg(Rx, Ry, Rz) = ({orientation[0]:.1f}, {orientation[1]:.1f}, {orientation[2]:.1f})"
        )
    if transform_notes:
        description = f"{description} {'; '.join(transform_notes)}."

    return CompositeAnalyticalSource(
        halos,
        label=label,
        description=description,
    )


def available_equivalent_mass_presets():
    return [
        "single_nfw",
        "los_cigar",
        "sky_cigar",
        "cigar_parametric",
        "triaxial_tilted",
        "triaxial_parametric",
        "disturbed_cluster",
        "custom_components",
    ]