from __future__ import annotations

PROFILE_COLOR_MAP = {
    "single_nfw": "#000000",
    "triaxial_nfw_science_mild_los_minor": "#ff7f0e",
    "triaxial_nfw_science_mild_los_major": "#d62728",
    "triaxial_nfw_science_fiducial_oblique": "#8c564b",
    "triaxial_nfw_science_strong_los_minor": "#e377c2",
    "triaxial_nfw_science_strong_los_major": "#bcbd22",
    "triaxial_nfw_science_cigar_los_major": "#17becf",
}

PROFILE_MARKER_MAP = {
    "single_nfw": "o",
    "triaxial_nfw_science_mild_los_minor": "s",
    "triaxial_nfw_science_mild_los_major": "^",
    "triaxial_nfw_science_fiducial_oblique": "D",
    "triaxial_nfw_science_strong_los_minor": "P",
    "triaxial_nfw_science_strong_los_major": "X",
    "triaxial_nfw_science_cigar_los_major": "v",
}

PROFILE_TOKEN_ALIASES = {
    "single_nfw": "single_nfw",
    "mild_los_minor": "triaxial_nfw_science_mild_los_minor",
    "mild_los_major": "triaxial_nfw_science_mild_los_major",
    "fiducial_oblique": "triaxial_nfw_science_fiducial_oblique",
    "strong_los_minor": "triaxial_nfw_science_strong_los_minor",
    "strong_los_major": "triaxial_nfw_science_strong_los_major",
    "cigar_los_major": "triaxial_nfw_science_cigar_los_major",
}

PROFILE_DISPLAY_MAP = {
    "single_nfw": "single_nfw",
    "triaxial_nfw_science_mild_los_minor": "mild LOS minor",
    "triaxial_nfw_science_mild_los_major": "mild LOS major",
    "triaxial_nfw_science_fiducial_oblique": "fiducial oblique",
    "triaxial_nfw_science_strong_los_minor": "strong LOS minor",
    "triaxial_nfw_science_strong_los_major": "strong LOS major",
    "triaxial_nfw_science_cigar_los_major": "cigar LOS major",
}

FALLBACK_COLORS = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
)

FALLBACK_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<")


def _canonical_profile_name(profile_name):
    if profile_name in PROFILE_COLOR_MAP:
        return profile_name
    for token, canonical_name in PROFILE_TOKEN_ALIASES.items():
        if token in profile_name:
            return canonical_name
    return profile_name


def equivalent_mass_profile_color(profile_name, fallback_index=0):
    canonical_name = _canonical_profile_name(str(profile_name))
    if canonical_name in PROFILE_COLOR_MAP:
        return PROFILE_COLOR_MAP[canonical_name]
    return FALLBACK_COLORS[int(fallback_index) % len(FALLBACK_COLORS)]


def equivalent_mass_profile_marker(profile_name, fallback_index=0):
    canonical_name = _canonical_profile_name(str(profile_name))
    if canonical_name in PROFILE_MARKER_MAP:
        return PROFILE_MARKER_MAP[canonical_name]
    return FALLBACK_MARKERS[int(fallback_index) % len(FALLBACK_MARKERS)]


def equivalent_mass_profile_style(profile_name, is_reference=False, fallback_index=0):
    color = equivalent_mass_profile_color(profile_name, fallback_index=fallback_index)
    marker = equivalent_mass_profile_marker(profile_name, fallback_index=fallback_index)
    return {
        "color": color,
        "marker": marker,
        "lw": 2.3 if is_reference else 1.7,
        "ms": 4.5 if is_reference else 4.0,
        "alpha": 0.98 if is_reference else 0.92,
        "mec": "#ffffff",
        "mew": 0.55,
        "zorder": 4 if is_reference else 3,
    }


def equivalent_mass_fit_style(profile_name, is_reference=False, fallback_index=0):
    color = equivalent_mass_profile_color(profile_name, fallback_index=fallback_index)
    return {
        "color": color,
        "lw": 2.1 if is_reference else 1.9,
        "alpha": 0.95,
        "linestyle": (0, (4.0, 2.2)),
        "zorder": 2,
    }


def equivalent_mass_profile_display_name(profile_name):
    canonical_name = _canonical_profile_name(str(profile_name))
    if canonical_name in PROFILE_DISPLAY_MAP:
        return PROFILE_DISPLAY_MAP[canonical_name]
    return str(profile_name).replace("triaxial_nfw_science_", "").replace("_", " ")