#!/usr/bin/env python3
"""Standalone entry point for the packaged interactive lensing viewer.

This wrapper is what PyInstaller freezes. It:
  * forces a GUI matplotlib backend (TkAgg) so a window opens on double-click,
  * locates the demo .npz whether running from source or from a frozen bundle,
  * launches the interactive raytracing viewer with no further prompts.

Any extra command-line arguments are forwarded to the underlying viewer, so
the packaged binary still accepts e.g. a custom .npz path or viewer flags.
"""
from __future__ import annotations

import os
import sys

# Force an interactive GUI backend before matplotlib is imported anywhere.
# lensed_sky_interactive_common reads this env var to pick the backend.
os.environ.setdefault("EXCALIBUR_MPL_BACKEND", "TkAgg")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Bundled viewer caches (kept in sync with lensing_app.spec). The first is the
# primary profile; the rest become selectable lens profiles via the in-app radio
# switcher: spherical, on-sky-elliptical, and a cigar aligned with the line of
# sight (circular projection but stronger lensing).
_CACHE_PRIMARY = "lensing_profile_cache_spherical.npz"
_CACHE_EXTRAS = [
    "lensing_profile_cache_elliptical_ba0p60.npz",       # prolate broadside (ellipse)
    "lensing_profile_cache_inclined_45deg_prolate.npz",  # prolate tilted 45 deg
    "lensing_profile_cache_cigar_par_los_ba0p50.npz",    # prolate end-on (circle)
    "lensing_profile_cache_triaxial_random_orient.npz",  # 3 distinct axes, random orient
]


def _bundle_dir() -> str:
    """Directory holding bundled data: _MEIPASS when frozen, else this dir."""
    return getattr(sys, "_MEIPASS", _THIS_DIR)


def _resolve_dataset(name: str) -> str | None:
    """Find a bundled .npz across frozen-bundle and source-tree layouts."""
    candidates = [
        os.path.join(_bundle_dir(), name),
        os.path.join(_bundle_dir(), "_data", "output", name),
        os.path.join(_bundle_dir(), "_data", "output", "cache", name),
        os.path.normpath(os.path.join(_THIS_DIR, "..", "_data", "output", name)),
        os.path.normpath(os.path.join(_THIS_DIR, "..", "_data", "output", "cache", name)),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def main() -> None:
    import make_lensed_sky_raytracing_interactive as viewer

    user_args = sys.argv[1:]
    argv = [sys.argv[0]]

    # If the user passed their own path, respect it; otherwise inject the bundled
    # profile caches: the spherical one as primary plus the elliptical and cigar
    # ones as selectable profiles (in-app radio switcher). We launch in fully
    # interactive mode (draggable source + sliders), NOT the auto-playing demo
    # animation. The user can still opt into the animation via --animate.
    user_gave_path = bool(user_args) and not user_args[0].startswith("-")
    if not user_gave_path:
        primary = _resolve_dataset(_CACHE_PRIMARY)
        if primary is not None:
            argv.append(primary)
        for extra in _CACHE_EXTRAS:
            path = _resolve_dataset(extra)
            if path is not None:
                argv.extend(["--add-profile", path])
    argv.extend(user_args)

    sys.argv = argv
    viewer.main()


if __name__ == "__main__":
    main()
