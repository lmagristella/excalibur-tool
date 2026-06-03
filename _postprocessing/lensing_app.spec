# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone interactive lensing viewer.

Build (from the project root, with the venv active):
    pyinstaller _postprocessing/lensing_app.spec

Produces a single-file executable in ./dist/ :
    * Linux   -> dist/lensing_app
    * Windows -> dist/lensing_app.exe   (build this ON a Windows machine)

PyInstaller cannot cross-compile: run it on the OS you want to target.
"""
import os

# SPECPATH is the directory containing this .spec file (_postprocessing).
PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
POSTPROC = os.path.join(PROJECT_ROOT, "_postprocessing")

# We bundle the compact *viewer caches* (a few MB each: the precomputed fine-grid
# mapping + scalar halo params) rather than the full multi-hundred-MB simulation
# .npz files. Build them first with:
#     python _postprocessing/build_profile_caches.py _data/output/cache <full1.npz> ...
# The three profiles are: spherical NFW, on-sky-elliptical triaxial, and a
# cigar/prolate halo whose long axis points along the line of sight.
_CACHE_DIR = os.path.join(PROJECT_ROOT, "_data", "output", "cache")
_CACHE_NAMES = [
    "lensing_profile_cache_spherical.npz",            # primary profile
    "lensing_profile_cache_elliptical_ba0p60.npz",    # prolate broadside (ellipse)
    "lensing_profile_cache_inclined_45deg_prolate.npz",  # prolate tilted 45 deg
    "lensing_profile_cache_cigar_par_los_ba0p50.npz",    # prolate end-on (circle)
    "lensing_profile_cache_triaxial_random_orient.npz",  # 3 distinct axes, random orient
]

datas = []
for _name in _CACHE_NAMES:
    _src = os.path.join(_CACHE_DIR, _name)
    if os.path.isfile(_src):
        datas.append((_src, "."))
    elif _name == _CACHE_NAMES[0]:
        raise SystemExit(
            f"Primary profile cache not found:\n  {_src}\n"
            "Run build_profile_caches.py first (see comment above)."
        )
    else:
        print(f"[spec] WARNING: profile cache not found, that profile is skipped:\n  {_src}")

a = Analysis(
    [os.path.join(POSTPROC, "lensing_app_launcher.py")],
    pathex=[POSTPROC, PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "make_lensed_sky_raytracing_interactive",
        "lensed_sky_interactive_common",
        "make_lensed_sky",
        "make_lensed_sky_raytracing",
        "excalibur.io.filename_utils",
        "excalibur.core.constants",
        # matplotlib loads its GUI backend dynamically via importlib, so the
        # static analysis misses it -> force the Tk backend + tkinter in.
        "matplotlib.backends.backend_tkagg",
        "matplotlib.backends.backend_agg",
        "tkinter",
        "tkinter.filedialog",
        # Pillow's Tk image bridge used by backend_tkagg; loaded indirectly.
        "PIL.ImageTk",
        "PIL._tkinter_finder",
    ],
    hookspath=[],
    runtime_hooks=[],
    # Heavy/unused libs the viewer never imports at runtime: keep them out so
    # the binary stays as small as possible.
    excludes=["numba", "llvmlite", "jax", "jaxlib", "torch", "tensorflow",
              "PyQt5", "PyQt6", "PySide2", "PySide6", "IPython", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="excalibur_lensing_applet",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,   # no terminal window; set True to see logs/errors
)
