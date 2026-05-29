# Packaging the interactive lensing viewer as a standalone executable

The viewer is frozen with [PyInstaller] into a **single-file** executable that
bundles the demo `.npz` dataset. End users just double-click it — no Python,
no venv, no tests.

## Files
- `lensing_app_launcher.py` — frozen entry point (forces TkAgg backend, finds
  the bundled dataset, launches the viewer in looping-demo mode).
- `lensing_app.spec` — shared PyInstaller recipe (Linux + Windows).
- `build_lensing_app.sh` / `build_lensing_app.bat` — one-command build helpers.

## Linux build
```bash
source .venv/bin/activate
./_postprocessing/build_lensing_app.sh
# -> dist/lensing_app   (~150 MB, double-clickable)
```

## Windows build (.exe)
PyInstaller **cannot cross-compile**: a Windows `.exe` must be built *on
Windows*. On a Windows machine with Python 3.12 and the project installed:
```bat
py -m venv .venv
.venv\Scripts\activate
pip install numpy scipy matplotlib pyinstaller
pip install -e .            REM installs the excalibur package
_postprocessing\build_lensing_app.bat
REM -> dist\lensing_app.exe
```
(No Wine setup is provided here; building natively on Windows is the reliable
path.)

## Notes / knobs
- **Dataset**: bundled via `--add-data`. To swap it, edit `_DATA_NAME` in both
  `lensing_app_launcher.py` and `lensing_app.spec`.
- **Console window**: `console=False` in the spec hides the terminal. Set it to
  `True` temporarily if you need to see errors/logs while debugging.
- **Custom input**: the binary still forwards CLI args, e.g.
  `./lensing_app /path/to/other_run.npz` or `./lensing_app --panels-only`.
- **Excluded libs**: numba/jax/torch/Qt etc. are excluded — the viewer only
  needs numpy/scipy/matplotlib/tkinter at runtime.
- **MP4 export** (`--mp4-out`) additionally requires `ffmpeg` on PATH; it is not
  bundled. Interactive viewing does not need it.

[PyInstaller]: https://pyinstaller.org
