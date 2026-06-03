#!/usr/bin/env bash
# Build the standalone Linux executable for the interactive lensing viewer.
# Run from the project root with the venv active:
#     ./_postprocessing/build_lensing_app.sh
# Output: dist/lensing_app  (single-file, double-clickable)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -c "import PyInstaller" 2>/dev/null || pip install pyinstaller

pyinstaller --noconfirm --distpath dist --workpath build/pyi \
    _postprocessing/lensing_app.spec

echo
echo "Done -> $ROOT/dist/lensing_app"
