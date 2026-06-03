#!/usr/bin/env bash
# Package the PyInstaller binary (dist/lensing_app) into a portable AppImage.
# Prereqs: dist/lensing_app built, _packaging/appimagetool + lensing_app.png present.
# Run from the project root:  ./_packaging/build_appimage.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PKG="_packaging"
APPDIR="$PKG/LensingApp.AppDir"
OUT="dist/Excalibur_Lensing_Applet-x86_64.AppImage"

# Accept either the default PyInstaller name or the renamed binary.
BIN=""
for cand in dist/excalibur_lensing_applet dist/lensing_app; do
    [ -f "$cand" ] && { BIN="$cand"; break; }
done
[ -n "$BIN" ] || { echo "No PyInstaller binary in dist/ (build: pyinstaller _postprocessing/lensing_app.spec)"; exit 1; }
echo "Using binary: $BIN"

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp "$BIN" "$APPDIR/usr/bin/lensing_app"
cp "$PKG/lensing_app.png" "$APPDIR/lensing_app.png"

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "${0}")")"
exec "${HERE}/usr/bin/lensing_app" "$@"
EOF
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/lensing_app.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Interactive Lensing
Comment=Interactive gravitational-lensing explorer (EXCALIBUR)
Exec=lensing_app
Icon=lensing_app
Categories=Science;Education;
Terminal=false
EOF

ARCH=x86_64 "$PKG/appimagetool" --appimage-extract-and-run "$APPDIR" "$OUT"
echo
echo "Done -> $ROOT/$OUT"
