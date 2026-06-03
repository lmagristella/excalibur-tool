@echo off
REM Build the standalone Windows .exe for the interactive lensing viewer.
REM Run this ON A WINDOWS MACHINE, from the project root, in a venv that has
REM numpy / scipy / matplotlib / the excalibur package installed:
REM     _postprocessing\build_lensing_app.bat
REM Output: dist\lensing_app.exe  (single-file, double-clickable)
setlocal
cd /d "%~dp0\.."

python -c "import PyInstaller" 2>NUL || pip install pyinstaller

pyinstaller --noconfirm --distpath dist --workpath build\pyi ^
    _postprocessing\lensing_app.spec

echo.
echo Done -^> %CD%\dist\lensing_app.exe
endlocal
