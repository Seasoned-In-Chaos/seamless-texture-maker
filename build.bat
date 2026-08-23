@echo off
setlocal enabledelayedexpansion

:: ── Configuration ──────────────────────────────────────────────────────
set "PY_VER=3.11+"
set "VENV_DIR=.venv-build"

for /f "tokens=*" %%v in ('python -c "import sys; sys.path.insert(0, '.'); from app.utils.config import APP_VERSION; print(APP_VERSION)" 2^>nul') do set "APP_VER=%%v"
if not defined APP_VER set "APP_VER=unknown"

echo ========================================
echo  SEAMS v%APP_VER% - Production Build
echo ========================================
echo.

:: ── Step 1: Find Python ────────────────────────────────────────────
echo [1/6] Finding Python...
for /f "tokens=*" %%p in ('python -c "import sys; print(sys.executable)"') do set "PY_EXE=%%p"

if not defined PY_EXE (
    echo [ERROR] Python not found!
    exit /b 1
)
echo [OK] Python: %PY_EXE%
echo.

:: ── Step 2: Create / update build venv ──────────────────────────────────
echo [2/6] Setting up build virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    "%PY_EXE%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create venv!
        exit /b 1
    )
)
call "%VENV_DIR%\Scripts\activate.bat"
echo [OK] Venv activated: %VIRTUAL_ENV%
echo.

:: ── Step 3: Install dependencies ────────────────────────────────────────
echo [3/6] Installing Python dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
python -m pip install --quiet pyinstaller packaging
echo [OK] Dependencies installed.
echo.

:: ── Step 4: PyInstaller ────────────────────────────────────────────────
echo [4/6] Building SEAMS.exe with PyInstaller...
if exist "dist\SEAMS.exe" del /f "dist\SEAMS.exe"
if exist "build" rmdir /s /q "build"

where upx >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%u in ('where upx') do set "UPX=%%u"
    echo [OK] UPX found: %UPX%
)

python -m PyInstaller build.spec --clean --noconfirm
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed!
    exit /b 1
)
echo [OK] SEAMS.exe built: dist\SEAMS.exe
echo.

:: ── Step 5: Code sign (optional) ───────────────────────────────────────
echo [5/6] Code signing...
if not defined SIGN_PFX (
    echo [SKIP] Code signing skipped (SIGN_PFX not set^).
    goto :skip_sign
)
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a /f "%SIGN_PFX%" /p "%SIGN_PASS%" "dist\SEAMS.exe"
if errorlevel 1 (
    echo [ERROR] Code signing failed! A SIGN_PFX was configured, so an
    echo         unsigned release build is not shipped silently.
    exit /b 1
)
echo [OK] EXE signed.
:skip_sign
echo.

:: ── Step 6: Inno Setup installer ────────────────────────────────────────
echo [6/6] Building Windows installer...
set "ISCC_EXE="

where iscc >nul 2>&1
if not errorlevel 1 set "ISCC_EXE=iscc"

if not defined ISCC_EXE if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "ISCC_EXE=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not defined ISCC_EXE if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"

if not defined ISCC_EXE (
    echo [WARNING] Inno Setup not found. Skipping installer.
    echo Install: winget install JRSoftware.InnoSetup
    goto :done
)

"%ISCC_EXE%" installer.iss
if errorlevel 1 (
    echo [ERROR] Installer build failed!
    goto :done
)

echo [OK] Installer created in dist\

if not defined SIGN_PFX goto :done
echo [*] Signing installer...
for %%f in (dist\SEAMS_Setup_*.exe) do (
    signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a /f "%SIGN_PFX%" /p "%SIGN_PASS%" "%%f"
    if errorlevel 1 (
        echo [ERROR] Installer signing failed!
        exit /b 1
    )
)

:done
echo.
echo ========================================
echo  BUILD COMPLETE!
echo  EXE:        dist\SEAMS.exe
echo  Installer:  dist\SEAMS_Setup_%APP_VER%.exe
echo ========================================

call deactivate
explorer dist
