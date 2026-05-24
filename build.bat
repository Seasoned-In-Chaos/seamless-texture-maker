@echo off
setlocal enabledelayedexpansion
echo ========================================
echo  SEAMS v3.0 - Production Build
echo ========================================
echo.

:: ── Configuration ──────────────────────────────────────────────────────
set "PY_VER=3.12"
set "VENV_DIR=.venv-build"
set "MSVC_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC"

:: ── Step 1: Find Python 3.12 ────────────────────────────────────────────
echo [1/8] Finding Python %PY_VER%...
set "PY312="

py -%PY_VER% --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%p in ('py -%PY_VER% -c "import sys; print(sys.executable)"') do set "PY312=%%p"
)

if not defined PY312 if exist "%LOCALAPPDATA%\Python\pythoncore-3.12-64\python.exe" set "PY312=%LOCALAPPDATA%\Python\pythoncore-3.12-64\python.exe"
if not defined PY312 if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY312 if exist "C:\Python312\python.exe" set "PY312=C:\Python312\python.exe"

if not defined PY312 (
    echo [ERROR] Python 3.12 not found!
    exit /b 1
)
echo [OK] Python 3.12: %PY312%
echo.

:: ── Step 2: Create / update build venv ──────────────────────────────────
echo [2/8] Setting up build virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    "%PY312%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create venv!
        exit /b 1
    )
)
call "%VENV_DIR%\Scripts\activate.bat"
echo [OK] Venv activated: %VIRTUAL_ENV%
echo.

:: ── Step 3: Install dependencies ────────────────────────────────────────
echo [3/8] Installing Python dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
python -m pip install --quiet maturin pyinstaller packaging
echo [OK] Dependencies installed.
echo.

:: ── Step 4: Verify MSVC toolchain ──────────────────────────────────────
echo [4/8] Verifying MSVC toolchain...
set "LINK_FOUND=0"
for /f "tokens=*" %%d in ('dir /b /ad /o-n "%MSVC_ROOT%" 2^>nul') do (
    if exist "%MSVC_ROOT%\%%d\bin\Hostx64\x64\link.exe" (
        set "LINK_FOUND=1"
        set "MSVC_VER=%%d"
    )
)
if "%LINK_FOUND%"=="0" (
    echo [ERROR] MSVC link.exe not found!
    exit /b 1
)
echo [OK] MSVC %MSVC_VER% found.
echo.

:: ── Step 5: Build Rust extension ────────────────────────────────────────
echo [5/8] Building Rust extension (seams_core)...

set "PYO3_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
for /f "tokens=*" %%s in ('dir /b /ad /o-n "C:\Program Files (x86)\Windows Kits\10\Lib" 2^>nul') do set "SDK_VER=%%s"
set "LIB=%MSVC_ROOT%\%MSVC_VER%\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\%SDK_VER%\um\x64;C:\Program Files (x86)\Windows Kits\10\Lib\%SDK_VER%\ucrt\x64"
set "INCLUDE=%MSVC_ROOT%\%MSVC_VER%\include"

cargo build --release --manifest-path seams_core\Cargo.toml
if errorlevel 1 (
    echo [WARNING] Rust build failed. Continuing with Numba JIT fallback.
    goto :skip_rust_install
)

if not exist "seams_core\target\release\seams_core.dll" (
    echo [WARNING] seams_core.dll not found after build.
    goto :skip_rust_install
)

copy /y "seams_core\target\release\seams_core.dll" "seams_core\target\release\seams_core.pyd" >nul
python -c "import shutil,pathlib;sp=pathlib.Path(r'%VIRTUAL_ENV%','Lib','site-packages');sp.mkdir(exist_ok=True);shutil.copy(r'seams_core/target/release/seams_core.pyd',sp/'seams_core.pyd');print('Installed seams_core.pyd')"
python -c "from seams_core import edge_blend_symmetric;print('  seams_core: VERIFIED OK')" 2>nul || echo [WARNING] seams_core import verification failed
echo [OK] Rust extension built.
:skip_rust_install
echo.

:: ── Step 6: PyInstaller ────────────────────────────────────────────────
echo [6/8] Building SEAMS.exe with PyInstaller...
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

:: ── Step 7: Code sign (optional) ───────────────────────────────────────
echo [7/8] Code signing...
if not defined SIGN_PFX (
    echo [SKIP] Code signing skipped (SIGN_PFX not set^).
    goto :skip_sign
)
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a /f "%SIGN_PFX%" /p "%SIGN_PASS%" "dist\SEAMS.exe"
if errorlevel 1 (
    echo [WARNING] Code signing failed.
) else (
    echo [OK] EXE signed.
)
:skip_sign
echo.

:: ── Step 8: Inno Setup installer ────────────────────────────────────────
echo [8/8] Building Windows installer...
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
)

:done
echo.
echo ========================================
echo  BUILD COMPLETE!
echo  EXE:        dist\SEAMS.exe
echo  Installer:  dist\SEAMS_Setup_3.0.0.exe
echo ========================================

call deactivate
explorer dist
