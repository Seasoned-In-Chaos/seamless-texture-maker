@echo off
setlocal enabledelayedexpansion
echo ========================================
echo  SEAMS v3.0 - Microsoft Store (MSIX) Build
echo ========================================
echo.

:: ── Configuration ──────────────────────────────────────────────────────
set "PY_VER=3.12"
set "VENV_DIR=.venv-build"

:: ── Step 1: Find Python 3.12 ────────────────────────────────────────────
echo [1/8] Finding Python %PY_VER%...
set "PY312="

py -%PY_VER% --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%p in ('py -%PY_VER% -c "import sys; print(sys.executable)"') do set "PY312=%%p"
)

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
python -m pip install --quiet pyinstaller packaging
echo [OK] Dependencies installed.
echo.

:: ── Step 4: Check Windows SDK (MakeAppx) ──────────────────────────────
echo [4/8] Checking Windows SDK...
set "MAKEAPPX="
set "SIGNTOOL="

for /f "delims=" %%i in ('dir /b /ad /o-n "C:\Program Files (x86)\Windows Kits\10\bin\10.*" 2^>nul') do (
    if exist "C:\Program Files (x86)\Windows Kits\10\bin\%%i\x64\MakeAppx.exe" (
        set "MAKEAPPX=C:\Program Files (x86)\Windows Kits\10\bin\%%i\x64\MakeAppx.exe"
        set "SIGNTOOL=C:\Program Files (x86)\Windows Kits\10\bin\%%i\x64\signtool.exe"
    )
)

if not defined MAKEAPPX (
    echo [ERROR] Windows SDK not found! Install Windows SDK 10 to get MakeAppx.exe
    call deactivate
    exit /b 1
)
echo [OK] MakeAppx: %MAKEAPPX%
echo.

:: ── Step 5: Build Rust extension (optional) ────────────────────────────
echo [5/8] Building Rust extension (optional)...
where cargo >nul 2>&1
if not errorlevel 1 (
    set "PYO3_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
    cargo build --release --manifest-path seams_core\Cargo.toml
    if not errorlevel 1 (
        if exist "seams_core\target\release\seams_core.dll" (
            copy /y "seams_core\target\release\seams_core.dll" "seams_core\target\release\seams_core.pyd" >nul
            python -c "import shutil,pathlib;sp=pathlib.Path(r'%VIRTUAL_ENV%','Lib','site-packages');sp.mkdir(exist_ok=True);shutil.copy(r'seams_core/target/release/seams_core.pyd',sp/'seams_core.pyd');print('Installed seams_core.pyd')"
            echo [OK] Rust extension built.
        ) else (
            echo [WARNING] seams_core.dll not found. Continuing with Numba JIT fallback.
        )
    ) else (
        echo [WARNING] Rust build failed. Continuing with Numba JIT fallback.
    )
) else (
    echo [SKIP] Cargo not found. Using Numba JIT fallback.
)
echo.

:: ── Step 6: PyInstaller onedir build ────────────────────────────────────
echo [6/8] Building onedir layout (for MSIX)...
if exist "dist\SEAMS" rmdir /s /q "dist\SEAMS"
if exist "build" rmdir /s /q "build"

where upx >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%u in ('where upx') do set "UPX=%%u"
    echo [OK] UPX found: %UPX%
)

python -m PyInstaller build_store.spec --clean --noconfirm
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed!
    call deactivate
    exit /b 1
)
echo [OK] Onedir build complete: dist\SEAMS\
echo.

:: ── Step 7: Assemble MSIX package ──────────────────────────────────────
echo [7/8] Assembling MSIX package...
set "MSIX_STAGING=dist\SEAMS_MSIX"
if exist "%MSIX_STAGING%" rmdir /s /q "%MSIX_STAGING%"
mkdir "%MSIX_STAGING%\images"

copy /y "store\AppxManifest.xml" "%MSIX_STAGING%\"
xcopy /e /i /y "dist\SEAMS" "%MSIX_STAGING%\SEAMS\"
copy /y "store\images\*.png" "%MSIX_STAGING%\images\" 2>nul

if not exist "%MSIX_STAGING%\images\StoreLogo.png" (
    if exist "resources\icon.png" copy /y "resources\icon.png" "%MSIX_STAGING%\images\StoreLogo.png"
)

echo [OK] MSIX staging ready.
echo.

:: ── Step 8: Package & sign MSIX ────────────────────────────────────────
echo [8/8] Packaging MSIX...
"%MAKEAPPX%" pack /d "%MSIX_STAGING%" /p "dist\SEAMS.msix" /v
if errorlevel 1 (
    echo [ERROR] MSIX packaging failed!
    call deactivate
    exit /b 1
)
echo [OK] MSIX package created: dist\SEAMS.msix

set "TEST_CERT_SUBJECT=CN=Shubham Panchasara, O=Seams Studio, C=US"
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "try { (Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Subject -eq '%TEST_CERT_SUBJECT%' } | Select-Object -First 1).Thumbprint } catch { '' }"') do set "CERT_THUMB=%%t"

if defined CERT_THUMB (
    "%SIGNTOOL%" sign /fd SHA256 /sha1 %CERT_THUMB% /a "dist\SEAMS.msix"
    if not errorlevel 1 (
        echo [OK] MSIX signed with test certificate.
    ) else (
        echo [WARNING] MSIX signing failed. You may need to install the cert as a Trusted Root.
    )
) else (
    echo [INFO] No test certificate found. Create one for local testing with:
    echo   powershell: New-SelfSignedCertificate -Type Custom -KeyUsage DigitalSignature -CertStoreLocation 'Cert:\CurrentUser\My' -Subject '%TEST_CERT_SUBJECT%'
)

call deactivate

echo.
echo ========================================
echo  MSIX BUILD COMPLETE!
echo  Output: dist\SEAMS.msix
echo.
echo  To install locally for testing:
echo    powershell: Add-AppxPackage -Path dist\SEAMS.msix
echo.
echo  To submit to the Microsoft Store:
echo    1. Go to https://partner.microsoft.com/dashboard
echo    2. Create new app ^> MSIX upload
echo    3. Upload dist\SEAMS.msix
echo ========================================

explorer dist
