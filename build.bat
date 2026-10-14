@echo off
setlocal enabledelayedexpansion

:: SEAMS production build: tests, release executable, optional code signing
:: and the Windows installer. Run from the repository root.

cd /d "%~dp0"

:: -- Step 1: Find the Rust toolchain ------------------------------------
echo [1/5] Finding the Rust toolchain...
where cargo >nul 2>&1
if errorlevel 1 if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
where cargo >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cargo not found. Install Rust: winget install Rustlang.Rustup
    exit /b 1
)

:: The GNU toolchain links with MinGW tools (dlltool, as, windres). Add a
:: winget-installed WinLibs to PATH when it is not there already.
where dlltool >nul 2>&1
if errorlevel 1 (
    for /d %%d in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs*") do (
        if exist "%%d\mingw64\bin\dlltool.exe" set "PATH=%%d\mingw64\bin;!PATH!"
    )
)
rustc -vV | findstr /c:"host: " | findstr /c:"gnu" >nul
if not errorlevel 1 (
    where dlltool >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] The GNU Rust toolchain needs MinGW-w64.
        echo         Install it: winget install BrechtSanders.WinLibs.POSIX.UCRT
        exit /b 1
    )
)

set "APP_VER=unknown"
for /f "tokens=2 delims== " %%v in ('findstr /b /c:"version = " Cargo.toml') do set "APP_VER=%%~v"

echo ========================================
echo  SEAMS v%APP_VER% - Production Build
echo ========================================
for /f "tokens=*" %%r in ('rustc --version') do echo [OK] %%r
echo.

:: -- Step 2: Tests ----------------------------------------
echo [2/5] Running the test suite...
cargo test --workspace --quiet
if errorlevel 1 (
    echo [ERROR] Tests failed. Nothing was built.
    exit /b 1
)
echo [OK] Tests passed.
echo.

:: -- Step 3: Release build ----------------------------------------
echo [3/5] Building the release executable...
cargo build --release -p seams
if errorlevel 1 (
    echo [ERROR] Build failed!
    exit /b 1
)
if not exist "dist" mkdir "dist"
copy /y "target\release\seams.exe" "dist\SEAMS.exe" >nul
if errorlevel 1 (
    echo [ERROR] Could not copy the executable to dist\. Is SEAMS still running?
    exit /b 1
)
echo [OK] Built: dist\SEAMS.exe
echo.

:: -- Step 4: Code sign (optional) ---------------------------------------
echo [4/5] Code signing...
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

:: -- Step 5: Inno Setup installer ---------------------------------------
echo [5/5] Building the Windows installer...
set "ISCC_EXE="
where iscc >nul 2>&1
if not errorlevel 1 set "ISCC_EXE=iscc"
if not defined ISCC_EXE if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "ISCC_EXE=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not defined ISCC_EXE if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"

if not defined ISCC_EXE (
    echo [WARNING] Inno Setup not found. Skipping the installer.
    echo           Install it: winget install JRSoftware.InnoSetup
    goto :done
)

"%ISCC_EXE%" /Q /DMyAppVersion=%APP_VER% installer.iss
if errorlevel 1 (
    echo [ERROR] Installer build failed!
    exit /b 1
)
echo [OK] Installer: dist\SEAMS_Setup_%APP_VER%.exe

if not defined SIGN_PFX goto :done
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a /f "%SIGN_PFX%" /p "%SIGN_PASS%" "dist\SEAMS_Setup_%APP_VER%.exe"
if errorlevel 1 (
    echo [ERROR] Installer signing failed!
    exit /b 1
)
echo [OK] Installer signed.

:done
echo.
echo ========================================
echo  BUILD COMPLETE
echo  EXE:        dist\SEAMS.exe
if defined ISCC_EXE echo  Installer:  dist\SEAMS_Setup_%APP_VER%.exe
echo ========================================
exit /b 0
