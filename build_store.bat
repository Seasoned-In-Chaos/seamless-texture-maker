@echo off
setlocal enabledelayedexpansion

:: SEAMS Microsoft Store build: packages the release executable as MSIX.
:: Needs the Windows 10 SDK for MakeAppx.exe. Run from the repository root.

cd /d "%~dp0"

where cargo >nul 2>&1
if errorlevel 1 if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
where dlltool >nul 2>&1
if errorlevel 1 (
    for /d %%d in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs*") do (
        if exist "%%d\mingw64\bin\dlltool.exe" set "PATH=%%d\mingw64\bin;!PATH!"
    )
)

set "APP_VER=unknown"
for /f "tokens=2 delims== " %%v in ('findstr /b /c:"version = " Cargo.toml') do set "APP_VER=%%~v"

echo ========================================
echo  SEAMS v%APP_VER% - Microsoft Store (MSIX) Build
echo ========================================
echo.

:: -- Step 1: Windows SDK (MakeAppx) -------------------------------------
echo [1/4] Checking the Windows SDK...
set "MAKEAPPX="
set "SIGNTOOL="
for /f "delims=" %%i in ('dir /b /ad /o-n "C:\Program Files (x86)\Windows Kits\10\bin\10.*" 2^>nul') do (
    if not defined MAKEAPPX if exist "C:\Program Files (x86)\Windows Kits\10\bin\%%i\x64\MakeAppx.exe" (
        set "MAKEAPPX=C:\Program Files (x86)\Windows Kits\10\bin\%%i\x64\MakeAppx.exe"
        set "SIGNTOOL=C:\Program Files (x86)\Windows Kits\10\bin\%%i\x64\signtool.exe"
    )
)
if not defined MAKEAPPX (
    echo [ERROR] Windows SDK not found. Install the Windows 10 SDK to get MakeAppx.exe.
    exit /b 1
)
echo [OK] MakeAppx: %MAKEAPPX%
echo.

:: -- Step 2: Release build ----------------------------------------
echo [2/4] Building the release executable...
cargo build --release -p seams
if errorlevel 1 (
    echo [ERROR] Build failed!
    exit /b 1
)
echo.

:: -- Step 3: Assemble the package layout --------------------------------
echo [3/4] Assembling the MSIX layout...
set "MSIX_STAGING=dist\SEAMS_MSIX"
if exist "%MSIX_STAGING%" rmdir /s /q "%MSIX_STAGING%"
mkdir "%MSIX_STAGING%\images"
mkdir "%MSIX_STAGING%\SEAMS"
copy /y "store\AppxManifest.xml" "%MSIX_STAGING%\" >nul
copy /y "target\release\seams.exe" "%MSIX_STAGING%\SEAMS\SEAMS.exe" >nul
copy /y "store\images\*.png" "%MSIX_STAGING%\images\" >nul
echo [OK] Layout ready: %MSIX_STAGING%
echo.

:: -- Step 4: Package and sign ----------------------------------------
echo [4/4] Packaging...
"%MAKEAPPX%" pack /o /d "%MSIX_STAGING%" /p "dist\SEAMS.msix"
if errorlevel 1 (
    echo [ERROR] MSIX packaging failed!
    exit /b 1
)
echo [OK] Package: dist\SEAMS.msix

set "TEST_CERT_SUBJECT=CN=Shubham Panchasara, O=Shubham Panchasara, C=US"
set "CERT_THUMB="
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "(Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Subject -eq '%TEST_CERT_SUBJECT%' } | Select-Object -First 1).Thumbprint"') do set "CERT_THUMB=%%t"
if defined CERT_THUMB (
    "%SIGNTOOL%" sign /fd SHA256 /sha1 %CERT_THUMB% /a "dist\SEAMS.msix"
    if errorlevel 1 (
        echo [WARNING] Signing failed. The certificate may need to be trusted first.
    ) else (
        echo [OK] Signed with the local test certificate.
    )
) else (
    echo [INFO] No test certificate found; the package is unsigned.
)

echo.
echo ========================================
echo  MSIX BUILD COMPLETE
echo  Output: dist\SEAMS.msix
echo  Install locally:  powershell Add-AppxPackage -Path dist\SEAMS.msix
echo ========================================
exit /b 0
