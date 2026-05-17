@echo off
echo ========================================
echo  Seamless Texture Maker - Build EXE
echo ========================================
echo.

:: Try to find Python - check common locations
set PYTHON_EXE=

for %%p in (
    "python"
    "python3"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "C:\Python312\python.exe"
) do (
    if defined PYTHON_EXE goto :found
    %%p --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_EXE=%%~p
    )
)

:found
if not defined PYTHON_EXE (
    echo [ERROR] Python not found! Please install Python 3.10+ and add it to PATH.
    echo Download from: https://www.python.org/downloads/
    :: pause
    exit /b 1
)

echo [OK] Found Python: %PYTHON_EXE%
%PYTHON_EXE% --version
echo.

:: Install / upgrade dependencies
echo [*] Installing dependencies...
%PYTHON_EXE% -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    :: pause
    exit /b 1
)
echo [OK] Dependencies ready.
echo.

:: Clean old build artifacts
echo [*] Cleaning previous build...
if exist "dist\SeamlessTextureMaker.exe" del /f "dist\SeamlessTextureMaker.exe"
if exist "build" rmdir /s /q "build"
echo [OK] Clean done.
echo.

:: Run PyInstaller
echo [*] Building EXE (this may take 2-5 minutes)...
%PYTHON_EXE% -m PyInstaller build.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed! Check the output above for errors.
    :: pause
    exit /b 1
)

echo.
echo ========================================
echo  BUILD SUCCESSFUL!
echo  Output: dist\SeamlessTextureMaker.exe
echo ========================================
echo.

:: Run Inno Setup Compiler
echo [*] Building Windows Installer...
set "ISCC_EXE="

iscc /? >nul 2>&1
if not errorlevel 1 set "ISCC_EXE=iscc"

if not defined ISCC_EXE (
    if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "ISCC_EXE=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)
if not defined ISCC_EXE (
    if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set "ISCC_EXE=C:\Program Files\Inno Setup 6\ISCC.exe"
)
if not defined ISCC_EXE (
    if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
)

if defined ISCC_EXE (
    echo [OK] Found Inno Setup Compiler: %ISCC_EXE%
    "%ISCC_EXE%" installer.iss
    if errorlevel 1 (
        echo [ERROR] Installer build failed!
    ) else (
        echo [OK] Installer successfully created in dist\ folder!
    )
) else (
    echo [WARNING] Inno Setup Compiler (ISCC) not found. Skipping installer build.
    echo Please install Inno Setup to automatically build the installer.
)

:: Open the dist folder
explorer dist
:: pause
