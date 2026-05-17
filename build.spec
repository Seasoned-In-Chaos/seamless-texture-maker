# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Seamless Texture Maker.
Build command: pyinstaller build.spec --clean
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

# Get the directory containing the spec file
spec_dir = Path(SPECPATH)

# Collect all numba dependencies
tmp_ret = collect_all('numba')
datas = tmp_ret[0]
binaries = tmp_ret[1]
hiddenimports = tmp_ret[2]

# Also collect llvmlite (required by numba)
tmp_llvm = collect_all('llvmlite')
datas += tmp_llvm[0]
binaries += tmp_llvm[1]
hiddenimports += tmp_llvm[2]

# Add 'resources' to datas
datas.append(('resources', 'resources'))

a = Analysis(
    ['main.py'],
    pathex=[str(spec_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'cv2',
        'numpy',
        'numba',
        'numba.core',
        'numba.typed',
        'numba.np.ufunc',
        'llvmlite',
        'llvmlite.binding',
        'PIL',
        'PIL.Image',
        'PIL.PngImagePlugin',
        'PIL.JpegImagePlugin',
        'PIL.TiffImagePlugin',
        'psutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SEAMS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icon.ico',
    version='version_info.txt',
)
