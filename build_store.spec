# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Microsoft Store (MSIX) build.

Produces an onedir (folder) layout instead of a single EXE,
which is required for MSIX packaging.  The onedir layout lets
MSIX properly catalog each file and avoids the self-extraction
issues that onefile mode causes inside the Store container.

Build command:
  pyinstaller build_store.spec --clean
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

spec_dir = Path(SPECPATH)

EXCLUDE_PATTERNS = [
    'tests', 'testing', 'cuda', 'test_', 'doc_examples',
    'pycc_distutils', 'gdb', 'benchmarks', 'rocksdb',
]

def filtered_submodules(pkg):
    subs = collect_submodules(pkg)
    return [s for s in subs if not any(p in s for p in EXCLUDE_PATTERNS)]

def filtered_data(pkg, include_py=False):
    datas = collect_data_files(pkg, include_py_files=include_py)
    return [(src, dst) for src, dst in datas
            if not any(p in src.replace('\\', '/') for p in EXCLUDE_PATTERNS)]

datas = []
binaries = []
hiddenimports = []

datas += filtered_data('numba')
datas += filtered_data('llvmlite')
binaries += collect_dynamic_libs('llvmlite')
hiddenimports += filtered_submodules('numba')
hiddenimports += filtered_submodules('llvmlite')

# PyQt6: PyInstaller hooks automatically collect the necessary .pyd
# and Qt6/bin DLLs from hiddenimports.  Do NOT use collect_data_files
# or collect_dynamic_libs on PyQt6 — they pull in unused submodules.

datas.append(('resources', 'resources'))

rust_pyd = spec_dir / 'seams_core' / 'target' / 'release'
if rust_pyd.exists():
    for pyd_file in rust_pyd.glob('*.pyd'):
        binaries.append((str(pyd_file), '.'))

a = Analysis(
    ['main.py'],
    pathex=[str(spec_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.QtOpenGL',
        'PyQt6.QtOpenGLWidgets',
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
        'seams_core',
        'packaging',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', '_tkinter', 'tk', 'tcl',
        'matplotlib', 'scipy', 'pandas',
        'IPython', 'jupyter', 'notebook',
        'setuptools', 'pkg_resources',
        'xml.etree', 'xmlrpc',
        'http.server',
        'unittest', 'doctest',
        'pdb', 'profile', 'cProfile',
        'numba.cuda',
        'numba.tests', 'numba.testing',
        'numba.np.ufunc.parallel',
        'numba.pycc',
        'llvmlite.tests',
        'llvmlite.binding.ffi',
        'llvmlite.ir.values',
        'PyQt6.QtNetwork',
        'PyQt6.QtSvg',
        'PyQt6.QtPdf',
        'PyQt6.QtQml',
        'PyQt6.QtQuick',
        'PyQt6.QtWebEngine',
        'PyQt6.QtWebSockets',
        'PyQt6.QtBluetooth',
        'PyQt6.QtMultimedia',
        'PyQt6.QtXml',
        'PyQt6.QtSql',
        'PyQt6.QtTest',
        'PyQt6.QtDesigner',
        'PyQt6.QtHelp',
        'PyQt6.QtPrintSupport',
        'PyQt6.QtNfc',
        'PyQt6.QtPositioning',
        'PyQt6.QtLocation',
        'PyQt6.QtSensors',
        'PyQt6.QtSerialPort',
        'PyQt6.QtTextToSpeech',
        'PyQt6.QAxContainer',
        'PyQt6.QtDBus',
        'PyQt6.QtMultimediaWidgets',
        'PyQt6.QtPdfWidgets',
        'PyQt6.QtQuick3D',
        'PyQt6.QtQuickWidgets',
        'PyQt6.QtRemoteObjects',
        'PyQt6.QtSpatialAudio',
        'PyQt6.QtStateMachine',
        'PyQt6.QtSvgWidgets',
        'PyQt6.QtWebChannel',
    ],
    noarchive=False,
    optimize=1,
)

_qt_plugin_whitelist = {'qwindows.dll', 'qjpeg.dll', 'qpng.dll', 'qtiff.dll', 'qicns.dll', 'qsvg.dll', 'qwindowsvistastyle.dll'}
_qt_plugin_dirs = {'/platforms/', '/imageformats/', '/styles/', '/iconengines/', '/generic/', '/platforminputcontexts/'}

def _is_qt_plugin(path):
    p = path.replace('\\', '/')
    return any(d in p for d in _qt_plugin_dirs)

a.binaries = [b for b in a.binaries if not ('Qt6' in b[0] and _is_qt_plugin(b[0]) and b[0].rsplit('\\', 1)[-1].lower() not in _qt_plugin_whitelist)]
a.datas = [d for d in a.datas if not ('Qt6' in d[0] and _is_qt_plugin(d[0]) and d[0].rsplit('\\', 1)[-1].lower() not in _qt_plugin_whitelist)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='SEAMS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',
        'ucrtbase.dll',
        'msvcp140.dll',
        'python3*.dll',
    ],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icon.ico',
    version='version_info.txt',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',
        'ucrtbase.dll',
        'msvcp140.dll',
        'python3*.dll',
    ],
    name='SEAMS',
)
