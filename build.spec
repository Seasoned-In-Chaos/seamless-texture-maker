# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SEAMS.
Build command: pyinstaller build.spec --clean
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

spec_dir = Path(SPECPATH)

EXCLUDE_PATTERNS = [
    # NOT 'tests' or 'testing': numba.cuda unconditionally imports
    # numba.runtests -> numba.testing._runtests at module load (purely to
    # define an unused numba.cuda.test() convenience helper), and a bare
    # 'tests' substring pattern would also match that -- 'test_' below still
    # filters individual test_*.py files, and the real ~4MB numba.tests/
    # suite is excluded precisely (not by substring) below instead.
    'test_', 'doc_examples',
    'pycc', 'gdb', 'benchmarks', 'rocksdb', 'tbbpool',
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

# NVIDIA CUDA redistributables for the optional Splat GPU path (Numba
# @cuda.jit -- see app/core/materialize_methods_cuda.py and
# gpu_utils._cuda_redist_roots/_ensure_cuda_home, which look for exactly
# this 'cuda_redist/nvcc/...' + 'cuda_redist/runtime/...' layout next to
# the frozen executable). Only the specific files Numba actually loads are
# bundled, not the ~37MB of C++ headers, static .lib files and ptxas.exe
# these two pip packages also carry, which are for compiling CUDA C++ and
# irrelevant to Numba's own NVVM-based JIT pipeline. Harmless if either
# package is somehow missing at build time (requirements.txt pins both for
# win32, so this should never happen) -- the app just falls back to the
# CPU-only Splat path at runtime, same as on a non-NVIDIA machine.
import glob

try:
    import nvidia.cuda_nvcc as _cuda_nvcc
    _nvcc_root = next(iter(_cuda_nvcc.__path__))
    # Globbed, not a hardcoded filename: nvvm64_40_0.dll is versioned and
    # requirements.txt only pins a major-version range, so a different
    # nvidia-cuda-nvcc-cu12 release could ship a different exact name.
    for _dll in glob.glob(os.path.join(_nvcc_root, 'nvvm', 'bin', 'nvvm64_*.dll')):
        datas.append((_dll, 'cuda_redist/nvcc/nvvm/bin'))
    for _bc in glob.glob(os.path.join(_nvcc_root, 'nvvm', 'libdevice', 'libdevice*.bc')):
        datas.append((_bc, 'cuda_redist/nvcc/nvvm/libdevice'))
except ImportError:
    pass

try:
    import nvidia.cuda_runtime as _cuda_runtime
    _runtime_root = next(iter(_cuda_runtime.__path__))
    for _dll in glob.glob(os.path.join(_runtime_root, 'bin', 'cudart64_*.dll')):
        datas.append((_dll, 'cuda_redist/runtime/bin'))
except ImportError:
    pass

# PyQt6: PyInstaller hooks (hook-PyQt6.QtCore.py etc.) automatically
# collect the necessary .pyd extension modules and Qt6/bin DLLs
# when the modules are listed in hiddenimports.  Do NOT use
# collect_data_files/collect_dynamic_libs on PyQt6 — they pull in
# everything including unused submodules.

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
        'PyQt6.QtOpenGL',
        'PyQt6.QtOpenGLWidgets',
        'cv2',
        'numpy',
        'PIL',
        'PIL.Image',
        'PIL.PngImagePlugin',
        'PIL.JpegImagePlugin',
        'PIL.TiffImagePlugin',
        'psutil',
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
        'doctest',
        'pdb',
        # NOT 'unittest', 'profile', 'cProfile': numba.testing.main (needed
        # by numba.cuda's import chain, see EXCLUDE_PATTERNS above) imports
        # all three at module load, even though this app never runs numba's
        # test suite through it.
        'numba.tests',
        'numba.pycc',
        'numba.np.ufunc.tbbpool',
        'numba.np.ufunc._tbbpool',
        'llvmlite.tests',
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
    a.binaries,
    a.datas,
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
