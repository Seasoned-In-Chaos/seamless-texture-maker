"""SEAMS application entry point."""
import sys
import os

app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "SeamlessTextureMaker", "numba_cache"))

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QSurfaceFormat

from app.gui.main_window import MainWindow
from app.gui.splash_screen import SplashScreen
from app.utils.config import APP_NAME
from app.utils.app_logging import LoggingApplication, install_exception_hook, setup_logging


def get_icon_path():
    """
    Resolve the application icon, handling both dev and PyInstaller frozen modes.

    Search order:
      1. PyInstaller _MEIPASS bundle dir  (frozen: resources bundled inside EXE)
      2. Directory containing sys.executable (frozen: icon.ico copied by installer)
      3. app_dir / resources               (dev mode)
      4. app_dir root                      (dev mode fallback)
    """
    search_roots = []

    # When frozen by PyInstaller, sys._MEIPASS is the temp extraction dir that
    # contains all bundled data files (including the 'resources' folder).
    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            search_roots.append(meipass)
        # The installer also copies icon.ico next to SEAMS.exe — check there too.
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        search_roots.append(exe_dir)

    # Always include the script / app directory as a fallback (dev mode).
    search_roots.append(app_dir)

    for root in search_roots:
        for rel in ('resources/icon.ico', 'resources/icon.png', 'icon.ico'):
            path = os.path.join(root, rel)
            if os.path.exists(path):
                return path

    return None


_app_mutex = None

def _set_native_window_icon(hwnd, icon_path):
    """
    Force-set the taskbar / title-bar icon on a native Win32 HWND.

    Qt's setWindowIcon works for the title bar but the Windows taskbar
    reads the icon from the HWND's ICON_BIG/ICON_SMALL atoms directly.
    LoadImage + WM_SETICON is the only reliable way to update both on
    every Windows PC regardless of DPI, Windows version, or deployment.

    This is a pure-ctypes call — no extra dependencies required.
    """
    try:
        import ctypes

        LR_LOADFROMFILE = 0x00000010
        LR_DEFAULTSIZE  = 0x00000040
        IMAGE_ICON      = 1
        WM_SETICON      = 0x0080
        ICON_SMALL      = 0
        ICON_BIG        = 1

        user32 = ctypes.windll.user32

        # Large icon — LoadImageW picks the best size from the multi-res .ico.
        hicon_big = user32.LoadImageW(
            None, icon_path, IMAGE_ICON, 0, 0,
            LR_LOADFROMFILE | LR_DEFAULTSIZE,
        )

        # Small icon — query the actual system small-icon dimensions so it is
        # pixel-perfect at every DPI (100 % = 16×16, 125 % = 20×20, etc.).
        sm_cx = user32.GetSystemMetrics(49)  # SM_CXSMICON
        sm_cy = user32.GetSystemMetrics(50)  # SM_CYSMICON
        hicon_small = user32.LoadImageW(
            None, icon_path, IMAGE_ICON, sm_cx, sm_cy,
            LR_LOADFROMFILE,
        )

        if hicon_big:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)
        if hicon_small:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)

    except Exception:
        pass  # Non-fatal — Qt icon will still show in title bar


def main():
    global _app_mutex
    if sys.platform == 'win32':
        import ctypes
        _app_mutex = ctypes.windll.kernel32.CreateMutexW(None, True, "SeamlessTextureMaker_Mutex_DA6FB758")
        if not _app_mutex:
            raise RuntimeError("SEAMS could not create its single-instance lock.")
        if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            return 0
        # SetCurrentProcessExplicitAppUserModelID must be called before any
        # window is created so that the taskbar groups/icons correctly.
        try:
            myappid = "SeasonedInChaos.SeamlessTextureMaker.v2"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

    logger = setup_logging()
    install_exception_hook()
    logger.info("Starting %s", APP_NAME)

    LoggingApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    gl_format = QSurfaceFormat()
    gl_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    gl_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    gl_format.setVersion(2, 1)
    gl_format.setSamples(4)
    gl_format.setDepthBufferSize(24)
    gl_format.setStencilBufferSize(8)
    gl_format.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(gl_format)

    app = LoggingApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("StudioTools")

    icon_path = get_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))

    # ── Cinematic splash screen ──────────────────────────────────────────────
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # Pre-create main window in background while splash plays
    window = MainWindow()
    window.setAcceptDrops(True)

    def _on_splash_done():
        splash.close()
        window.show()
        # After show() the native HWND is fully created.  Push the .ico
        # directly via WM_SETICON so the taskbar button shows the right icon.
        if sys.platform == 'win32' and icon_path and icon_path.lower().endswith('.ico'):
            hwnd = int(window.winId())
            _set_native_window_icon(hwnd, icon_path)

    splash.finished.connect(_on_splash_done)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
