"""SEAMS application entry point."""
import sys
import os

app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QSurfaceFormat

from app.gui.main_window import MainWindow
from app.gui.splash_screen import SplashScreen
from app.utils.config import APP_NAME
from app.utils.app_logging import LoggingApplication, install_exception_hook, setup_logging


def get_icon_path():
    for path in [
        os.path.join(app_dir, 'resources', 'icon.ico'),
        os.path.join(app_dir, 'resources', 'icon.png'),
        os.path.join(app_dir, 'icon.ico'),
    ]:
        if os.path.exists(path):
            return path
    return None


_app_mutex = None

def main():
    global _app_mutex
    if sys.platform == 'win32':
        import ctypes
        _app_mutex = ctypes.windll.kernel32.CreateMutexW(None, True, "SeamlessTextureMaker_Mutex_DA6FB758")
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

    splash.finished.connect(_on_splash_done)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
