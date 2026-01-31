"""
Seamless Texture Maker - Application Entry Point
Create perfectly seamless textures for 3D workflows.
"""
import sys
import os

# Add app directory to path
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from app.gui.main_window import MainWindow
from app.utils.config import APP_NAME


def get_icon_path():
    """Get path to application icon."""
    # Look for icon in resources folder
    possible_paths = [
        os.path.join(app_dir, 'resources', 'icon.ico'),
        os.path.join(app_dir, 'resources', 'icon.png'),
        os.path.join(app_dir, 'icon.ico'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    """Main entry point."""
    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("StudioTools")
    
    # Set application icon
    icon_path = get_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
    
    # Create and show main window
    window = MainWindow()
    window.setAcceptDrops(True)
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
