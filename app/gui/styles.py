"""
Dark theme stylesheet for the application.
"""

DARK_THEME = """
/* Main window */
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

/* Menu bar */
QMenuBar {
    background-color: #252526;
    color: #e0e0e0;
    border-bottom: 1px solid #3c3c3c;
    padding: 4px;
}

QMenuBar::item {
    background-color: transparent;
    padding: 6px 12px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #094771;
}

QMenu {
    background-color: #252526;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #094771;
}

QMenu::separator {
    height: 1px;
    background: #3c3c3c;
    margin: 4px 8px;
}

/* Toolbar */
QToolBar {
    background-color: #252526;
    border: none;
    spacing: 4px;
    padding: 4px;
}

QToolButton {
    background-color: transparent;
    border: none;
    border-radius: 4px;
    padding: 6px;
    color: #e0e0e0;
}

QToolButton:hover {
    background-color: #3c3c3c;
}

QToolButton:pressed {
    background-color: #094771;
}

/* Status bar */
QStatusBar {
    background-color: #007acc;
    color: white;
    border: none;
    padding: 4px;
}

QStatusBar::item {
    border: none;
}

/* Group boxes */
QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #0098ff;
}

/* Labels */
QLabel {
    color: #e0e0e0;
    background-color: transparent;
}

/* Sliders */
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #3c3c3c;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #0098ff;
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #1ba1e2;
}

QSlider::sub-page:horizontal {
    background: #0098ff;
    border-radius: 3px;
}

/* Spin boxes */
QSpinBox, QDoubleSpinBox {
    background-color: #3c3c3c;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e0e0e0;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #0098ff;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #4a4a4a;
    border: none;
    width: 16px;
}

/* Combo boxes */
QComboBox {
    background-color: #3c3c3c;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 6px 12px;
    color: #e0e0e0;
    min-width: 100px;
}

QComboBox:hover {
    border: 1px solid #0098ff;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox::down-arrow {
    width: 12px;
    height: 12px;
}

QComboBox QAbstractItemView {
    background-color: #252526;
    border: 1px solid #3c3c3c;
    selection-background-color: #094771;
    color: #e0e0e0;
}

/* Check boxes */
QCheckBox {
    spacing: 8px;
    color: #e0e0e0;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #4a4a4a;
    border-radius: 4px;
    background-color: #3c3c3c;
}

QCheckBox::indicator:checked {
    background-color: #0098ff;
    border-color: #0098ff;
}

QCheckBox::indicator:hover {
    border-color: #0098ff;
}

/* Radio buttons */
QRadioButton {
    spacing: 8px;
    color: #e0e0e0;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #4a4a4a;
    border-radius: 9px;
    background-color: #3c3c3c;
}

QRadioButton::indicator:checked {
    background-color: #0098ff;
    border-color: #0098ff;
}

/* Push buttons */
QPushButton {
    background-color: #0098ff;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #1ba1e2;
}

QPushButton:pressed {
    background-color: #007acc;
}

QPushButton:disabled {
    background-color: #3c3c3c;
    color: #808080;
}

/* Secondary buttons */
QPushButton[secondary="true"] {
    background-color: #3c3c3c;
    color: #e0e0e0;
}

QPushButton[secondary="true"]:hover {
    background-color: #4a4a4a;
}

/* Scroll areas */
QScrollArea {
    border: none;
    background-color: #1e1e1e;
}

QScrollBar:vertical {
    background: #252526;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: #4a4a4a;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #5a5a5a;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #252526;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background: #4a4a4a;
    border-radius: 6px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: #5a5a5a;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* Splitter */
QSplitter::handle {
    background-color: #3c3c3c;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* Tab widget */
QTabWidget::pane {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    background-color: #252526;
}

QTabBar::tab {
    background-color: #1e1e1e;
    color: #808080;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #252526;
    color: #e0e0e0;
}

QTabBar::tab:hover:!selected {
    background-color: #2d2d2d;
}

/* Frame */
QFrame {
    background-color: transparent;
}

QFrame[frameShape="4"] {
    background-color: #3c3c3c;
    max-height: 1px;
}

/* Progress bar */
QProgressBar {
    border: none;
    border-radius: 4px;
    background-color: #3c3c3c;
    text-align: center;
    color: white;
}

QProgressBar::chunk {
    background-color: #0098ff;
    border-radius: 4px;
}

/* Tooltips */
QToolTip {
    background-color: #252526;
    color: #e0e0e0;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 4px 8px;
}
"""


def get_dark_theme():
    """Return the dark theme stylesheet."""
    return DARK_THEME
