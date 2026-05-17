"""Global visual system for SEAMS."""

DARK_THEME = """
QMainWindow, QWidget {
    background: #07080c;
    color: #e6e2f2;
    font-family: "Segoe UI Variable", "Segoe UI", Arial, sans-serif;
    font-size: 12px;
}

QWidget#AppRoot {
    background: #05060a;
}

QMenuBar {
    background: #090b12;
    color: #d7d9ea;
    border-bottom: 1px solid #171b2a;
    padding: 2px 8px;
    font-weight: 700;
}

QMenuBar::item {
    background: transparent;
    padding: 5px 10px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background: #151a29;
    color: #ffffff;
}

QMenu {
    background: #0d111b;
    color: #e6e2f2;
    border: 1px solid #252d43;
    padding: 6px;
}

QMenu::item {
    padding: 7px 28px 7px 20px;
    border-radius: 4px;
}

QMenu::item:selected {
    background: #211936;
    color: #ffffff;
}

QMenu::separator {
    height: 1px;
    background: #252d43;
    margin: 5px 8px;
}

QWidget#CenterWorkspace {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #080a10, stop:1 #05060a);
    border-left: 1px solid #111522;
    border-right: 1px solid #111522;
}

QWidget#TopBar {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #080a10, stop:0.55 #0b0d16, stop:1 #07080d);
    border-bottom: 1px solid #171b2b;
}

QLabel#MicroLabel, QLabel#NavSection, QLabel#ParamLabel {
    color: #737891;
    font-size: 9px;
    font-weight: 900;
    letter-spacing: 1.5px;
}

QLabel#ProjectName {
    color: #f2efff;
    font-size: 16px;
    font-weight: 800;
}

QPushButton#ToolButton, QPushButton#MiniButton {
    background: #0d1018;
    border: 1px solid #20263a;
    border-radius: 6px;
    color: #b9b6d2;
    padding: 6px 10px;
    font-weight: 800;
}

QPushButton#ToolButton:hover, QPushButton#MiniButton:hover {
    color: #ffffff;
    border-color: #7d63ff;
    background: #151727;
}

QPushButton#SecondaryAction {
    background: #101420;
    border: 1px solid #252d44;
    border-radius: 7px;
    color: #d7d2f2;
    padding: 8px 16px;
    font-weight: 900;
}

QPushButton#HeaderExport, QPushButton#PrimaryAction {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #5b3dff, stop:0.58 #8f62ff, stop:1 #18d6b6);
    border: 1px solid rgba(255,255,255,40);
    border-radius: 8px;
    color: white;
    padding: 9px 18px;
    font-weight: 950;
    letter-spacing: 1px;
}

QPushButton#HeaderExport:hover, QPushButton#PrimaryAction:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #6d52ff, stop:0.58 #a67cff, stop:1 #28eccd);
    border-color: #c7bdff;
}

QPushButton#PrimaryAction:disabled {
    background: #111521;
    color: #444961;
    border-color: #1c2233;
}

QLabel#StatusPill {
    background: #0b1d19;
    border: 1px solid #1a5c4d;
    border-radius: 13px;
    color: #31e6bd;
    padding: 5px 12px;
    font-weight: 950;
    letter-spacing: 1px;
}

QLabel#StatusPill[busy="true"] {
    background: #211628;
    border-color: #8f62ff;
    color: #d8caff;
}

QWidget#ToolRail {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #07080c, stop:0.5 #05060a, stop:1 #07080d);
    border-right: 1px solid #171b2a;
}

QWidget#BrandBlock {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #090b12, stop:1 #05060a);
    border-bottom: 1px solid #171b2a;
}

QLabel#BrandTitle {
    color: #ffffff;
    font-size: 32px;
    font-weight: 950;
    letter-spacing: 2px;
}

QLabel#BrandSubtitle {
    color: #9a9cb2;
    font-size: 9px;
    font-weight: 900;
    letter-spacing: 2px;
}

QLabel#NavSection {
    padding: 18px 18px 7px 18px;
}

QWidget#NavItem {
    background: transparent;
    border-left: 3px solid transparent;
}

QWidget#NavItem:hover {
    background: #10131e;
    border-left-color: #353d5a;
}

QWidget#NavItem[active="true"] {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #171329, stop:1 rgba(23,19,41,0));
    border-left-color: #9a78ff;
}

QLabel#NavIcon {
    color: #737891;
    font-size: 16px;
    font-weight: 500;
}

QLabel#NavLabel {
    color: #c6c3d8;
    font-size: 12px;
    font-weight: 850;
}

QLabel#NavShortcut {
    color: #4e536c;
    font-size: 10px;
    font-weight: 700;
}

QLabel#InfoKey {
    color: #8c8ea6;
    font-size: 11px;
    font-weight: 500;
}

QLabel#InfoValue {
    color: #f2efff;
    font-size: 11px;
    font-weight: 700;
}

QLabel#InfoValueLarge {
    color: #f2efff;
    font-size: 13px;
    font-weight: 800;
}

QWidget#NavItem[active="true"] QLabel#NavIcon,
QWidget#NavItem[active="true"] QLabel#NavLabel {
    color: #f1ecff;
}

QFrame#RailCard, QFrame#PluginCard, QFrame#PreviewCard, QFrame#DockIntro {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0d1018, stop:1 #080a10);
    border: 1px solid #1b2132;
    border-radius: 8px;
}

QFrame#RailCard {
    margin: 12px;
}

QWidget#RightInspector {
    background: #07080d;
    border-left: 1px solid #171b2a;
}

QScrollArea#PanelScroll, QWidget#PanelBody {
    background: transparent;
}

QPushButton#PanButton {
    background: #5b3dff;
    color: white;
    border-radius: 8px;
    font-size: 18px;
    border: none;
}

QPushButton#PanButton:unchecked {
    background: #1a1b26;
    color: #737891;
}

QPushButton#FitButton {
    background: #1a1b26;
    color: #e6e2f2;
    border: 1px solid #20263a;
    border-radius: 8px;
    font-size: 16px;
}

QPushButton#FitButton:hover {
    background: #252839;
    border-color: #5b3dff;
}

QComboBox#ZoomCombo {
    background: #1a1b26;
    color: #e6e2f2;
    border: 1px solid #20263a;
    border-radius: 8px;
    padding: 2px 10px;
}
QLabel#PanelTitle {
    color: #ffffff;
    font-size: 24px;
    font-weight: 950;
    letter-spacing: 1.5px;
}

QLabel#PanelKicker, QLabel#CardSubtitle, QLabel#MutedText, QLabel#InfoLabel {
    color: #8b8da3;
    font-size: 11px;
    line-height: 1.35;
}

QLabel#CardTitle {
    color: #f2efff;
    font-size: 10px;
    font-weight: 950;
    letter-spacing: 1.4px;
}

QLabel#CardDot {
    background: #31e6bd;
    border-radius: 4px;
}

QLabel#Recommendation {
    color: #bcb6db;
    background: #0c111a;
    border: 1px solid #1c2538;
    border-radius: 6px;
    padding: 9px;
}

QLabel#ValueChip {
    background: #101521;
    border: 1px solid #252d43;
    border-radius: 5px;
    color: #dddcff;
    font-size: 10px;
    font-weight: 900;
}

QPushButton#Chip {
    background: #101420;
    border: 1px solid #252c40;
    border-radius: 12px;
    color: #9fa2ba;
    padding: 5px 9px;
    font-size: 10px;
    font-weight: 850;
}

QPushButton#Chip:hover {
    color: #ffffff;
    border-color: #6755d6;
}

QPushButton#Chip:checked {
    background: #211936;
    border-color: #8f70ff;
    color: #f1ecff;
}

QWidget#ViewportToolbar {
    background: #070910;
    border-bottom: 1px solid #151a29;
}

QLabel#ToolbarLabel {
    color: #737891;
    font-size: 9px;
    font-weight: 950;
    letter-spacing: 1.5px;
}

QWidget#ModePanel {
    background: #070910;
    border-right: 1px solid #151a29;
}

QLabel#ModeBadge {
    background: #0d121d;
    border: 1px solid #26324a;
    border-radius: 6px;
    color: #31e6bd;
    font-size: 10px;
    font-weight: 950;
    letter-spacing: 1px;
    padding: 7px 10px;
}

QWidget#ViewportToolbar QPushButton {
    background: #0d1018;
    border: 1px solid #20263a;
    border-radius: 6px;
    color: #aeb0c8;
    padding: 6px 10px;
    font-weight: 850;
}

QWidget#ViewportToolbar QPushButton:hover {
    color: #ffffff;
    border-color: #7d63ff;
}

QWidget#ViewportToolbar QPushButton:checked {
    background: #211936;
    border-color: #9a78ff;
    color: #ffffff;
}

QWidget#PreviewDock {
    background: #06070b;
    border-top: 1px solid #171b2a;
}

QFrame#ChannelCard {
    background: #0b0e15;
    border: 1px solid #20263a;
    border-radius: 7px;
}

QFrame#ChannelCard:hover {
    background: #101420;
    border-color: #465070;
}

QFrame#ChannelCard[active="true"] {
    background: #151827;
    border-color: #31e6bd;
}

QLabel#ChannelThumb {
    background: #05060a;
    border: 1px solid #20263a;
    border-radius: 5px;
    color: #737891;
    font-size: 10px;
    font-weight: 850;
}

QLabel#ChannelTitle {
    color: #f2efff;
    font-size: 12px;
    font-weight: 950;
}

QLabel#ChannelMeta {
    color: #85889f;
    font-size: 10px;
    font-weight: 700;
}

QPushButton#ModeToggle {
    color: #ffffff;
    background: #192033;
    border: 1px solid #31e6bd;
    border-radius: 7px;
    font-size: 11px;
    font-weight: 950;
    padding: 7px 14px;
}

QPushButton#ModeToggle:hover {
    background: #21304a;
}

QLabel#DockTitle {
    color: #ffffff;
    font-size: 24px;
    font-weight: 950;
    letter-spacing: 1px;
}

QComboBox {
    background: #0d111b;
    border: 1px solid #242c41;
    border-radius: 6px;
    color: #e6e2f2;
    padding: 6px 28px 6px 10px;
    min-height: 22px;
    font-weight: 750;
}

QComboBox:hover, QComboBox:focus {
    border-color: #8f70ff;
}

QComboBox:disabled {
    background: #090b11;
    border-color: #161b2a;
    color: #42485d;
}

QComboBox::drop-down {
    border: none;
    width: 22px;
}

QComboBox QAbstractItemView {
    background: #0d111b;
    border: 1px solid #303950;
    color: #e6e2f2;
    selection-background-color: #211936;
    selection-color: #ffffff;
    outline: 0;
}

QSlider::groove:horizontal {
    height: 5px;
    background: #161b2a;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #5b3dff, stop:1 #31e6bd);
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #8f70ff;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    border-color: #31e6bd;
    width: 16px;
    height: 16px;
    margin: -7px 0;
}

QCheckBox, QRadioButton {
    color: #c8c5da;
    spacing: 8px;
    font-weight: 700;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #303950;
    background: #0d111b;
}

QCheckBox::indicator {
    border-radius: 4px;
}

QRadioButton::indicator {
    border-radius: 8px;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background: #8f70ff;
    border-color: #d4c9ff;
}

QWidget#PanelFooter {
    background: #07080d;
    border-top: 1px solid #171b2a;
}

QStatusBar {
    background: #05060a;
    border-top: 1px solid #141827;
    color: #85889f;
    min-height: 26px;
}

QStatusBar::item {
    border: none;
}

QProgressBar {
    background: #111521;
    border: none;
    border-radius: 2px;
    max-height: 5px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #5b3dff, stop:1 #31e6bd);
    border-radius: 2px;
}

QScrollBar:vertical {
    background: transparent;
    width: 6px;
}

QScrollBar::handle:vertical {
    background: #252c40;
    border-radius: 3px;
    min-height: 28px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #05060a;
    height: 6px;
    border: none;
}

QScrollBar::handle:horizontal {
    background: #252c40;
    border-radius: 3px;
    min-width: 32px;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
}

QToolTip {
    background: #0d111b;
    border: 1px solid #303950;
    color: #f2efff;
    border-radius: 6px;
    padding: 7px 9px;
}
"""


def get_dark_theme():
    return DARK_THEME
