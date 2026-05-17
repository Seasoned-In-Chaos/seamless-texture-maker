"""
System Monitor Widget — CPU/GPU/RAM overlay.
Positioned bottom-right inside ViewportContainer.
Updates every 1.5s via QTimer (non-blocking).
"""
import subprocess
import sys
import psutil
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont

from ..utils.app_logging import get_logger


logger = get_logger(__name__)


def _get_gpu_percent():
    """Return GPU utilization % or None if unavailable.
    Uses nvidia-smi directly with CREATE_NO_WINDOW to avoid console flash."""
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2,
            startupinfo=si,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        if result.returncode == 0:
            val = result.stdout.strip().split('\n')[0].strip()
            return float(val)
    except Exception as exc:
        logger.debug("GPU utilization query unavailable: %s", exc)
    return None


class _UsageBar(QWidget):
    """Compact horizontal bar showing label + percentage."""

    def __init__(self, label_text: str, color: QColor, parent=None):
        super().__init__(parent)
        self._label = label_text
        self._color = color
        self._value = 0.0  # 0-100
        self.setFixedHeight(16)
        self.setMinimumWidth(140)

    def set_value(self, v: float):
        self._value = max(0.0, min(100.0, v))
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background track
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 15))
        p.drawRoundedRect(0, 0, w, h, 3, 3)

        # Filled bar
        fill_w = int(w * self._value / 100.0)
        if fill_w > 0:
            bar_color = QColor(self._color)
            bar_color.setAlpha(180)
            p.setBrush(bar_color)
            p.drawRoundedRect(0, 0, fill_w, h, 3, 3)

        # Text (label + value)
        p.setPen(QColor(120, 110, 180))
        font = QFont("Segoe UI", 7)
        font.setBold(True)
        p.setFont(font)
        text = f"{self._label}  {self._value:.0f}%"
        p.drawText(4, 0, w - 8, h, Qt.AlignmentFlag.AlignVCenter, text)
        p.end()


class SystemMonitorWidget(QWidget):
    """
    Semi-transparent overlay showing real-time system utilization.
    Must be parented to ViewportContainer and repositioned on resize.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Frameless, transparent-for-mouse, stays on top of sibling widgets
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")

        self._gpu_available = _get_gpu_percent() is not None

        # Bars
        self._cpu_bar = _UsageBar("CPU", QColor(91, 61, 255))    # neon purple
        self._ram_bar = _UsageBar("RAM", QColor(61, 91, 255))    # neon blue
        self._gpu_bar = _UsageBar("GPU", QColor(61, 138, 255)) if self._gpu_available else None  # blue

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addWidget(self._cpu_bar)
        layout.addWidget(self._ram_bar)
        if self._gpu_bar:
            layout.addWidget(self._gpu_bar)

        self.setFixedWidth(170)
        self.adjustSize()

        # Polling timer — 1.5 sec, non-blocking
        self._timer = QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

        # Initial sample (non-blocking call)
        psutil.cpu_percent(interval=None)
        self._refresh()

    # ── painting ──────────────────────────────────────────────
    def paintEvent(self, _event):
        """Draw semi-transparent rounded background."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(8, 8, 20, 200))
        p.drawRoundedRect(self.rect(), 5, 5)
        p.end()

    # ── data refresh ──────────────────────────────────────────
    def _refresh(self):
        """Non-blocking update of all bars."""
        self._cpu_bar.set_value(psutil.cpu_percent(interval=None))
        self._ram_bar.set_value(psutil.virtual_memory().percent)
        if self._gpu_bar:
            gpu_pct = _get_gpu_percent()
            if gpu_pct is not None:
                self._gpu_bar.set_value(gpu_pct)

    # ── cleanup ───────────────────────────────────────────────
    def stop(self):
        self._timer.stop()


class _MiniBar(QWidget):
    """
    Premium status bar chip: pill background, colored accent bar,
    bright label + contrasted percentage, and a subtle glow edge.
    """

    # Color presets per metric type for dynamic usage coloring
    _WARN_THRESHOLD = 75.0
    _CRIT_THRESHOLD = 90.0

    def __init__(self, label_text: str, color: QColor, parent=None):
        super().__init__(parent)
        self._label = label_text
        self._color = color        # base accent color
        self._value = 0.0
        # Wider + taller for readability: label + value clearly visible
        self.setFixedSize(100, 20)

    def set_value(self, v: float):
        self._value = max(0.0, min(100.0, v))
        self.update()

    def _accent_color(self) -> QColor:
        """Shift hue toward amber/red as load climbs."""
        if self._value >= self._CRIT_THRESHOLD:
            return QColor(255, 80, 80)    # critical — red
        if self._value >= self._WARN_THRESHOLD:
            return QColor(255, 165, 50)   # warning — amber
        return self._color                # normal — base color

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        accent = self._accent_color()

        # ── 1. Pill background ────────────────────────────────────
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(22, 22, 30))
        p.drawRoundedRect(0, 0, w, h, h // 2, h // 2)

        # ── 2. Thin accent bar along the bottom (2 px) ───────────
        bar_h = 2
        fill_w = max(0, int((w - 4) * self._value / 100.0))
        if fill_w > 0:
            bar_color = QColor(accent)
            bar_color.setAlpha(220)
            p.setBrush(bar_color)
            p.drawRoundedRect(2, h - bar_h - 1, fill_w, bar_h, 1, 1)

        # ── 3. Small colored dot / bullet ────────────────────────
        dot_r = 3
        dot_x = 7
        dot_y = h // 2
        dot_c = QColor(accent)
        dot_c.setAlpha(230)
        p.setBrush(dot_c)
        p.drawEllipse(dot_x - dot_r, dot_y - dot_r, dot_r * 2, dot_r * 2)

        # ── 4. Label text (dim, uppercase) ────────────────────────
        label_font = QFont("Segoe UI", 7, QFont.Weight.Bold)
        p.setFont(label_font)
        p.setPen(QColor(130, 130, 155))   # muted grey-blue
        label_x = dot_x + dot_r + 3
        p.drawText(label_x, 0, 28, h, Qt.AlignmentFlag.AlignVCenter, self._label)

        # ── 5. Value text (bright white, right-aligned) ───────────
        value_font = QFont("Segoe UI", 7, QFont.Weight.Bold)
        p.setFont(value_font)
        # Bright white for normal, tinted for warn/crit
        if self._value >= self._CRIT_THRESHOLD:
            p.setPen(QColor(255, 110, 110))
        elif self._value >= self._WARN_THRESHOLD:
            p.setPen(QColor(255, 190, 80))
        else:
            p.setPen(QColor(210, 210, 230))
        val_str = f"{self._value:.0f}%"
        p.drawText(0, 0, w - 6, h, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, val_str)

        p.end()


class StatusBarMonitor(QWidget):
    """
    Compact horizontal monitor widget for embedding in a QStatusBar.
    Shows CPU | VRAM | GPU as polished inline chip bars.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gpu_available = _get_gpu_percent() is not None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 6, 0)
        layout.setSpacing(6)

        self._cpu = _MiniBar("CPU", QColor(120, 80, 255))    # purple
        self._ram = _MiniBar("MEM", QColor(60, 130, 255))    # blue
        layout.addWidget(self._cpu)
        layout.addWidget(self._ram)

        if self._gpu_available:
            self._gpu = _MiniBar("GPU", QColor(50, 200, 180))  # teal
            layout.addWidget(self._gpu)
        else:
            self._gpu = None

        # Polling timer — 1.5 s
        self._timer = QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

        psutil.cpu_percent(interval=None)
        self._refresh()

    def _refresh(self):
        self._cpu.set_value(psutil.cpu_percent(interval=None))
        self._ram.set_value(psutil.virtual_memory().percent)
        if self._gpu:
            gpu_pct = _get_gpu_percent()
            if gpu_pct is not None:
                self._gpu.set_value(gpu_pct)

    def stop(self):
        self._timer.stop()

