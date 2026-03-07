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
    except Exception:
        pass
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
        p.setPen(QColor(220, 220, 220))
        font = QFont("Segoe UI", 8)
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
        self._cpu_bar = _UsageBar("CPU", QColor(0, 152, 255))       # Blue
        self._ram_bar = _UsageBar("RAM", QColor(0, 200, 120))       # Green
        self._gpu_bar = _UsageBar("GPU", QColor(255, 140, 0)) if self._gpu_available else None  # Orange

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
        p.setBrush(QColor(20, 20, 20, 180))
        p.drawRoundedRect(self.rect(), 6, 6)
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
    """Tiny inline bar for status bar use (label + colored fill)."""

    def __init__(self, label_text: str, color: QColor, parent=None):
        super().__init__(parent)
        self._label = label_text
        self._color = color
        self._value = 0.0
        self.setFixedSize(80, 14)

    def set_value(self, v: float):
        self._value = max(0.0, min(100.0, v))
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background track
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(50, 50, 50))
        p.drawRoundedRect(0, 0, w, h, 2, 2)

        # Filled bar
        fill_w = int(w * self._value / 100.0)
        if fill_w > 0:
            c = QColor(self._color)
            c.setAlpha(200)
            p.setBrush(c)
            p.drawRoundedRect(0, 0, fill_w, h, 2, 2)

        # Text
        p.setPen(QColor(230, 230, 230))
        font = QFont("Segoe UI", 7)
        font.setBold(True)
        p.setFont(font)
        p.drawText(3, 0, w - 6, h, Qt.AlignmentFlag.AlignVCenter,
                   f"{self._label} {self._value:.0f}%")
        p.end()


class StatusBarMonitor(QWidget):
    """
    Compact horizontal monitor widget for embedding in a QStatusBar.
    Shows CPU | RAM | GPU as small inline bars.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gpu_available = _get_gpu_percent() is not None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._cpu = _MiniBar("CPU", QColor(0, 152, 255))
        self._ram = _MiniBar("RAM", QColor(0, 200, 120))
        layout.addWidget(self._cpu)
        layout.addWidget(self._ram)

        if self._gpu_available:
            self._gpu = _MiniBar("GPU", QColor(255, 140, 0))
            layout.addWidget(self._gpu)
        else:
            self._gpu = None

        # Polling timer — 1.5s
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

