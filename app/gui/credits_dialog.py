"""Poster-style credits dialog."""

from __future__ import annotations

import os

from PyQt6.QtCore import QRect, QRectF, QSize, Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QGuiApplication, QPainter, QPixmap, QFont, QColor, QPen
from PyQt6.QtWidgets import QDialog, QPushButton, QWidget, QHBoxLayout, QLabel, QVBoxLayout


POSTER_SIZE = QSize(1536, 1024)
STUDIO_URL = "https://studiotrivima.in"
LINKEDIN_URL = "https://linkedin.com/in/shubham-panchasara-4416b023a"
INSTAGRAM_URL = "https://instagram.com/panchasarashubham"
HARSH_GITHUB_URL = "https://github.com/harshvasudeva"


def _resource(path: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "resources", path)


class _CreditsPoster(QWidget):
    def __init__(self, parent: QDialog):
        super().__init__(parent)
        self._pixmap = QPixmap(_resource("credits_page.png"))
        self._image_size = self._pixmap.size() if not self._pixmap.isNull() else POSTER_SIZE
        self._buttons: list[tuple[QPushButton, QRect]] = []
        self.setAutoFillBackground(False)

        self._add_hotspot(QRect(1455, 27, 36, 42), parent.accept)
        self._add_hotspot(QRect(833, 545, 655, 206), lambda: self._open(STUDIO_URL))
        self._add_hotspot(QRect(1048, 455, 352, 31), lambda: self._open(LINKEDIN_URL))
        self._add_hotspot(QRect(1048, 490, 320, 31), lambda: self._open(INSTAGRAM_URL))
        self._add_hotspot(QRect(1092, 824, 360, 31), lambda: self._open(LINKEDIN_URL))
        self._add_hotspot(QRect(1092, 860, 320, 31), lambda: self._open(INSTAGRAM_URL))

        self._contributor_rect = QRect(80, 950, 600, 55)
        self._add_hotspot(self._contributor_rect, lambda: self._open(HARSH_GITHUB_URL))

    def sizeHint(self) -> QSize:  # noqa: N802
        return self._image_size

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self._pixmap.isNull():
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Credits artwork missing")
            return

        painter.drawPixmap(self._poster_rect(), self._pixmap, QRectF(self._pixmap.rect()))
        self._draw_contributor_overlay(painter)

    def _draw_contributor_overlay(self, painter: QPainter):
        target = self._poster_rect()
        sx = target.width() / self._image_size.width()
        sy = target.height() / self._image_size.height()

        cx = target.x() + self._contributor_rect.x() * sx
        cy = target.y() + self._contributor_rect.y() * sy
        fs = max(8, int(11 * min(sx, sy)))

        f = QFont("Segoe UI", fs, QFont.Weight.Normal)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.5)
        painter.setFont(f)

        painter.setPen(QColor(140, 135, 160, 200))
        painter.drawText(QRectF(cx, cy, 500 * sx, 22 * sy), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "Contributors")

        fl = QFont("Segoe UI", fs, QFont.Weight.Bold)
        fl.setUnderline(True)
        painter.setFont(fl)
        painter.setPen(QColor(160, 80, 255, 220))
        painter.drawText(QRectF(cx, cy + 24 * sy, 400 * sx, 22 * sy), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "Harsh Vasudeva")

        fd = QFont("Segoe UI", max(7, int(9 * min(sx, sy))))
        painter.setFont(fd)
        painter.setPen(QColor(120, 115, 140, 180))
        painter.drawText(QRectF(cx + 220 * sx, cy + 26 * sy, 350 * sx, 18 * sy), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "[ Optimization & Production Readiness ]")

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        target = self._poster_rect()
        sx = target.width() / self._image_size.width()
        sy = target.height() / self._image_size.height()
        for button, rect in self._buttons:
            button.setGeometry(
                QRect(
                    round(target.x() + rect.x() * sx),
                    round(target.y() + rect.y() * sy),
                    round(rect.width() * sx),
                    round(rect.height() * sy),
                )
            )

    def _poster_rect(self) -> QRectF:
        bounds = self.rect()
        source = self._image_size
        scale = min(bounds.width() / source.width(), bounds.height() / source.height())
        width = source.width() * scale
        height = source.height() * scale
        return QRectF(
            (bounds.width() - width) / 2,
            (bounds.height() - height) / 2,
            width,
            height,
        )

    def _add_hotspot(self, rect: QRect, callback) -> None:
        button = QPushButton(self)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setStyleSheet("QPushButton { background: transparent; border: none; }")
        button.clicked.connect(callback)
        self._buttons.append((button, rect))

    def _open(self, url: str) -> None:
        QDesktopServices.openUrl(QUrl(url))


def show_credits(parent, app_version="3.0.0"):
    del app_version

    dialog = QDialog(parent)
    dialog.setWindowTitle("Credits")
    dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
    dialog.setModal(True)
    dialog.setStyleSheet("QDialog { background: #000000; }")

    available = QGuiApplication.primaryScreen().availableGeometry()
    scale = min(1.0, (available.width() - 24) / POSTER_SIZE.width(), (available.height() - 24) / POSTER_SIZE.height())
    dialog_size = QSize(round(POSTER_SIZE.width() * scale), round(POSTER_SIZE.height() * scale))
    dialog.setFixedSize(dialog_size)

    poster = _CreditsPoster(dialog)
    poster.setGeometry(dialog.rect())
    poster.show()

    dialog.exec()
