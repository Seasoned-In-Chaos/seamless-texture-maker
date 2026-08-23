"""Poster-style credits dialog."""

from __future__ import annotations

import os

from PyQt6.QtCore import QRect, QRectF, QSize, Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QGuiApplication, QPainter, QPixmap
from PyQt6.QtWidgets import QDialog, QPushButton, QWidget


LINKEDIN_URL = "https://linkedin.com/in/shubham-panchasara-4416b023a"
INSTAGRAM_URL = "https://instagram.com/panchasarashubham"
EMAIL_ADDRESS = "spanchasara1@gmail.com"


def _resource(path: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "resources", path)


_credits_pixmap_cache: QPixmap | None = None


def _load_credits_pixmap() -> QPixmap:
    """Decode credits_page.png once and reuse it on every subsequent open.

    Lazily populated (not at import time) since QPixmap needs a QApplication
    to already exist; safe to share since nothing ever mutates the pixmap
    after load.
    """
    global _credits_pixmap_cache
    if _credits_pixmap_cache is None:
        _credits_pixmap_cache = QPixmap(_resource("credits_page.png"))
    return _credits_pixmap_cache


class _CreditsPoster(QWidget):
    # Fallback canvas size only used if the artwork fails to load.
    _FALLBACK_SIZE = QSize(2048, 2048)

    def __init__(self, parent: QDialog):
        super().__init__(parent)
        self._pixmap = _load_credits_pixmap()
        self._image_size = self._pixmap.size() if not self._pixmap.isNull() else self._FALLBACK_SIZE
        self._buttons: list[tuple[QPushButton, QRect]] = []
        self.setAutoFillBackground(False)

        # Hotspot rects are pixel coordinates in the 2048x2048 artwork's own
        # space (see the "LinkedIn / Instagram / Email" row near the
        # bottom of the "Developed By" section); resizeEvent below scales
        # them to wherever the poster is actually drawn on screen.
        self._add_hotspot(QRect(693, 1416, 217, 88), lambda: self._open(LINKEDIN_URL))
        self._add_hotspot(QRect(1030, 1416, 230, 88), lambda: self._open(INSTAGRAM_URL))
        self._add_hotspot(QRect(1355, 1416, 179, 88), lambda: self._open(f"mailto:{EMAIL_ADDRESS}"))


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


def _make_close_button(dialog: QDialog) -> QPushButton:
    """A Qt-drawn close control, independent of whatever the artwork does
    or doesn't include -- the poster image has no baked-in close icon, so
    this is the only way to close the dialog other than pressing Escape."""
    btn = QPushButton("✕", dialog)
    btn.setFixedSize(34, 34)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    btn.setStyleSheet(
        "QPushButton {"
        "  background: rgba(10, 10, 16, 200);"
        "  color: #b9a8ff;"
        "  border: 1px solid rgba(143, 112, 255, 120);"
        "  border-radius: 17px;"
        "  font-size: 14px;"
        "}"
        "QPushButton:hover {"
        "  background: rgba(143, 112, 255, 70);"
        "  color: #ffffff;"
        "  border-color: #8f70ff;"
        "}"
    )
    btn.clicked.connect(dialog.accept)
    return btn


def show_credits(parent, app_version=None):
    del app_version  # the version is baked into the poster artwork itself

    dialog = QDialog(parent)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.setWindowTitle("Credits")
    dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
    dialog.setModal(True)
    dialog.setStyleSheet("QDialog { background: #000000; }")

    poster = _CreditsPoster(dialog)

    # Size the dialog from the actual artwork's own dimensions rather than a
    # hardcoded constant, so it always fits correctly no matter what shape
    # the poster image is. And size it relative to the main window's
    # current size (not the whole screen) so it reads as a proportionate
    # overlay on the app rather than a near-fullscreen window of its own --
    # this also means it naturally tracks whatever size the main window
    # happens to be resized to before Credits is opened.
    image_size = poster._image_size
    if parent is not None:
        host = parent.geometry()
    else:
        host = QGuiApplication.primaryScreen().availableGeometry()
    max_w = max(360, int(host.width() * 0.82))
    max_h = max(360, int(host.height() * 0.82))
    scale = min(1.0, max_w / image_size.width(), max_h / image_size.height())
    dialog_size = QSize(round(image_size.width() * scale), round(image_size.height() * scale))
    dialog.setFixedSize(dialog_size)

    if parent is not None:
        dialog.move(
            host.x() + (host.width() - dialog_size.width()) // 2,
            host.y() + (host.height() - dialog_size.height()) // 2,
        )

    poster.setGeometry(dialog.rect())
    poster.show()

    close_btn = _make_close_button(dialog)
    close_btn.move(dialog_size.width() - close_btn.width() - 12, 12)
    close_btn.raise_()

    dialog.exec()
