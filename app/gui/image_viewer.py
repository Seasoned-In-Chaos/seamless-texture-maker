"""Image viewing and realtime material studio workspaces."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..utils.app_logging import get_logger, log_exception
from .pbr_viewport import CHANNELS, PBRViewport


logger = get_logger(__name__)
CHANNEL_ORDER = [
    "Base Color",
    "Normal",
    "Roughness",
    "Metallic",
    "AO",
    "Height",
    "Opacity",
    "Emissive",
]

CHANNEL_LABELS = {
    "Base Color": "BaseColor",
    "Normal": "Normal",
    "Roughness": "Roughness",
    "Metallic": "Metallic",
    "AO": "AO",
    "Height": "Height",
    "Opacity": "Opacity",
    "Emissive": "Emissive",
}


def numpy_to_pixmap(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    fmt = QImage.Format.Format_RGBA8888 if c == 4 else QImage.Format.Format_RGB888
    qimg = QImage(img.data, w, h, w * c, fmt)
    return QPixmap.fromImage(qimg.copy())


class TextureViewport(QWidget):
    zoomChanged = pyqtSignal(float)
    fileDropped = pyqtSignal(str)
    importRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._before = None
        self._after = None
        self._mode = "split"
        self._zoom = 1.0
        self._show_guides = True
        self._tiles = 2
        self._split_ratio = 0.5
        self._pan = QPoint(0, 0)
        self._last_mouse = None
        self._dragging_pan = False
        self._dragging_split = False
        self._last_content_rect = QRect()
        self.setMouseTracking(True)

    def set_before_image(self, img):
        self._before = numpy_to_pixmap(img)
        self.update()

    def set_after_image(self, img):
        self._after = numpy_to_pixmap(img)
        self.update()

    def set_after_pixmap(self, pix):
        self._after = pix
        self.update()

    def set_mode(self, mode):
        self._mode = "single" if mode == "seamless" else mode
        self.update()

    def set_tiles(self, n):
        self._tiles = max(1, int(n))
        self.update()

    def set_show_guides(self, show):
        self._show_guides = show
        self.update()

    def set_zoom(self, zoom):
        self._zoom = max(0.1, float(zoom))
        self.update()

    def fit_to_view(self):
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#07080c"))
        if not self._before and not self._after:
            painter.setPen(QColor("#737891"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Double-click to import image")
            return

        target = self._after if self._after else self._before
        if not target or target.isNull():
            return

        content_rect = QRect()

        if self._mode == "side_by_side":
            content_rect = self._paint_side_by_side(painter)
        else:
            w, h = target.width(), target.height()
            view_w, view_h = self.width(), self.height()
            tile_factor = self._tiles if self._mode == "tile" else 1
            scale = min(view_w / max(1, w * tile_factor), view_h / max(1, h * tile_factor)) * self._zoom
            scale = max(0.001, scale)
            tile_w, tile_h = int(w * scale), int(h * scale)
            sw, sh = tile_w * tile_factor, tile_h * tile_factor
            if sw <= 0 or sh <= 0:
                return

            ox = (view_w - sw) // 2 + self._pan.x()
            oy = (view_h - sh) // 2 + self._pan.y()
            content_rect = QRect(ox, oy, sw, sh)

            if self._mode == "tile":
                for row in range(self._tiles):
                    for col in range(self._tiles):
                        painter.drawPixmap(ox + col * tile_w, oy + row * tile_h, tile_w, tile_h, target)
                if self._show_guides:
                    self._draw_tile_guides(painter, content_rect, tile_w, tile_h)
            elif self._mode == "split":
                self._paint_split(painter, target, content_rect, scale)
            else:
                painter.drawPixmap(content_rect, target)

        self._last_content_rect = content_rect

    def _paint_side_by_side(self, painter):
        before = self._before
        after = self._after if self._after else self._before
        if not before and not after:
            return QRect()
        gap = 24
        margin = 32
        panel_w = max(1, (self.width() - gap - margin * 2) // 2)
        panel_h = max(1, self.height() - margin * 2)
        left_area = QRect(margin, margin, panel_w, panel_h)
        right_area = QRect(margin + panel_w + gap, margin, panel_w, panel_h)
        self._draw_panel_backdrop(painter, left_area)
        self._draw_panel_backdrop(painter, right_area)
        left_rect = self._draw_fitted_pixmap(painter, before, left_area)
        right_rect = self._draw_fitted_pixmap(painter, after, right_area)
        painter.setPen(QPen(QColor(255, 255, 255, 28), 1))
        painter.drawLine(left_area.right() + gap // 2, margin, left_area.right() + gap // 2, self.height() - margin)
        self._draw_corner_label(painter, left_area, "BEFORE")
        self._draw_corner_label(painter, right_area, "AFTER")
        return left_rect.united(right_rect)

    def _paint_split(self, painter, target, rect, scale):
        before = self._before if self._before else target
        after = self._after if self._after else target
        painter.drawPixmap(rect, before)
        split_x = rect.left() + int(rect.width() * self._split_ratio)
        if after:
            target_rect = QRectF(float(split_x), float(rect.top()), float(rect.right() - split_x + 1), float(rect.height()))
            src_x = max(0.0, (split_x - rect.left()) / scale)
            src_w = max(1.0, target_rect.width() / scale)
            src_rect = QRectF(src_x, 0.0, src_w, after.height())
            painter.drawPixmap(target_rect, after, src_rect)
        if self._show_guides:
            self._draw_texture_bounds(painter, rect)
        self._draw_split_handle(painter, split_x, rect)

    def _draw_fitted_pixmap(self, painter, pixmap, area):
        if pixmap is None or pixmap.isNull():
            return QRect()
        scale = min(area.width() / max(1, pixmap.width()), area.height() / max(1, pixmap.height())) * self._zoom
        scale = max(0.001, scale)
        sw, sh = int(pixmap.width() * scale), int(pixmap.height() * scale)
        rect = QRect(
            area.left() + (area.width() - sw) // 2 + self._pan.x(),
            area.top() + (area.height() - sh) // 2 + self._pan.y(),
            sw,
            sh,
        )
        painter.save()
        painter.setClipRect(area)
        painter.drawPixmap(rect, pixmap)
        painter.restore()
        return rect

    def _draw_panel_backdrop(self, painter, area):
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(3, 5, 9, 150))
        painter.drawRect(area)
        painter.restore()

    def _draw_texture_bounds(self, painter, rect):
        if rect.isNull() or rect.width() <= 1 or rect.height() <= 1:
            return
        painter.save()
        painter.setClipRect(rect)
        painter.setPen(QPen(QColor(255, 255, 255, 42), 1, Qt.PenStyle.DashLine))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.restore()

    def _draw_tile_guides(self, painter, rect, tile_w, tile_h):
        painter.save()
        painter.setClipRect(rect)
        painter.setPen(QPen(QColor("#38d5c2"), 1, Qt.PenStyle.DashLine))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        for col in range(1, self._tiles):
            x = rect.left() + col * tile_w
            painter.drawLine(x, rect.top(), x, rect.bottom())
        for row in range(1, self._tiles):
            y = rect.top() + row * tile_h
            painter.drawLine(rect.left(), y, rect.right(), y)
        painter.restore()

    def _draw_split_handle(self, painter, split_x, rect):
        painter.save()
        painter.setPen(QPen(QColor("#31e6bd"), 2))
        painter.drawLine(split_x, rect.top(), split_x, rect.bottom())
        handle_x = max(rect.left() + 4, min(rect.right() - 40, split_x - 18))
        handle = QRect(handle_x, rect.bottom() - 42, 36, 28)
        painter.setBrush(QColor("#101823"))
        painter.setPen(QPen(QColor("#31e6bd"), 1))
        painter.drawRoundedRect(handle, 6, 6)
        painter.setPen(QColor("#dffdf8"))
        painter.drawText(handle, Qt.AlignmentFlag.AlignCenter, "<>")
        self._draw_corner_label(painter, QRect(rect.left(), rect.top(), split_x - rect.left(), rect.height()), "BEFORE")
        self._draw_corner_label(painter, QRect(split_x, rect.top(), rect.right() - split_x + 1, rect.height()), "AFTER")
        painter.restore()

    def _draw_corner_label(self, painter, rect, text):
        if rect.isNull() or rect.width() < 48 or rect.height() < 24:
            return
        label = QRect(rect.left() + 10, rect.top() + 10, 68, 22)
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(4, 6, 10, 170))
        painter.drawRoundedRect(label, 5, 5)
        painter.setPen(QColor("#dffdf8"))
        painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton and self._mode == "split" and self._is_on_split_handle(pos):
            self._dragging_split = True
            self._update_split_ratio(pos)
        elif event.button() == Qt.MouseButton.LeftButton:
            self._dragging_pan = True
            self._last_mouse = pos

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._dragging_split:
            self._update_split_ratio(pos)
        elif self._dragging_pan:
            self._pan += pos - self._last_mouse
            self._last_mouse = pos
            self.update()
        elif self._mode == "split" and self._is_on_split_handle(pos):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        if not self._before and not self._after:
            self.importRequested.emit()

    def mouseReleaseEvent(self, event):
        self._dragging_pan = False
        self._dragging_split = False

    def _is_on_split_handle(self, pos):
        if self._last_content_rect.isNull():
            return False
        split_x = self._last_content_rect.left() + int(self._last_content_rect.width() * self._split_ratio)
        edge_hit = (
            self._last_content_rect.contains(pos)
            and abs(pos.x() - split_x) <= 18
        )
        handle_hit = QRect(
            max(self._last_content_rect.left() + 4, min(self._last_content_rect.right() - 40, split_x - 18)),
            self._last_content_rect.bottom() - 42,
            36,
            28,
        ).contains(pos)
        return edge_hit or handle_hit

    def _update_split_ratio(self, pos):
        if self._last_content_rect.isNull():
            return
        rel = (pos.x() - self._last_content_rect.left()) / max(1, self._last_content_rect.width())
        self._split_ratio = max(0.0, min(1.0, rel))
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 1200.0
        self._zoom = max(0.1, self._zoom + delta)
        self.zoomChanged.emit(self._zoom)
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.fileDropped.emit(urls[0].toLocalFile())


class ViewportToolbar(QWidget):
    modeChanged = pyqtSignal(str)
    tilesChanged = pyqtSignal(int)
    guidesChanged = pyqtSignal(bool)
    fitRequested = pyqtSignal()
    zoomRequested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ViewportToolbar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)
        self._last_mode = "split"
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.mode_buttons = {}
        for text, mode in [("Split", "split"), ("Side by Side", "side_by_side"), ("Single", "seamless")]:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, m=mode: self._set_mode(m))
            layout.addWidget(btn)
            self.group.addButton(btn)
            self.mode_buttons[mode] = btn
            if mode == "split":
                btn.setChecked(True)

        self.guides_btn = QPushButton("Guides")
        self.guides_btn.setCheckable(True)
        self.guides_btn.setChecked(True)
        self.guides_btn.toggled.connect(self.guidesChanged.emit)
        layout.addWidget(self.guides_btn)

        self.tiled_btn = QPushButton("Tiled")
        self.tiled_btn.setCheckable(True)
        self.tiled_btn.toggled.connect(self._set_tiled)
        layout.addWidget(self.tiled_btn)

        self.tile_combo = QComboBox()
        self.tile_combo.addItems(["2x2", "3x3", "4x4"])
        self.tile_combo.setEnabled(False)
        self.tile_combo.currentIndexChanged.connect(self._set_tile_count)
        layout.addWidget(self.tile_combo)
        self.mode_lbl = QLabel("VIEW: SPLIT")
        self.mode_lbl.setObjectName("ModeBadge")
        layout.addWidget(self.mode_lbl)
        layout.addStretch()
        self.zoom_lbl = QLabel("100%")
        self.zoom_lbl.setObjectName("ToolbarLabel")
        layout.addWidget(self.zoom_lbl)
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self.fitRequested.emit)
        layout.addWidget(fit_btn)

    def set_zoom_text(self, zoom):
        self.zoom_lbl.setText(f"{int(zoom * 100)}%")

    def set_active_mode(self, mode):
        if mode == "single":
            mode = "seamless"
        if mode != "tile":
            self._last_mode = mode
            if mode in self.mode_buttons:
                self.mode_buttons[mode].setChecked(True)
            if self.tiled_btn.isChecked():
                self.tiled_btn.blockSignals(True)
                self.tiled_btn.setChecked(False)
                self.tiled_btn.blockSignals(False)
            self.tile_combo.setEnabled(False)
        self._refresh_label(mode)

    def _set_mode(self, mode):
        self._last_mode = mode
        if self.tiled_btn.isChecked():
            self.tiled_btn.blockSignals(True)
            self.tiled_btn.setChecked(False)
            self.tiled_btn.blockSignals(False)
            self.tile_combo.setEnabled(False)
        self._refresh_label(mode)
        self.modeChanged.emit(mode)

    def _set_tiled(self, checked):
        self.tile_combo.setEnabled(checked)
        if checked:
            self._refresh_label("tile")
            self.tilesChanged.emit(self.tile_combo.currentIndex() + 2)
            self.modeChanged.emit("tile")
        else:
            self._refresh_label(self._last_mode)
            self.modeChanged.emit(self._last_mode)

    def _set_tile_count(self, index):
        self.tilesChanged.emit(index + 2)
        if self.tiled_btn.isChecked():
            self.modeChanged.emit("tile")

    def _refresh_label(self, mode):
        labels = {
            "split": "VIEW: SPLIT",
            "side_by_side": "VIEW: SIDE BY SIDE",
            "seamless": "VIEW: SINGLE",
            "single": "VIEW: SINGLE",
            "tile": f"VIEW: TILED {self.tile_combo.currentText()}",
        }
        self.mode_lbl.setText(labels.get(mode, "VIEW: SINGLE"))


class PBRViewportToolbar(QWidget):
    def __init__(self, viewport: PBRViewport, splitter=None, parent=None):
        super().__init__(parent)
        self.setObjectName("ViewportToolbar")
        self.viewport = viewport
        self._studio_splitter = splitter          # direct ref — no runtime parent traversal
        self._in_fullscreen = False
        self._fs_hidden = []
        self._fs_splitter_sizes = []
        self._fs_esc = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self.mesh_combo = QComboBox()
        self.mesh_combo.addItems(["Sphere", "Cube", "Plane"])
        self.mesh_combo.currentTextChanged.connect(viewport.set_mesh)
        layout.addWidget(self._labeled("Mesh", self.mesh_combo))

        self.hdri_combo = QComboBox()
        self.hdri_combo.addItems(["Studio", "Outdoor", "Archviz", "Neutral Gray"])
        self.hdri_combo.currentTextChanged.connect(viewport.set_hdri)
        layout.addWidget(self._labeled("HDRI", self.hdri_combo))

        self.tile_combo = QComboBox()
        self.tile_combo.addItems(["1x", "2x", "4x", "Infinite"])
        self.tile_combo.currentTextChanged.connect(self._on_tiling_changed)
        layout.addWidget(self._labeled("Tiling", self.tile_combo))

        for text, slot, checked in [
            ("Wire", viewport.set_wireframe, False),
            ("UV", viewport.set_uv_checker, False),
            ("Triplanar", viewport.set_triplanar, False),
            ("Tess", viewport.set_tessellation, False),
        ]:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setChecked(checked)
            btn.toggled.connect(slot)
            layout.addWidget(btn)

        self.displacement = QSlider(Qt.Orientation.Horizontal)
        self.displacement.setRange(0, 35)
        self.displacement.setValue(8)
        self.displacement.setFixedWidth(96)
        self.displacement.valueChanged.connect(viewport.set_displacement_strength)
        layout.addWidget(self._labeled("Height", self.displacement))

        self.exposure = QSlider(Qt.Orientation.Horizontal)
        self.exposure.setRange(20, 220)
        self.exposure.setValue(100)
        self.exposure.setFixedWidth(92)
        self.exposure.valueChanged.connect(viewport.set_exposure)
        layout.addWidget(self._labeled("Exposure", self.exposure))

        self.env = QSlider(Qt.Orientation.Horizontal)
        self.env.setRange(0, 220)
        self.env.setValue(100)
        self.env.setFixedWidth(92)
        self.env.valueChanged.connect(viewport.set_environment_intensity)
        layout.addWidget(self._labeled("Env", self.env))

        self.rotate = QSlider(Qt.Orientation.Horizontal)
        self.rotate.setRange(0, 360)
        self.rotate.setValue(0)
        self.rotate.setFixedWidth(92)
        self.rotate.valueChanged.connect(viewport.set_environment_rotation)
        layout.addWidget(self._labeled("Rotate", self.rotate))

        self.fullscreen_btn = QPushButton("Fullscreen")
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        layout.addWidget(self.fullscreen_btn)

        layout.addStretch()
        self.status = QLabel("FPS --   GPU tex --   No texture   PBR")
        self.status.setObjectName("ToolbarLabel")
        self.status.setMinimumWidth(270)
        layout.addWidget(self.status)
        viewport.statsChanged.connect(self.status.setText)

    def _labeled(self, label, widget):
        holder = QWidget()
        holder.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        text = QLabel(label.upper())
        text.setObjectName("ToolbarLabel")
        layout.addWidget(text)
        layout.addWidget(widget)
        return holder

    def _on_tiling_changed(self, text):
        value = 8 if text == "Infinite" else int(text.replace("x", ""))
        self.viewport.set_tiling(value)

    def _toggle_fullscreen(self):
        try:
            if self._in_fullscreen:
                self._exit_fullscreen()
            else:
                self._enter_fullscreen()
        except Exception as exc:
            log_exception(logger, "Failed to toggle fullscreen viewport", exc)

    def _enter_fullscreen(self):
        win = self.window()

        # Save and collapse the 2D side of the splitter
        if self._studio_splitter:
            self._fs_splitter_sizes = list(self._studio_splitter.sizes())
            total = sum(self._fs_splitter_sizes)
            self._studio_splitter.setSizes([total, 0])

        # Hide main-window side panels (rail + inspector)
        self._fs_hidden = []
        
        # 1. Hide direct attributes
        for attr in ('rail', 'bottom_bar'):
            w = getattr(win, attr, None)
            if w is not None and w.isVisible():
                w.hide()
                self._fs_hidden.append(w)
                
        # 2. Hide the right inspector panel (parent of control_stack)
        stack = getattr(win, 'control_stack', None)
        if stack and stack.parentWidget() and stack.parentWidget().isVisible():
            w = stack.parentWidget()
            w.hide()
            self._fs_hidden.append(w)
            
        # 3. Hide the channel dock inside ImageViewer
        iv = getattr(win, 'image_viewer', None)
        if iv and iv.bottom_widget and iv.bottom_widget.isVisible():
            w = iv.bottom_widget
            w.hide()
            self._fs_hidden.append(w)
            
        # 4. Hide menu bar
        mb = win.menuBar()
        if mb and mb.isVisible():
            mb.hide()
            self._fs_hidden.append(mb)

        # showMaximized is GL-safe on Windows (no HWND change unlike showFullScreen)
        win.showMaximized()
        self._in_fullscreen = True
        self.fullscreen_btn.setText("Exit Fullscreen")

        # Escape to exit
        from PyQt6.QtWidgets import QShortcut
        from PyQt6.QtGui import QKeySequence
        if self._fs_esc:
            self._fs_esc.deleteLater()
        self._fs_esc = QShortcut(QKeySequence("Escape"), win)
        self._fs_esc.activated.connect(self._exit_fullscreen)

    def _exit_fullscreen(self):
        if not self._in_fullscreen:
            return
        win = self.window()
        win.showNormal()

        # Restore splitter
        if self._studio_splitter and self._fs_splitter_sizes:
            self._studio_splitter.setSizes(self._fs_splitter_sizes)

        # Restore hidden panels
        for w in self._fs_hidden:
            try:
                w.show()
            except RuntimeError:
                continue
        self._fs_hidden = []

        # Remove escape shortcut
        if self._fs_esc:
            self._fs_esc.deleteLater()
            self._fs_esc = None

        self._in_fullscreen = False
        self.fullscreen_btn.setText("Fullscreen")


class ChannelCard(QFrame):
    selected = pyqtSignal(str)
    enabledChanged = pyqtSignal(str, bool)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self._active = False
        self.setObjectName("ChannelCard")
        self.setFixedSize(172, 96)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 8, 9, 8)
        layout.setSpacing(9)

        self.thumb = QLabel()
        self.thumb.setObjectName("ChannelThumb")
        self.thumb.setFixedSize(58, 58)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setText("...")
        layout.addWidget(self.thumb)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(2)
        self.title = QLabel(CHANNEL_LABELS.get(name, name))
        self.title.setObjectName("ChannelTitle")
        self.colorspace = QLabel(CHANNELS.get(name, {}).get("colorspace", "Linear"))
        self.colorspace.setObjectName("ChannelMeta")
        self.resolution = QLabel("--")
        self.resolution.setObjectName("ChannelMeta")
        self.toggle = QCheckBox("Enabled")
        self.toggle.setChecked(True)
        self.toggle.toggled.connect(lambda state: self.enabledChanged.emit(self.name, state))
        body.addWidget(self.title)
        body.addWidget(self.colorspace)
        body.addWidget(self.resolution)
        body.addWidget(self.toggle)
        layout.addLayout(body, 1)

    def set_pixmap(self, pixmap: QPixmap | None):
        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(self.thumb.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            crop = QPixmap(self.thumb.size())
            crop.fill(QColor("#0b0d14"))
            painter = QPainter(crop)
            painter.drawPixmap((crop.width() - scaled.width()) // 2, (crop.height() - scaled.height()) // 2, scaled)
            painter.end()
            self.thumb.setPixmap(crop)
            self.thumb.setText("")
            self.resolution.setText(f"{pixmap.width()} x {pixmap.height()}")
        else:
            self.thumb.clear()
            self.thumb.setText("Empty")
            self.resolution.setText("--")

    def set_active(self, active):
        self._active = active
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_enabled_visual(self, enabled: bool):
        """Visually dim the card and strike out title when the channel is disabled."""
        opacity = 1.0 if enabled else 0.38
        # Dim the entire card using a stylesheet opacity trick via setGraphicsEffect
        from PyQt6.QtWidgets import QGraphicsOpacityEffect
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(opacity)
        self.setGraphicsEffect(effect)
        # Strike-through on title for extra clarity
        font = self.title.font()
        font.setStrikeOut(not enabled)
        self.title.setFont(font)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.name)


class ChannelDock(QWidget):
    mapChanged = pyqtSignal(str)
    channelEnabledChanged = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PreviewDock")
        self._active = "Base Color"
        self._enabled: dict[str, bool] = {}
        self.cards: dict[str, ChannelCard] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 6)
        root.setSpacing(6)

        top = QHBoxLayout()
        label = QLabel("MATERIAL CHANNELS")
        label.setObjectName("ToolbarLabel")
        top.addWidget(label)
        top.addStretch()
        self.loading = QLabel("READY")
        self.loading.setObjectName("ToolbarLabel")
        top.addWidget(self.loading)
        root.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        for name in CHANNEL_ORDER:
            card = ChannelCard(name)
            self._enabled[name] = True
            card.selected.connect(self.select)
            card.enabledChanged.connect(self._on_card_enabled_changed)
            self.cards[name] = card
            content_layout.addWidget(card)
        content_layout.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)
        self.select("Base Color", emit=False)

    def _on_card_enabled_changed(self, name: str, enabled: bool):
        """Track state, update visual, auto-navigate away from a disabled active map."""
        self._enabled[name] = enabled
        card = self.cards.get(name)
        if card:
            card.set_enabled_visual(enabled)
        # If the currently viewed map is being disabled, fall back to Base Color
        if not enabled and self._active == name:
            self.select("Base Color")
        self.channelEnabledChanged.emit(name, enabled)

    def is_channel_enabled(self, name: str) -> bool:
        return self._enabled.get(name, True)

    def select(self, name: str, emit: bool = True):
        if name not in self.cards:
            return
        self._active = name
        for card_name, card in self.cards.items():
            card.set_active(card_name == name)
        if emit:
            self.mapChanged.emit(name)

    def checked_name(self):
        return self._active

    def set_channel(self, name: str, pixmap: QPixmap | None):
        if name == "Displacement":
            name = "Height"
        card = self.cards.get(name)
        if card:
            card.set_pixmap(pixmap)

    def set_loading(self, loading: bool):
        self.loading.setText("UPDATING" if loading else "READY")


class StudioWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.viewport3d = PBRViewport()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setStyleSheet("QSplitter::handle { background: #171b2a; width: 2px; }")
        self.viewport2d = TextureViewport()
        self.viewport2d.set_mode("single")
        self.splitter.addWidget(self.viewport3d)
        self.splitter.addWidget(self.viewport2d)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([760, 430])
        # Pass splitter directly so toolbar never needs to traverse parent chain
        self.toolbar = PBRViewportToolbar(self.viewport3d, splitter=self.splitter)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.splitter, 1)


class ClassicWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.viewport = TextureViewport()
        layout.addWidget(self.viewport, 1)


class ImageViewer(QWidget):
    fileDropped = pyqtSignal(str)
    importRequested = pyqtSignal()
    studioModeRequested = pyqtSignal()
    classicModeRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.maps: dict[str, QPixmap] = {}
        self._mode = "seamless"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.workspace_splitter = QSplitter(Qt.Orientation.Vertical)
        self.workspace_splitter.setStyleSheet("QSplitter::handle { background: #171b2a; height: 2px; }")
        self.stack = QStackedWidget()
        self.classic = ClassicWorkspace()
        self.studio = StudioWorkspace()
        self.stack.addWidget(self.classic)
        self.stack.addWidget(self.studio)
        self.workspace_splitter.addWidget(self.stack)

        self.bottom_widget = QWidget()
        self.bottom_widget.setMinimumHeight(210)
        self.bottom_widget.setMaximumHeight(320)
        self.bottom_widget.setObjectName("PreviewDock")
        bottom_layout = QHBoxLayout(self.bottom_widget)
        bottom_layout.setContentsMargins(10, 5, 10, 5)
        bottom_layout.setSpacing(12)

        mode_panel = QWidget()
        mode_panel.setObjectName("ModePanel")
        mode_layout = QVBoxLayout(mode_panel)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)
        self.mode_badge = QLabel("CLASSIC MODE")
        self.mode_badge.setObjectName("ModeBadge")
        self.mode_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_badge.setMinimumWidth(124)
        mode_layout.addWidget(self.mode_badge)
        self.toggle_btn = QPushButton("Switch to Studio")
        self.toggle_btn.setObjectName("ModeToggle")
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.clicked.connect(self._toggle_mode)
        mode_layout.addWidget(self.toggle_btn)
        mode_layout.addStretch()
        bottom_layout.addWidget(mode_panel)

        self.controls_panel = QWidget()
        controls_layout = QVBoxLayout(self.controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        self.classic_toolbar = ViewportToolbar()
        self.map_selector = ChannelDock()
        controls_layout.addWidget(self.classic_toolbar, 0)
        controls_layout.addWidget(self.map_selector, 1)
        bottom_layout.addWidget(self.controls_panel, 1)
        self.workspace_splitter.addWidget(self.bottom_widget)
        self.workspace_splitter.setStretchFactor(0, 1)
        self.workspace_splitter.setStretchFactor(1, 0)
        self.workspace_splitter.setSizes([700, 228])
        layout.addWidget(self.workspace_splitter, 1)

        self.classic.viewport.fileDropped.connect(self.fileDropped.emit)
        self.studio.viewport2d.fileDropped.connect(self.fileDropped.emit)
        self.classic.viewport.importRequested.connect(self.importRequested.emit)
        self.studio.viewport2d.importRequested.connect(self.importRequested.emit)
        self.map_selector.mapChanged.connect(self._on_map_changed)
        self.map_selector.channelEnabledChanged.connect(self._on_channel_enabled)
        self.studio.viewport3d.loadingChanged.connect(self.map_selector.set_loading)

        self.classic_toolbar.modeChanged.connect(self._on_classic_view_mode_changed)
        self.classic_toolbar.tilesChanged.connect(self.classic.viewport.set_tiles)
        self.classic_toolbar.fitRequested.connect(self.classic.viewport.fit_to_view)
        self.classic_toolbar.zoomRequested.connect(self.classic.viewport.set_zoom)
        self.classic_toolbar.guidesChanged.connect(self.classic.viewport.set_show_guides)
        self.classic.viewport.zoomChanged.connect(self.classic_toolbar.set_zoom_text)

    def _toggle_mode(self):
        if self.stack.currentIndex() == 0:
            self.studioModeRequested.emit()
        else:
            self.classicModeRequested.emit()

    def _on_classic_view_mode_changed(self, mode):
        self._mode = mode
        self.classic.viewport.set_mode(mode)

    def _on_channel_enabled(self, name: str, enabled: bool):
        """Forward to 3D viewport and update the 2D classic view if needed."""
        self.studio.viewport3d.set_channel_enabled(name, enabled)
        # If the channel being disabled is what the 2D classic view is showing,
        # fall back to displaying Base Color in the classic viewport.
        if not enabled and self.map_selector.checked_name() == name:
            base_pix = self.maps.get("Base Color")
            self.classic.viewport.set_after_pixmap(base_pix)
            self.studio.viewport2d.set_after_pixmap(base_pix)

    def set_workspace(self, index):
        self.stack.setCurrentIndex(index)
        is_studio = index == 1
        self.mode_badge.setText("STUDIO MODE" if is_studio else "CLASSIC MODE")
        self.toggle_btn.setText("Switch to Classic" if is_studio else "Switch to Studio")
        self._update_bottom_bar()

    def set_mode(self, mode):
        self._mode = mode
        self.classic.viewport.set_mode(mode)
        self.classic_toolbar.set_active_mode(mode)
        self._update_bottom_bar()

    def _update_bottom_bar(self):
        self.classic_toolbar.setVisible(self.stack.currentIndex() == 0)

    def fit_to_view(self):
        self.classic.viewport.fit_to_view()
        self.studio.viewport2d.fit_to_view()

    def set_before_image(self, img):
        pix = numpy_to_pixmap(img)
        self.maps["Base Color"] = pix
        self.classic.viewport.set_before_image(img)
        self.studio.viewport3d.set_material_map("Base Color", img)
        self.studio.viewport2d.set_before_image(img)
        self.map_selector.set_channel("Base Color", pix)
        if self.map_selector.checked_name() == "Base Color":
            self.studio.viewport2d.set_after_pixmap(pix)

    def set_after_image(self, img):
        pix = numpy_to_pixmap(img)
        self.maps["Base Color"] = pix
        self.classic.viewport.set_after_image(img)
        self.studio.viewport3d.set_material_map("Base Color", img)
        self.studio.viewport2d.set_after_image(img)
        self.map_selector.set_channel("Base Color", pix)
        if self.map_selector.checked_name() == "Base Color":
            self.studio.viewport2d.set_after_pixmap(pix)

    def set_map(self, name, img):
        if name == "Displacement":
            name = "Height"
        pix = numpy_to_pixmap(img)
        self.maps[name] = pix
        self.map_selector.set_channel(name, pix)
        self.studio.viewport3d.set_material_map(name, img)
        if name == self.map_selector.checked_name():
            self.studio.viewport2d.set_after_pixmap(pix)
            self.classic.viewport.set_after_pixmap(pix)
        if name == "Base Color":
            self.classic.viewport.set_after_pixmap(pix)

    def _on_map_changed(self, name):
        pix = self.maps.get(name)
        self.studio.viewport2d.set_after_pixmap(pix)
        self.classic.viewport.set_after_pixmap(pix)
        self.studio.viewport3d.isolate_channel(None if name == "Base Color" else name)

    def select_map(self, name):
        normalized = "Base Color" if name.lower() in ("basecolor", "base color", "albedo") else name
        self.map_selector.select(normalized)

    def set_studio_map_selector_visible(self, visible):
        self.map_selector.setVisible(visible)

    def set_delighted_image(self, image):
        pass

    def set_tiles(self, count):
        self.classic.viewport.set_tiles(count)
        self.studio.viewport2d.set_tiles(count)
        self.studio.viewport3d.set_tiling(count)

    def set_show_guides(self, show):
        self.classic.viewport.set_show_guides(show)
        self.studio.viewport2d.set_show_guides(show)

    def cleanup(self):
        self.studio.viewport3d.cleanup()
