"""
Image viewer widget with before/after split view and tiled preview.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QTabWidget, QPushButton, QSlider
)
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent, QWheelEvent
import numpy as np
import cv2


def numpy_to_qimage(img):
    """Convert numpy array (BGR) to QImage."""
    if img is None:
        return None
    
    if len(img.shape) == 2:
        # Grayscale
        h, w = img.shape
        bytes_per_line = w
        return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
    else:
        h, w, c = img.shape
        if c == 4:
            # BGRA to RGBA
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            bytes_per_line = 4 * w
            return QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        else:
            # BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * w
            return QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)


class ImageCanvas(QWidget):
    """Canvas for displaying an image with pan and zoom."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._last_mouse_pos = None
        self._dragging = False
        
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)
    
    def set_image(self, image):
        """Set image from numpy array."""
        if image is None:
            self._pixmap = None
        else:
            qimg = numpy_to_qimage(image)
            if qimg:
                self._pixmap = QPixmap.fromImage(qimg)
        self.update()
    
    def set_pixmap(self, pixmap):
        """Set pixmap directly."""
        self._pixmap = pixmap
        self.update()
    
    def fit_to_view(self):
        """Fit image to view."""
        if self._pixmap:
            w_ratio = self.width() / self._pixmap.width()
            h_ratio = self.height() / self._pixmap.height()
            self._zoom = min(w_ratio, h_ratio) * 0.95
            self._pan_offset = QPoint(0, 0)
            self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Dark background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        if self._pixmap:
            # Calculate scaled size
            scaled_w = int(self._pixmap.width() * self._zoom)
            scaled_h = int(self._pixmap.height() * self._zoom)
            
            # Center position with pan offset
            x = (self.width() - scaled_w) // 2 + self._pan_offset.x()
            y = (self.height() - scaled_h) // 2 + self._pan_offset.y()
            
            # Draw image
            target_rect = QRect(x, y, scaled_w, scaled_h)
            painter.drawPixmap(target_rect, self._pixmap)
        else:
            # Draw placeholder text
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
    
    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap:
            delta = event.angleDelta().y()
            zoom_factor = 1.1 if delta > 0 else 0.9
            new_zoom = self._zoom * zoom_factor
            
            # Clamp zoom
            new_zoom = max(0.1, min(10.0, new_zoom))
            self._zoom = new_zoom
            self.update()
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_mouse_pos = event.position().toPoint()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging and self._last_mouse_pos:
            current_pos = event.position().toPoint()
            delta = current_pos - self._last_mouse_pos
            self._pan_offset += delta
            self._last_mouse_pos = current_pos
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_mouse_pos = None


class SplitViewCanvas(QWidget):
    """Canvas for before/after split view comparison."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._before_pixmap = None
        self._after_pixmap = None
        self._split_position = 0.5  # 0.0 to 1.0
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._last_mouse_pos = None
        self._dragging_split = False
        self._dragging_pan = False
        
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)
    
    def set_before_image(self, image):
        """Set before image from numpy array."""
        if image is None:
            self._before_pixmap = None
        else:
            qimg = numpy_to_qimage(image)
            if qimg:
                self._before_pixmap = QPixmap.fromImage(qimg)
        self.update()
    
    def set_after_image(self, image):
        """Set after image from numpy array."""
        if image is None:
            self._after_pixmap = None
        else:
            qimg = numpy_to_qimage(image)
            if qimg:
                self._after_pixmap = QPixmap.fromImage(qimg)
        self.update()
    
    def fit_to_view(self):
        """Fit image to view."""
        pixmap = self._before_pixmap or self._after_pixmap
        if pixmap:
            w_ratio = self.width() / pixmap.width()
            h_ratio = self.height() / pixmap.height()
            self._zoom = min(w_ratio, h_ratio) * 0.95
            self._pan_offset = QPoint(0, 0)
            self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Dark background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Determine logical size from BEFORE image (Original)
        # If not available, fall back to AFTER image
        ref_pixmap = self._before_pixmap or self._after_pixmap
        
        if ref_pixmap is None:
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
            return
            
        logical_w = ref_pixmap.width()
        logical_h = ref_pixmap.height()
        
        # Calculate scaled size based on Logical Size
        scaled_w = int(logical_w * self._zoom)
        scaled_h = int(logical_h * self._zoom)
        
        # Center position with pan offset
        x = (self.width() - scaled_w) // 2 + self._pan_offset.x()
        y = (self.height() - scaled_h) // 2 + self._pan_offset.y()
        
        target_rect = QRect(x, y, scaled_w, scaled_h)
        
        # Calculate split position in widget coordinates
        split_x = x + int(scaled_w * self._split_position)
        
        # Draw before image (left side)
        if self._before_pixmap:
            painter.setClipRect(QRect(x, y, int(scaled_w * self._split_position), scaled_h))
            # Draw Before (always full res)
            painter.drawPixmap(target_rect, self._before_pixmap)
        
        # Draw after image (right side)
        if self._after_pixmap:
             # Calculate clip rect for right side
             clip_w = scaled_w - int(scaled_w * self._split_position)
             if clip_w > 0:
                 painter.setClipRect(QRect(split_x, y, clip_w, scaled_h))
                 # Draw After (possibly low res/preview) scaled to target_rect
                 painter.drawPixmap(target_rect, self._after_pixmap)
        
        painter.setClipping(False)
        
        # Draw split line
        pen = QPen(QColor(0, 152, 255), 2)
        painter.setPen(pen)
        painter.drawLine(split_x, y, split_x, y + scaled_h)
        
        # Draw split handle
        handle_y = y + scaled_h // 2
        painter.setBrush(QColor(0, 152, 255))
        painter.drawEllipse(QPoint(split_x, handle_y), 8, 8)
        
        # Draw labels
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        
        if self._before_pixmap:
            painter.drawText(x + 10, y + 25, "BEFORE")
        if self._after_pixmap:
            painter.drawText(split_x + 10, y + 25, "AFTER")
    
    def _get_split_handle_rect(self):
        """Get the clickable area around the split line."""
        pixmap = self._before_pixmap or self._after_pixmap
        if pixmap is None:
            return QRect()
        
        scaled_w = int(pixmap.width() * self._zoom)
        scaled_h = int(pixmap.height() * self._zoom)
        x = (self.width() - scaled_w) // 2 + self._pan_offset.x()
        y = (self.height() - scaled_h) // 2 + self._pan_offset.y()
        split_x = x + int(scaled_w * self._split_position)
        
        return QRect(split_x - 10, y, 20, scaled_h)
    
    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        
        if event.button() == Qt.MouseButton.LeftButton:
            handle_rect = self._get_split_handle_rect()
            if handle_rect.contains(pos):
                self._dragging_split = True
            else:
                self._dragging_pan = True
                self._last_mouse_pos = pos
    
    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        
        if self._dragging_split:
            pixmap = self._before_pixmap or self._after_pixmap
            if pixmap:
                scaled_w = int(pixmap.width() * self._zoom)
                x = (self.width() - scaled_w) // 2 + self._pan_offset.x()
                
                # Calculate new split position
                relative_x = pos.x() - x
                self._split_position = max(0.0, min(1.0, relative_x / scaled_w))
                self.update()
        
        elif self._dragging_pan and self._last_mouse_pos:
            delta = pos - self._last_mouse_pos
            self._pan_offset += delta
            self._last_mouse_pos = pos
            self.update()
        
        # Update cursor
        handle_rect = self._get_split_handle_rect()
        if handle_rect.contains(pos):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_split = False
            self._dragging_pan = False
            self._last_mouse_pos = None
    
    def wheelEvent(self, event: QWheelEvent):
        pixmap = self._before_pixmap or self._after_pixmap
        if pixmap:
            delta = event.angleDelta().y()
            zoom_factor = 1.1 if delta > 0 else 0.9
            new_zoom = self._zoom * zoom_factor
            new_zoom = max(0.1, min(10.0, new_zoom))
            self._zoom = new_zoom
            self.update()


class TiledCanvas(QWidget):
    """Canvas for displaying tiled preview to verify seamlessness."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._tiles = 2
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._last_mouse_pos = None
        self._dragging = False
        self._logical_size = None
        
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)
    
    def set_image(self, image, logical_size=None):
        """Set image from numpy array with optional logical size."""
        self._logical_size = logical_size
        if image is None:
            self._pixmap = None
        else:
            qimg = numpy_to_qimage(image)
            if qimg:
                self._pixmap = QPixmap.fromImage(qimg)
        self.update()
    
    def set_tiles(self, count):
        """Set number of tiles."""
        self._tiles = max(1, min(6, count))
        self.update()
    
    def fit_to_view(self):
        """Fit tiled image to view."""
        if self._pixmap:
            total_w = self._pixmap.width() * self._tiles
            total_h = self._pixmap.height() * self._tiles
            w_ratio = self.width() / total_w
            h_ratio = self.height() / total_h
            self._zoom = min(w_ratio, h_ratio) * 0.95
            self._pan_offset = QPoint(0, 0)
            self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Dark background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        if self._pixmap is None:
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
            return
        
        # Calculate scaled tile size based on logical size
        if self._logical_size:
            base_w, base_h = self._logical_size
        else:
            base_w = self._pixmap.width()
            base_h = self._pixmap.height()
            
        tile_w = int(base_w * self._zoom)
        tile_h = int(base_h * self._zoom)
        total_w = tile_w * self._tiles
        total_h = tile_h * self._tiles
        
        # Starting position (centered with pan)
        start_x = (self.width() - total_w) // 2 + self._pan_offset.x()
        start_y = (self.height() - total_h) // 2 + self._pan_offset.y()
        
        # Draw tiles
        for ty in range(self._tiles):
            for tx in range(self._tiles):
                x = start_x + tx * tile_w
                y = start_y + ty * tile_h
                target_rect = QRect(x, y, tile_w, tile_h)
                painter.drawPixmap(target_rect, self._pixmap)
        
        # Draw grid lines to show tile boundaries
        pen = QPen(QColor(0, 152, 255, 100), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        
        for tx in range(1, self._tiles):
            x = start_x + tx * tile_w
            painter.drawLine(x, start_y, x, start_y + total_h)
        
        for ty in range(1, self._tiles):
            y = start_y + ty * tile_h
            painter.drawLine(start_x, y, start_x + total_w, y)
    
    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap:
            delta = event.angleDelta().y()
            zoom_factor = 1.1 if delta > 0 else 0.9
            new_zoom = self._zoom * zoom_factor
            new_zoom = max(0.05, min(5.0, new_zoom))
            self._zoom = new_zoom
            self.update()
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_mouse_pos = event.position().toPoint()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging and self._last_mouse_pos:
            current_pos = event.position().toPoint()
            delta = current_pos - self._last_mouse_pos
            self._pan_offset += delta
            self._last_mouse_pos = current_pos
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_mouse_pos = None


class ImageViewer(QWidget):
    """Main image viewer with tabs for different view modes."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # Split view tab
        self.split_view = SplitViewCanvas()
        self.tabs.addTab(self.split_view, "Before / After")
        
        # Tiled preview tab
        tiled_container = QWidget()
        tiled_layout = QVBoxLayout(tiled_container)
        tiled_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tiled_view = TiledCanvas()
        tiled_layout.addWidget(self.tiled_view)
        
        # Tile count slider
        tile_controls = QHBoxLayout()
        tile_controls.setContentsMargins(8, 4, 8, 4)
        tile_controls.addWidget(QLabel("Tiles:"))
        self.tile_slider = QSlider(Qt.Orientation.Horizontal)
        self.tile_slider.setMinimum(1)
        self.tile_slider.setMaximum(6)
        self.tile_slider.setValue(2)
        self.tile_slider.setMaximumWidth(150)
        self.tile_slider.valueChanged.connect(self._on_tile_count_changed)
        tile_controls.addWidget(self.tile_slider)
        self.tile_count_label = QLabel("2x2")
        tile_controls.addWidget(self.tile_count_label)
        tile_controls.addStretch()
        
        # Fit button for tiled view
        fit_btn = QPushButton("Fit")
        fit_btn.setMaximumWidth(60)
        fit_btn.clicked.connect(self.tiled_view.fit_to_view)
        tile_controls.addWidget(fit_btn)
        
        tiled_layout.addLayout(tile_controls)
        self.tabs.addTab(tiled_container, "Tiled Preview")
        
        layout.addWidget(self.tabs)
        
        # Bottom toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 4, 8, 4)
        
        self.fit_btn = QPushButton("Fit to View")
        self.fit_btn.setMaximumWidth(100)
        self.fit_btn.clicked.connect(self._fit_current_view)
        toolbar.addWidget(self.fit_btn)
        
        toolbar.addStretch()
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #808080;")
        toolbar.addWidget(self.zoom_label)
        
        layout.addLayout(toolbar)
    
    def _on_tile_count_changed(self, value):
        self.tiled_view.set_tiles(value)
        self.tile_count_label.setText(f"{value}x{value}")
    
    def _fit_current_view(self):
        if self.tabs.currentIndex() == 0:
            self.split_view.fit_to_view()
        else:
            self.tiled_view.fit_to_view()
    
    def set_before_image(self, image):
        """Set the original image."""
        self.split_view.set_before_image(image)
    
    def set_after_image(self, image):
        """Set the processed image."""
        self.split_view.set_after_image(image)
        
        # Pass logical size if we have a BEFORE image
        logical_size = None
        if self.split_view._before_pixmap:
            logical_size = (self.split_view._before_pixmap.width(), self.split_view._before_pixmap.height())
            
        self.tiled_view.set_image(image, logical_size=logical_size)
    
    def fit_to_view(self):
        """Fit both views."""
        self.split_view.fit_to_view()
        self.tiled_view.fit_to_view()
