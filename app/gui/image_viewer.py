"""
Image viewer widget with edge-focused comparison and tiled preview.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QTabWidget, QPushButton, QSlider
)
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent, QWheelEvent, QRegion
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
    """Canvas for edge-focused comparison view (formerly split view)."""
    
    zoomChanged = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._before_pixmap = None
        self._after_pixmap = None
        self._blend_factor = 0.5  # 0.0 (Original) to 1.0 (Processed)
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._last_mouse_pos = None
        self._dragging_blend = False
        self._dragging_pan = False
        self._show_guides = True
        self._seam_width_pct = 0.10  # 10% coverage
        
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
            self.zoomChanged.emit(self._zoom)
            self.update()
            
    def set_show_guides(self, show):
        """Toggle seam guides."""
        self._show_guides = show
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
        
        # --- Edge-Focused Comparison Logic ---
        
        # 1. Define Seam Regions (Borders + Center Cross)
        pad = int(min(scaled_w, scaled_h) * self._seam_width_pct / 2)
        pad = max(4, pad) 
        
        cx = x + scaled_w // 2
        cy = y + scaled_h // 2
        
        # Create regions for Seams using QRegion union
        seam_region = QRegion()
        
        # Borders (Outer seam bands)
        seam_region += QRect(x, y, scaled_w, pad) # Top
        seam_region += QRect(x, y + scaled_h - pad, scaled_w, pad) # Bottom
        seam_region += QRect(x, y, pad, scaled_h) # Left
        seam_region += QRect(x + scaled_w - pad, y, pad, scaled_h) # Right
        
        # Center Cross (Offset Intersection)
        seam_region += QRect(x, cy - pad, scaled_w, pad*2) # Horizontal Center
        seam_region += QRect(cx - pad, y, pad*2, scaled_h) # Vertical Center
        
        # 2. Draw Processed Image (Background) everywhere
        if self._after_pixmap:
            painter.drawPixmap(target_rect, self._after_pixmap)
            
        # 3. Dim the Non-Seam Areas
        # Calculate the non-seam region by subtracting seam_region from image rect
        full_image_region = QRegion(target_rect)
        dim_region = full_image_region - seam_region
        
        painter.setClipRegion(dim_region)
        painter.fillRect(target_rect, QColor(0, 0, 0, 180)) # Dimmed overlay
        painter.setClipping(False)
        
        # 4. Draw Original Image (Mixed in Seam Regions)
        if self._before_pixmap:
            # Opacity based on blend factor. 
            # t=0 (Original) -> 100% Original
            # t=1 (Processed) -> 0% Original (showing Processed underneath)
            opacity = 1.0 - self._blend_factor
            
            # Only draw/clip if we have some opacity
            if opacity > 0.01:
                painter.setClipRegion(seam_region)
                painter.setOpacity(opacity)
                painter.drawPixmap(target_rect, self._before_pixmap)
                painter.setOpacity(1.0)
                painter.setClipping(False)
        
        # 5. Draw Visual Guides
        if self._show_guides:
            pen = QPen(QColor(0, 152, 255, 128), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            
            # Draw outlines of the seam bands
            # Inner border lines
            painter.drawLine(x + pad, y + pad, x + scaled_w - pad, y + pad) 
            painter.drawLine(x + pad, y + scaled_h - pad, x + scaled_w - pad, y + scaled_h - pad) 
            painter.drawLine(x + pad, y + pad, x + pad, y + scaled_h - pad) 
            painter.drawLine(x + scaled_w - pad, y + pad, x + scaled_w - pad, y + scaled_h - pad)
            
            # Center cross lines
            painter.drawLine(x, cy - pad, x + scaled_w, cy - pad)
            painter.drawLine(x, cy + pad, x + scaled_w, cy + pad)
            painter.drawLine(cx - pad, y, cx - pad, y + scaled_h)
            painter.drawLine(cx + pad, y, cx + pad, y + scaled_h)

        # 6. Draw Blend Slider UI
        slider_h = 6
        slider_y = self.height() - 30
        slider_margin_x = 40
        slider_rect = QRect(slider_margin_x, slider_y, self.width() - slider_margin_x*2, slider_h)
        
        # Background track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(60, 60, 60))
        painter.drawRoundedRect(slider_rect, slider_h//2, slider_h//2)
        
        # Active track (blue)
        handle_x = int(slider_rect.left() + (slider_rect.width() * self._blend_factor))
        if handle_x > slider_rect.left():
            active_rect = QRect(slider_rect.left(), slider_y, handle_x - slider_rect.left(), slider_h)
            painter.setBrush(QColor(0, 152, 255))
            painter.drawRoundedRect(active_rect, slider_h//2, slider_h//2)
        
        # Handle
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(QPoint(handle_x, slider_y + slider_h//2), 8, 8)
        
        # Labels
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(painter.font())
        label_y = slider_y - 12
        painter.drawText(slider_rect.left(), label_y, "Original Edges")
        
        # Right align 'Processed Edges'
        fm = painter.fontMetrics()
        proc_text = "Processed Edges"
        proc_w = fm.horizontalAdvance(proc_text)
        painter.drawText(slider_rect.right() - proc_w, label_y, proc_text)
        
        # Title
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(0, 20, 0, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, 
                        "Edge Seam Visualization")
    
    def _get_slider_rect(self):
        """Get the slider interaction area."""
        slider_y = self.height() - 50
        return QRect(0, slider_y, self.width(), 50)
    
    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self._get_slider_rect().contains(pos):
                self._dragging_blend = True
                self._update_blend_from_pos(pos)
            else:
                self._dragging_pan = True
                self._last_mouse_pos = pos
    
    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        
        if self._dragging_blend:
            self._update_blend_from_pos(pos)
        
        elif self._dragging_pan and self._last_mouse_pos:
            delta = pos - self._last_mouse_pos
            self._pan_offset += delta
            self._last_mouse_pos = pos
            self.update()
            
        if self._get_slider_rect().contains(pos):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _update_blend_from_pos(self, pos):
        """Update blend factor from mouse position."""
        slider_margin_x = 40
        slider_rect = QRect(slider_margin_x, 0, self.width() - slider_margin_x*2, 0)
        relative_x = pos.x() - slider_rect.left()
        if slider_rect.width() > 0:
            self._blend_factor = max(0.0, min(1.0, relative_x / slider_rect.width()))
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_blend = False
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
            self.zoomChanged.emit(self._zoom)
            self.update()


class TiledCanvas(QWidget):
    """Canvas for displaying tiled preview to verify seamlessness."""
    
    zoomChanged = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._tiles = 2
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._last_mouse_pos = None
        self._dragging = False
        self._logical_size = None
        self._show_guides = True
        
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)
    
    def set_show_guides(self, show):
        """Toggle grid lines."""
        self._show_guides = show
        self.update()
    
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
            self.zoomChanged.emit(self._zoom)
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
        
        # Draw grid lines to show tile boundaries (only if guides enabled)
        if self._show_guides:
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
            self.zoomChanged.emit(self._zoom)
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
        
        # Initialize views
        self.split_view = SplitViewCanvas()
        self.split_view.zoomChanged.connect(self._update_zoom_label)
        
        # Tiled preview tab
        tiled_container = QWidget()
        tiled_layout = QVBoxLayout(tiled_container)
        tiled_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tiled_view = TiledCanvas()
        self.tiled_view.zoomChanged.connect(self._update_zoom_label)
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
        
        # Add tabs in requested order (Tiled FIRST, then Comparison)
        self.tabs.addTab(tiled_container, "Tiled Preview")
        self.tabs.addTab(self.split_view, "Seam Comparison")
        
        layout.addWidget(self.tabs)
        
        # Bottom toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 4, 8, 4)
        
        self.fit_btn = QPushButton("Fit to View")
        self.fit_btn.setMaximumWidth(100)
        self.fit_btn.clicked.connect(self._fit_current_view)
        toolbar.addWidget(self.fit_btn)
        
        # Toggle Guide Button
        self.guide_btn = QPushButton("Guides")
        self.guide_btn.setCheckable(True)
        self.guide_btn.setChecked(True)
        self.guide_btn.setMaximumWidth(80)
        self.guide_btn.clicked.connect(self._on_guide_toggled)
        toolbar.addWidget(self.guide_btn)
        
        toolbar.addStretch()
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #808080;")
        toolbar.addWidget(self.zoom_label)
        
        layout.addLayout(toolbar)
        
        # Connect tab signal LAST to prevent startup crashes when label doesn't exist
        self.tabs.currentChanged.connect(self._on_tab_changed)
    
    def _update_zoom_label(self, zoom):
        """Update the zoom percentage label."""
        self.zoom_label.setText(f"{int(zoom * 100)}%")
        
    def _on_tab_changed(self, index):
        """Update zoom label when switching tabs."""
        if index == 0:
            self._update_zoom_label(self.tiled_view._zoom)
        else:
            self._update_zoom_label(self.split_view._zoom)

    def _on_tile_count_changed(self, value):
        self.tiled_view.set_tiles(value)
        self.tile_count_label.setText(f"{value}x{value}")
    
    def _on_guide_toggled(self, checked):
        self.split_view.set_show_guides(checked)
        self.tiled_view.set_show_guides(checked)
    
    def _fit_current_view(self):
        # Index 0 is now Tiled Preview
        if self.tabs.currentIndex() == 0:
            self.tiled_view.fit_to_view()
        else:
            self.split_view.fit_to_view()
    
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
