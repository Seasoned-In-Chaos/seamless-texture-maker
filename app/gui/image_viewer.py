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
from app.core.normal_generator import compute_lighting_jit


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
        self._tiles = 2
        
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
            total_w = pixmap.width() * self._tiles
            total_h = pixmap.height() * self._tiles
            w_ratio = self.width() / total_w
            h_ratio = self.height() / total_h
            self._zoom = min(w_ratio, h_ratio) * 0.95
            self._pan_offset = QPoint(0, 0)
            self.zoomChanged.emit(self._zoom)
            self.update()
            
    def set_tiles(self, count):
        """Set number of tiles."""
        self._tiles = max(1, min(6, count))
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
        
        # 1. Calculate tile size
        # Logical size is original image size
        tile_w = int(logical_w * self._zoom)
        tile_h = int(logical_h * self._zoom)
        scaled_w = tile_w * self._tiles
        scaled_h = tile_h * self._tiles
        
        # Center the grid
        start_x = (self.width() - scaled_w) // 2 + self._pan_offset.x()
        start_y = (self.height() - scaled_h) // 2 + self._pan_offset.y()
        
        # 2. Draw Processed Image (Background Grid)
        if self._after_pixmap:
            for ty in range(self._tiles):
                for tx in range(self._tiles):
                    tile_rect = QRect(start_x + tx * tile_w, start_y + ty * tile_h, tile_w, tile_h)
                    painter.drawPixmap(tile_rect, self._after_pixmap)
            
        # 3. Draw Original Image (Mixed Grid) with opacity
        if self._before_pixmap:
            opacity = 1.0 - self._blend_factor
            if opacity > 0.01:
                painter.setOpacity(opacity)
                for ty in range(self._tiles):
                    for tx in range(self._tiles):
                        tile_rect = QRect(start_x + tx * tile_w, start_y + ty * tile_h, tile_w, tile_h)
                        painter.drawPixmap(tile_rect, self._before_pixmap)
                painter.setOpacity(1.0)
        
        # 4. Draw Seam Guides (Optional)
        if self._show_guides:
            pen = QPen(QColor(0, 152, 255, 80), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            
            # Draw internal grid lines for any tile count
            for tx in range(1, self._tiles):
                lx = start_x + tx * tile_w
                painter.drawLine(lx, start_y, lx, start_y + scaled_h)
            
            for ty in range(1, self._tiles):
                ly = start_y + ty * tile_h
                painter.drawLine(start_x, ly, start_x + scaled_w, ly)
            
            # Draw outer border
            painter.drawRect(start_x, start_y, scaled_w, scaled_h)

        # 5. Draw Blend Slider UI
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
        
        # Title and Header Line
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


class NormalPreviewCanvas(QWidget):
    """Canvas for real-time lighting preview of normal maps."""
    
    zoomChanged = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._normals_raw = None # (H, W, 3) float32
        self._pixmap = None      # Standard RGB Normal Map
        self._shading_img = None # Result of lighting
        self._light_dir = np.array([0.5, 0.5, 1.0], dtype=np.float32)
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._last_mouse_pos = None
        self._dragging = False
        self._preview_shape = "flat plane"
        self._sphere_normals = None
        
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)

    def set_preview_shape(self, shape):
        """Set preview shape: 'flat plane' or 'sphere'."""
        self._preview_shape = shape.lower()
        self._update_lighting()
        self.update()
        
    def set_normals(self, normals_raw, normal_map_pixmap=None):
        """Set the raw normals and the packed pixmap."""
        self._normals_raw = normals_raw
        self._pixmap = normal_map_pixmap
        self._update_lighting()
        self.update()

    def _get_sphere_normals(self, h, w):
        """Generate normals for a sphere."""
        y, x = np.ogrid[:h, :w]
        # Normalize to -1, 1
        ny = (y - h/2.0) / (h/2.0)
        nx = (x - w/2.0) / (w/2.0)
        r2 = nx*nx + ny*ny
        mask = r2 < 0.95 # Slightly smaller than full for anti-aliasing feel
        nz = np.sqrt(np.clip(1.0 - r2, 0, 1))
        
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[..., 0] = nx * mask
        normals[..., 1] = ny * mask
        normals[..., 2] = nz * mask + (1.0 - mask) # 1.0 for background (flat)
        return normals
        
    def _update_lighting(self):
        try:
            if self._normals_raw is None:
                self._shading_img = None
                return
                
            h, w = self._normals_raw.shape[:2]
            
            # Decide which normals to use
            if self._preview_shape == "sphere":
                # Generate sphere normals if needed or resize
                if self._sphere_normals is None or self._sphere_normals.shape[:2] != (h, w):
                    self._sphere_normals = self._get_sphere_normals(h, w)
                
                # Combine: Add the texture normals to the sphere surface
                # This is a simplification but works well for artist tools
                active_normals = self._sphere_normals + self._normals_raw * 0.3
                # Re-normalize
                mags = np.sqrt(np.sum(active_normals**2, axis=2))[..., np.newaxis]
                active_normals /= np.maximum(mags, 1e-6)
            else:
                active_normals = self._normals_raw

            # 1. Compute Shading
            ldir = self._light_dir / np.linalg.norm(self._light_dir)
            shading = compute_lighting_jit(active_normals, ldir)
            
            # 2. Render to QImage
            base_gray = 180
            shaded_data = (shading * base_gray).astype(np.uint8)
            
            # Convert grayscale shading to QImage
            self._shading_img = QImage(shaded_data.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        except Exception as e:
            print(f"Error updating lighting: {e}")
            import traceback
            traceback.print_exc()
            self._shading_img = None
        
    def fit_to_view(self):
        if self._normals_raw is not None:
            h, w = self._normals_raw.shape[:2]
            w_ratio = self.width() / w
            h_ratio = self.height() / h
            self._zoom = min(w_ratio, h_ratio) * 0.95
            self._pan_offset = QPoint(0, 0)
            self.zoomChanged.emit(self._zoom)
            self.update()
            
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        if self._shading_img is None:
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Generate Normal Map to see preview")
            return
            
        # Draw the shaded image
        h, w = self._shading_img.height(), self._shading_img.width()
        sw, sh = int(w * self._zoom), int(h * self._zoom)
        
        tx = (self.width() - sw) // 2 + self._pan_offset.x()
        ty = (self.height() - sh) // 2 + self._pan_offset.y()
        
        target_rect = QRect(tx, ty, sw, sh)
        painter.drawImage(target_rect, self._shading_img)
        
        # Draw light indicator (small circle showing light direction)
        painter.setPen(QColor(255, 255, 0, 150))
        painter.setBrush(QColor(255, 255, 0, 100))
        # Project light dir to 2D
        lx = tx + sw // 2 + int(self._light_dir[0] * sw // 4)
        ly = ty + sh // 2 - int(self._light_dir[1] * sh // 4)
        painter.drawEllipse(QPoint(lx, ly), 5, 5)
        
        # Title
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(self.rect().adjusted(0, 10, 0, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, 
                        "Interactive Lighting Preview (Move mouse to relight)")

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        
        # If not dragging, update light direction based on mouse position
        if not self._dragging:
            # Calculate light direction relative to center of widget
            cx, cy = self.width() // 2, self.height() // 2
            dx = (pos.x() - cx) / (self.width() / 2)
            dy = -(pos.y() - cy) / (self.height() / 2) # Qt Y is down
            
            # Clamp and set Z
            self._light_dir[0] = np.clip(dx, -1.0, 1.0)
            self._light_dir[1] = np.clip(dy, -1.0, 1.0)
            self._light_dir[2] = 0.5 + 0.5 * (1.0 - min(1.0, np.sqrt(dx*dx + dy*dy)))
            
            self._update_lighting()
            self.update()
        else:
            # Panning logic
            if self._last_mouse_pos:
                delta = pos - self._last_mouse_pos
                self._pan_offset += delta
                self._last_mouse_pos = pos
                self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # Shift+Click or similar for panning? No, let's use Right Click for Pan
            pass
        elif event.button() == Qt.MouseButton.RightButton:
            self._dragging = True
            self._last_mouse_pos = event.position().toPoint()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False


class ImageViewer(QWidget):
    """Main image viewer with tabs for different view modes."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # 1. Tiled Preview Tab
        tiled_container = QWidget()
        tiled_layout = QVBoxLayout(tiled_container)
        tiled_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tiled_view = TiledCanvas()
        self.tiled_view.zoomChanged.connect(self._update_zoom_label)
        tiled_layout.addWidget(self.tiled_view)
        
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
        
        fit_btn = QPushButton("Fit")
        fit_btn.setMaximumWidth(60)
        fit_btn.clicked.connect(self.tiled_view.fit_to_view)
        tile_controls.addWidget(fit_btn)
        tiled_layout.addLayout(tile_controls)
        
        # 2. Seam Comparison Tab
        comp_container = QWidget()
        comp_layout = QVBoxLayout(comp_container)
        comp_layout.setContentsMargins(0, 0, 0, 0)
        
        self.split_view = SplitViewCanvas()
        self.split_view.zoomChanged.connect(self._update_zoom_label)
        comp_layout.addWidget(self.split_view)
        
        comp_tile_controls = QHBoxLayout()
        comp_tile_controls.setContentsMargins(8, 4, 8, 4)
        comp_tile_controls.addWidget(QLabel("Tiles:"))
        self.comp_tile_slider = QSlider(Qt.Orientation.Horizontal)
        self.comp_tile_slider.setMinimum(1)
        self.comp_tile_slider.setMaximum(6)
        self.comp_tile_slider.setValue(2)
        self.comp_tile_slider.setMaximumWidth(150)
        self.comp_tile_slider.valueChanged.connect(self._on_comp_tile_count_changed)
        comp_tile_controls.addWidget(self.comp_tile_slider)
        self.comp_tile_count_label = QLabel("2x2")
        comp_tile_controls.addWidget(self.comp_tile_count_label)
        comp_tile_controls.addStretch()
        
        comp_fit_btn = QPushButton("Fit")
        comp_fit_btn.setMaximumWidth(60)
        comp_fit_btn.clicked.connect(self.split_view.fit_to_view)
        comp_tile_controls.addWidget(comp_fit_btn)
        comp_layout.addLayout(comp_tile_controls)
        
        # 3. Normal Map Preview Tab
        self.normal_view = NormalPreviewCanvas()
        self.normal_view.zoomChanged.connect(self._update_zoom_label)
        
        # Add tabs
        self.tabs.addTab(tiled_container, "Tiled Preview")
        self.tabs.addTab(comp_container, "Seam Comparison")
        self.tabs.addTab(self.normal_view, "3D Preview")
        
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

    def _on_comp_tile_count_changed(self, value):
        self.split_view.set_tiles(value)
        self.comp_tile_count_label.setText(f"{value}x{value}")
    
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
