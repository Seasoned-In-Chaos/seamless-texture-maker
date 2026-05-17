"""Main window for SEAMS - Seamless Texture Studio."""
import collections
import os
import time

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QRect, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QPainter, QPen, QPixmap, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .controls import ControlPanel, PreprocessingPanel
from .image_viewer import ImageViewer, numpy_to_pixmap
from .normal_controls import MaterialControlPanel
from .styles import get_dark_theme
from .system_monitor import StatusBarMonitor
from ..core.normal_generator import NormalGenerator
from ..core.seamless import SeamlessProcessor
from ..utils.app_logging import get_logger, log_exception
from ..utils.config import APP_VERSION, load_settings, save_settings
from ..utils.image_io import (
    ensure_writable_directory,
    get_file_info,
    get_format_filter,
    get_output_path,
    load_image,
    sanitize_filename_component,
    save_image,
)


logger = get_logger(__name__)


class ProcessingThread(QThread):
    finished = pyqtSignal(object, float, int)
    error = pyqtSignal(str, int)

    def __init__(self, image, params, generation, parent=None):
        super().__init__(parent)
        self.image = image.copy()
        self.params = params.copy()
        self.generation = generation

    def run(self):
        try:
            t0 = time.time()
            processor = SeamlessProcessor()
            processor.load_image(self.image)
            processor.set_parameters(**self.params)
            result = processor.process()
            self.finished.emit(result, time.time() - t0, self.generation)
        except Exception as exc:
            log_exception(logger, "Texture processing failed", exc)
            self.error.emit(str(exc), self.generation)


class PreviewThread(QThread):
    result_ready = pyqtSignal(object, int)
    error = pyqtSignal(str, int)

    def __init__(self):
        super().__init__()
        self.image = None
        self.params = None
        self.generation = 0
        self._mutex = QMutex()
        self._restart = False
        self._abort = False

    def request(self, image, params, generation):
        self._mutex.lock()
        self.image = image.copy()
        self.params = params.copy()
        self.generation = generation
        self._restart = True
        self._mutex.unlock()
        if not self.isRunning():
            self.start()

    def stop(self):
        self._mutex.lock()
        self._abort = True
        self._restart = True
        self._mutex.unlock()

    def run(self):
        while True:
            self._mutex.lock()
            abort = self._abort
            image = None if self.image is None else self.image.copy()
            params = None if self.params is None else self.params.copy()
            generation = self.generation
            self._restart = False
            self._mutex.unlock()
            if abort or image is None or params is None:
                break
            if params:
                try:
                    processor = SeamlessProcessor()
                    processor.load_image(image)
                    result = processor.process(preview=True, params=params)
                    self._mutex.lock()
                    restart = self._restart or self._abort
                    self._mutex.unlock()
                    if not restart:
                        self.result_ready.emit(result, generation)
                except Exception as exc:
                    log_exception(logger, "Live preview failed", exc)
                    self.error.emit(str(exc), generation)
            self._mutex.lock()
            done = not self._restart or self._abort
            self._mutex.unlock()
            if done:
                break


class MaterialMapThread(QThread):
    maps_ready = pyqtSignal(object, int)
    error = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.params = None
        self.generation = 0
        self._mutex = QMutex()
        self._restart = False
        self._abort = False

    def request(self, image, params, generation):
        self._mutex.lock()
        self.image = image.copy()
        self.params = params.copy()
        self.generation = generation
        self._restart = True
        self._mutex.unlock()
        if not self.isRunning():
            self.start()

    def stop(self):
        self._mutex.lock()
        self._abort = True
        self._restart = True
        self._mutex.unlock()

    def run(self):
        while True:
            self._mutex.lock()
            abort = self._abort
            image = None if self.image is None else self.image.copy()
            params = None if self.params is None else self.params.copy()
            generation = self.generation
            self._restart = False
            self._mutex.unlock()
            if abort or image is None or params is None:
                break
            try:
                maps = NormalGenerator.process(image, **params)
                self._mutex.lock()
                restart = self._restart
                self._mutex.unlock()
                if not restart:
                    self.maps_ready.emit(maps, generation)
            except Exception as exc:
                log_exception(logger, "Material map processing failed", exc)
                self.error.emit(str(exc), generation)

            self._mutex.lock()
            done = not self._restart
            self._mutex.unlock()
            if done:
                break


class ImageLoadThread(QThread):
    loaded = pyqtSignal(str, object, object, object, int)
    error = pyqtSignal(str, str, int)

    def __init__(self, path, generation, parent=None):
        super().__init__(parent)
        self.path = path
        self.generation = generation

    def run(self):
        try:
            image, metadata = load_image(self.path)
            info = get_file_info(self.path)
            self.loaded.emit(self.path, image, metadata, info, self.generation)
        except Exception as exc:
            log_exception(logger, f"Image load failed for {self.path}", exc)
            self.error.emit(self.path, str(exc), self.generation)


class PBRExportThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, base_img, gen_params, data, metadata, parent=None):
        super().__init__(parent)
        self.base_img = base_img.copy()
        self.gen_params = gen_params.copy()
        self.data = data.copy()
        self.metadata = metadata

    def run(self):
        try:
            written = export_pbr_package(self.base_img, self.gen_params, self.data, self.metadata)
            self.finished.emit({"engine": self.data.get("engine", "PBR"), "written": written})
        except Exception as exc:
            log_exception(logger, "PBR export failed", exc)
            self.error.emit(str(exc))


def export_pbr_package(base_img, gen_params, data, metadata=None):
    export_dir = ensure_writable_directory(data.get("directory", ""))
    material = sanitize_filename_component(data.get("name", "Material_01"), "Material_01")
    token = sanitize_filename_component(data.get("renderer_token", "PBR"), "PBR")
    data = data.copy()
    data["name"] = material
    data["renderer_token"] = token

    if data.get("create_subfolder"):
        subfolder = sanitize_filename_component(data.get("subfolder_name") or f"{material}_{token}", f"{material}_{token}")
        export_dir = ensure_writable_directory(os.path.join(export_dir, subfolder))

    h, w = base_img.shape[:2]
    target_res = int(data.get("resolution", max(h, w)))
    if target_res > 0 and target_res != max(h, w):
        scale = target_res / max(h, w)
        export_base = cv2.resize(
            base_img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
        )
    else:
        export_base = base_img

    generated_maps = NormalGenerator.process(export_base, **gen_params)
    maps = {
        "BaseColor": export_base,
        "Normal": generated_maps["Normal"],
        "Roughness": generated_maps["Roughness"],
        "Metallic": generated_maps["Metallic"],
        "AO": generated_maps["AO"],
        "Height": generated_maps["Height"],
        "Opacity": generated_maps["Opacity"],
        "Emissive": generated_maps["Emissive"],
    }
    maps["Displacement"] = generated_maps.get("Displacement", maps["Height"])

    written = write_renderer_maps(export_dir, maps, data, metadata)
    write_material_sidecars(export_dir, written, data)
    return written


def write_renderer_maps(export_dir, maps, data, metadata=None):
    written = {}
    selected = data.get("maps", {})
    token = data.get("renderer_token", "PBR")
    material = data.get("name", "Material")
    ext = export_extension(data.get("format", "png"))

    def save_map(channel, image, suffix=None):
        if not selected.get(channel, False):
            return
        out_suffix = suffix or renderer_channel_suffix(channel, data)
        path = os.path.join(export_dir, f"{material}_{token}_{out_suffix}{ext}")
        save_image(image, path, metadata=metadata)
        written[channel] = path

    if data.get("renderer_key") == "ue5":
        if selected.get("BaseColor", False):
            path = os.path.join(export_dir, f"T_{material}_D{ext}")
            save_image(maps["BaseColor"], path, metadata=metadata)
            written["BaseColor"] = path
        if selected.get("Normal", False):
            path = os.path.join(export_dir, f"T_{material}_N{ext}")
            save_image(maps["Normal"], path, metadata=metadata)
            written["Normal"] = path
        if data.get("packing", False):
            ao = gray_image(maps["AO"])
            roughness = gray_image(maps["Roughness"])
            metallic = gray_image(maps["Metallic"])
            orm = cv2.merge([metallic, roughness, ao])
            path = os.path.join(export_dir, f"T_{material}_ORM{ext}")
            save_image(orm, path, metadata=metadata)
            written["ORM"] = path
        else:
            save_map("AO", maps["AO"])
            save_map("Roughness", maps["Roughness"])
            save_map("Metallic", maps["Metallic"])
        save_map("Height", maps["Height"], "H")
        save_map("Opacity", maps["Opacity"], "Opacity")
        save_map("Emissive", maps["Emissive"], "Emissive")
        return written

    save_map("BaseColor", maps["BaseColor"])
    save_map("Normal", maps["Normal"])
    if uses_glossiness(data):
        glossiness = cv2.bitwise_not(gray_image(maps["Roughness"]))
        save_map("Roughness", glossiness, "Glossiness")
    else:
        save_map("Roughness", maps["Roughness"])
    save_map("Metallic", maps["Metallic"])
    save_map("AO", maps["AO"])
    save_map("Height", maps["Height"])
    save_map("Opacity", maps["Opacity"])
    save_map("Emissive", maps["Emissive"])
    if selected.get("Height", False) or selected.get("Displacement", False):
        displacement = clamp_displacement(maps["Displacement"], data)
        path = os.path.join(export_dir, f"{material}_{token}_Displacement{ext}")
        save_image(displacement, path, metadata=metadata)
        written["Displacement"] = path
    return written


def write_material_sidecars(export_dir, written, data):
    material_options = data.get("material_options", {})
    if not any(material_options.values()) and not any(data.get("options", {}).values()):
        return

    material = data.get("name", "Material")
    engine = data.get("engine", "Generic PBR")
    token = data.get("renderer_token", "PBR")
    payload = {
        "material": material,
        "engine": engine,
        "workflow": data.get("workflow"),
        "normal_format": data.get("normal_format"),
        "color_management": data.get("colorspace"),
        "texture_scale": data.get("texture_scale"),
        "maps": written,
    }
    path = os.path.join(export_dir, f"{material}_{token}_material_preset.json")
    with open(path, "w", encoding="utf-8") as handle:
        import json
        json.dump(payload, handle, indent=2)

    if data.get("renderer_key") in ("corona", "vray"):
        script = os.path.join(export_dir, f"{material}_{token}_3dsmax.ms")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(f"-- {engine} material preset generated by SEAMS\n")
            handle.write(f"-- Material: {material}\n")
            for channel, filepath in written.items():
                handle.write(f"-- {channel}: {filepath}\n")
    elif data.get("renderer_key") == "blender":
        script = os.path.join(export_dir, f"{material}_BLENDER_nodes.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("# Blender Principled BSDF setup generated by SEAMS\n")
            handle.write(f"material_name = {material!r}\n")
            handle.write(f"texture_maps = {written!r}\n")
    elif data.get("renderer_key") == "ue5":
        script = os.path.join(export_dir, f"{material}_UE5_material_instance.txt")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("UE5 material instance setup generated by SEAMS\n")
            handle.write("ORM packing: R=AO, G=Roughness, B=Metallic\n")
            for channel, filepath in written.items():
                handle.write(f"{channel}: {filepath}\n")


def renderer_channel_suffix(channel, data):
    if data.get("renderer_key") == "generic":
        return channel
    return {
        "BaseColor": "BaseColor",
        "Roughness": "Roughness",
        "Metallic": "Metallic",
        "Normal": "Normal",
        "AO": "AO",
        "Height": "Height",
        "Opacity": "Opacity",
        "Emissive": "Emissive",
    }.get(channel, channel)


def export_extension(fmt):
    return {"png": ".png", "tiff": ".tiff", "tif": ".tiff", "tga": ".tga", "exr": ".exr"}.get(str(fmt).lower(), ".png")


def gray_image(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def uses_glossiness(data):
    opts = data.get("options", {})
    return (
        data.get("workflow") == "Specular / Glossiness"
        or opts.get("Convert Roughness to Glossiness", False)
        or opts.get("Reflection glossiness mode", False)
    )


def clamp_displacement(image, data):
    if not data.get("options", {}).get("Clamp displacement values", False):
        return image
    gray = gray_image(image)
    return cv2.cvtColor(np.clip(gray, 8, 247).astype(np.uint8), cv2.COLOR_GRAY2BGR)


class TopBar(QWidget):
    openClicked = pyqtSignal()
    exportClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(68)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(14)

        project = QVBoxLayout()
        project.setSpacing(2)
        kicker = QLabel("PROJECT")
        kicker.setObjectName("MicroLabel")
        self.project_name = QLabel("Untitled Material")
        self.project_name.setObjectName("ProjectName")
        project.addWidget(kicker)
        project.addWidget(self.project_name)
        layout.addLayout(project)
        layout.addStretch()

        for text, tip in [("Undo", "Ctrl+Z"), ("Redo", "Ctrl+Y"), ("Hand", "Pan workspace"), ("Frame", "Fit texture")]:
            b = QPushButton(text)
            b.setObjectName("ToolButton")
            b.setToolTip(tip)
            b.setFixedHeight(34)
            layout.addWidget(b)

        self.status = QLabel("READY")
        self.status.setObjectName("StatusPill")
        layout.addWidget(self.status)

        open_btn = QPushButton("IMPORT")
        open_btn.setObjectName("SecondaryAction")
        open_btn.clicked.connect(self.openClicked.emit)
        layout.addWidget(open_btn)

        export_btn = QPushButton("EXPORT TEXTURE")
        export_btn.setObjectName("HeaderExport")
        export_btn.clicked.connect(self.exportClicked.emit)
        layout.addWidget(export_btn)

    def set_project(self, name):
        self.project_name.setText(name)

    def set_busy(self, busy):
        self.status.setText("PROCESSING" if busy else "READY")
        self.status.setProperty("busy", busy)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)


class NavItem(QWidget):
    clicked = pyqtSignal(str)

    def __init__(self, key, icon, label, shortcut="", parent=None):
        super().__init__(parent)
        self.key = key
        self._active = False
        self.setObjectName("NavItem")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(44)
        self.setToolTip(f"{label}  {shortcut}".strip())
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 12, 0)
        layout.setSpacing(10)
        self.icon = QLabel(icon)
        self.icon.setObjectName("NavIcon")
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon.setFixedWidth(24)
        self.label = QLabel(label)
        self.label.setObjectName("NavLabel")
        self.shortcut = QLabel(shortcut)
        self.shortcut.setObjectName("NavShortcut")
        layout.addWidget(self.icon)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.shortcut)
        self._refresh()

    def set_active(self, active):
        self._active = active
        self.setProperty("active", active)
        self._refresh()

    def _refresh(self):
        for w in [self, self.icon, self.label, self.shortcut]:
            w.style().unpolish(w)
            w.style().polish(w)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.key)


class ToolRail(QWidget):
    navChanged = pyqtSignal(str)
    exportClicked = pyqtSignal()
    openClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._expanded = True
        self._items = {}
        self.setObjectName("ToolRail")
        self.setFixedWidth(236)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        brand = QWidget()
        brand.setObjectName("BrandBlock")
        brand.setFixedHeight(112)
        bl = QVBoxLayout(brand)
        bl.setContentsMargins(18, 18, 18, 12)
        mark = QLabel("SEAMS")
        mark.setObjectName("BrandTitle")
        subtitle = QLabel("SEAMLESS TEXTURE STUDIO")
        subtitle.setObjectName("BrandSubtitle")
        bl.addWidget(mark)
        bl.addWidget(subtitle)
        bl.addStretch()
        layout.addWidget(brand)

        for section, entries in [
            ("IMPORT", [("import", "⊞", "Import Texture", "Ctrl+O")]),
            ("PROCESS", [("delight", "◑", "Delight", "1"), ("seamless", "⧉", "Seamless", "2")]),
            ("MATERIAL", [("material", "⎈", "Material Lab", "3")]),
            ("EXPORT", [("export", "⇲", "Export", "Ctrl+S")]),
        ]:
            label = QLabel(section)
            label.setObjectName("NavSection")
            layout.addWidget(label)
            for key, icon, text, shortcut in entries:
                item = NavItem(key, icon, text, shortcut)
                item.clicked.connect(self._on_clicked)
                self._items[key] = item
                layout.addWidget(item)

        layout.addStretch()

        info_card = QFrame()
        info_card.setObjectName("RailCard")
        il = QVBoxLayout(info_card)
        il.setContentsMargins(16, 14, 16, 14)
        il.setSpacing(10)
        
        title = QLabel("QUICK INFO")
        title.setObjectName("CardTitle")
        il.addWidget(title)
        il.addSpacing(4)
        
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setContentsMargins(0, 0, 0, 0)
        
        self.info_labels = {}
        rows = [
            ("Resolution", "---"),
            ("Color Space", "---"),
            ("Format", "---"),
            ("Bit Depth", "---"),
        ]
        
        for i, (label, default) in enumerate(rows):
            l_lbl = QLabel(label)
            l_lbl.setObjectName("InfoKey")
            v_lbl = QLabel(default)
            v_lbl.setObjectName("InfoValue")
            v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(l_lbl, i, 0)
            grid.addWidget(v_lbl, i, 1)
            self.info_labels[label] = v_lbl
            
        il.addLayout(grid)
        il.addSpacing(8)
        
        size_row = QHBoxLayout()
        sz_lbl = QLabel("File Size")
        sz_lbl.setObjectName("InfoKey")
        self.size_val = QLabel("---")
        self.size_val.setObjectName("InfoValueLarge")
        self.size_val.setAlignment(Qt.AlignmentFlag.AlignRight)
        size_row.addWidget(sz_lbl)
        size_row.addWidget(self.size_val)
        il.addLayout(size_row)
        
        layout.addWidget(info_card)

    def _on_clicked(self, key):
        if key == "import":
            self.openClicked.emit()
        elif key == "export":
            self.exportClicked.emit()
        else:
            self.navChanged.emit(key)

    def set_active(self, key):
        for item_key, item in self._items.items():
            item.set_active(item_key == key)

    def update_info(self, metadata, file_info):
        self.info_labels["Resolution"].setText(f"{metadata.width} x {metadata.height}")
        self.info_labels["Color Space"].setText("sRGB" if metadata.channels >= 3 else "Grayscale")
        self.info_labels["Format"].setText(str(metadata.format or "RAW").upper())
        self.info_labels["Bit Depth"].setText(f"{metadata.bit_depth} Bit")
        self.size_val.setText(file_info['size_str'])


class PreviewCard(QFrame):
    def __init__(self, title, subtitle, parent=None):
        super().__init__(parent)
        self.setObjectName("PreviewCard")
        self._pixmap = None
        self.title = title
        self.subtitle = subtitle
        self.setMinimumWidth(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, image):
        self._pixmap = numpy_to_pixmap(image)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(10, 10, -10, -10)
        p.setPen(QColor(232, 228, 255))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Black))
        p.drawText(QRect(r.left(), r.top(), r.width(), 18), Qt.AlignmentFlag.AlignLeft, self.title.upper())
        p.setPen(QColor(110, 112, 136))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        p.drawText(QRect(r.left(), r.top() + 18, r.width(), 18), Qt.AlignmentFlag.AlignLeft, self.subtitle)
        preview = QRect(r.left(), r.top() + 42, r.width(), max(34, r.height() - 42))
        p.setPen(QPen(QColor(255, 255, 255, 20), 1))
        p.setBrush(QColor(5, 7, 12, 190))
        p.drawRoundedRect(preview, 7, 7)
        if self._pixmap:
            scaled = self._pixmap.scaled(preview.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            x = preview.center().x() - scaled.width() // 2
            y = preview.center().y() - scaled.height() // 2
            p.setClipRect(preview.adjusted(1, 1, -1, -1))
            p.drawPixmap(x, y, scaled)
            p.setClipping(False)
        else:
            p.setPen(QColor(80, 82, 106))
            p.drawText(preview, Qt.AlignmentFlag.AlignCenter, "Awaiting texture")
        p.end()


class MaterialSphereCard(PreviewCard):
    def __init__(self, title, subtitle, parent=None):
        super().__init__(title, subtitle, parent)
        self._albedo = None
        self._normal = None
        self._roughness = None
        self._ao = None
        self._height = None
        self._render_cache = None
        self._render_key = None

    def set_material(self, albedo=None, normal=None, roughness=None, ao=None, height=None):
        self._albedo = albedo
        self._normal = normal
        self._roughness = roughness
        self._ao = ao
        self._height = height
        self._pixmap = None
        self._render_cache = None
        self._render_key = None
        self.update()

    def paintEvent(self, event):
        QFrame.paintEvent(self, event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(10, 10, -10, -10)
        p.setPen(QColor(232, 228, 255))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Black))
        p.drawText(QRect(r.left(), r.top(), r.width(), 18), Qt.AlignmentFlag.AlignLeft, self.title.upper())
        p.setPen(QColor(110, 112, 136))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        p.drawText(QRect(r.left(), r.top() + 18, r.width(), 18), Qt.AlignmentFlag.AlignLeft, self.subtitle)

        preview = QRect(r.left(), r.top() + 42, r.width(), max(34, r.height() - 42))
        p.setPen(QPen(QColor(255, 255, 255, 20), 1))
        p.setBrush(QColor(5, 7, 12, 190))
        p.drawRoundedRect(preview, 7, 7)
        area = preview.adjusted(12, 12, -12, -10)
        if self._albedo is None:
            p.setPen(QColor(80, 82, 106))
            p.drawText(preview, Qt.AlignmentFlag.AlignCenter, "Awaiting material")
            p.end()
            return

        side = max(48, min(area.width(), area.height()))
        shadow = QRect(area.center().x() - side // 2, area.bottom() - max(10, side // 7), side, max(8, side // 6))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 95))
        p.drawEllipse(shadow)

        render = self._render_material(side, side)
        if render:
            x = area.center().x() - render.width() // 2
            y = area.center().y() - render.height() // 2 - max(2, side // 20)
            p.drawPixmap(x, y, render)
        p.end()

    def _render_material(self, width, height):
        key = (
            width,
            height,
            self._image_key(self._albedo),
            self._image_key(self._normal),
            self._image_key(self._roughness),
            self._image_key(self._ao),
            self._image_key(self._height),
        )
        if self._render_cache is not None and key == self._render_key:
            return self._render_cache

        albedo = self._as_bgr(self._albedo)
        if albedo is None:
            return None

        w = max(48, int(width))
        h = max(48, int(height))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        radius = min(w, h) * 0.46
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        sx = (xx - cx) / radius
        sy = (yy - cy) / radius
        radial2 = sx * sx + sy * sy
        mask = radial2 <= 1.0

        out = np.zeros((h, w, 4), dtype=np.uint8)
        if not np.any(mask):
            return None

        geom_z = np.sqrt(np.clip(1.0 - radial2, 0.0, 1.0))
        geom = np.dstack((sx, -sy, geom_z)).astype(np.float32)
        geom = self._normalize(geom)

        u = np.mod((sx * 0.5 + 0.5) * 1.22, 1.0)
        v = np.mod((sy * 0.5 + 0.5) * 1.22, 1.0)
        base = self._sample_bgr(albedo, u, v).astype(np.float32) / 255.0

        height_map = self._as_gray(self._height if self._height is not None else self._albedo)
        height_sample = self._sample_gray(height_map, u, v).astype(np.float32) / 255.0 if height_map is not None else 0.5

        tangent_normal = self._sample_tangent_normal(u, v, height_map)
        normal_strength = 0.78
        shaded_normal = self._normalize(
            geom * np.expand_dims(np.clip(tangent_normal[..., 2], 0.08, 1.0), 2)
            + np.dstack((tangent_normal[..., 0], tangent_normal[..., 1], np.zeros((h, w), dtype=np.float32))) * normal_strength
        )

        roughness = self._sample_optional_gray(self._roughness, u, v, default=0.48)
        ao = self._sample_optional_gray(self._ao, u, v, default=1.0)
        ao_factor = 0.48 + 0.52 * ao
        height_light = 0.9 + 0.14 * height_sample

        light = np.array([-0.38, 0.58, 0.72], dtype=np.float32)
        light = light / np.linalg.norm(light)
        fill = np.array([0.58, -0.36, 0.55], dtype=np.float32)
        fill = fill / np.linalg.norm(fill)
        view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        half_vec = light + view
        half_vec = half_vec / np.linalg.norm(half_vec)

        diffuse = np.clip(np.sum(shaded_normal * light, axis=2), 0.0, 1.0)
        fill_term = np.clip(np.sum(shaded_normal * fill, axis=2), 0.0, 1.0) * 0.24
        rim = np.power(np.clip(1.0 - geom[..., 2], 0.0, 1.0), 2.2) * 0.28
        spec_power = 10.0 + (1.0 - roughness) * 62.0
        specular = np.power(np.clip(np.sum(shaded_normal * half_vec, axis=2), 0.0, 1.0), spec_power)
        specular *= np.power(1.0 - roughness, 1.45) * 0.46

        shade = (0.25 + diffuse * 0.92 + fill_term + rim) * ao_factor * height_light
        color = np.clip(base * np.expand_dims(shade, 2) + np.expand_dims(specular, 2), 0.0, 1.0)
        color = np.power(color, 1.0 / 1.08)

        edge_alpha = np.clip((1.0 - np.sqrt(np.clip(radial2, 0.0, 1.0))) * 34.0, 0.0, 1.0)
        out[..., :3] = np.clip(color * 255.0, 0, 255).astype(np.uint8)
        out[..., 3] = np.where(mask, np.clip(edge_alpha * 255.0, 0, 255).astype(np.uint8), 0)

        self._render_cache = numpy_to_pixmap(out)
        self._render_key = key
        return self._render_cache

    def _sample_tangent_normal(self, u, v, height_map):
        normal = self._normal
        if normal is not None and not self._is_grayscale_rgb(normal):
            sampled = self._sample_bgr(self._as_bgr(normal), u, v).astype(np.float32) / 255.0
            n = np.dstack(
                (
                    sampled[..., 2] * 2.0 - 1.0,
                    sampled[..., 1] * 2.0 - 1.0,
                    sampled[..., 0] * 2.0 - 1.0,
                )
            )
            return self._normalize(n)

        source = self._as_gray(normal) if normal is not None else height_map
        if source is None:
            flat = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.float32)
            flat[..., 2] = 1.0
            return flat

        height = source.astype(np.float32) / 255.0
        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
        sx = self._sample_gray(gx, u, v)
        sy = self._sample_gray(gy, u, v)
        n = np.dstack((-sx * 1.7, sy * 1.7, np.ones_like(sx, dtype=np.float32)))
        return self._normalize(n)

    def _sample_bgr(self, image, u, v):
        ih, iw = image.shape[:2]
        x = np.mod((u * iw).astype(np.int32), iw)
        y = np.mod((v * ih).astype(np.int32), ih)
        return image[y, x]

    def _sample_gray(self, image, u, v):
        ih, iw = image.shape[:2]
        x = np.mod((u * iw).astype(np.int32), iw)
        y = np.mod((v * ih).astype(np.int32), ih)
        return image[y, x]

    def _sample_optional_gray(self, image, u, v, default):
        gray = self._as_gray(image)
        if gray is None:
            return np.full(u.shape, default, dtype=np.float32)
        return self._sample_gray(gray, u, v).astype(np.float32) / 255.0

    def _as_bgr(self, image):
        if image is None:
            return None
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def _as_gray(self, image):
        if image is None:
            return None
        if len(image.shape) == 2:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _is_grayscale_rgb(self, image):
        if image is None or len(image.shape) == 2:
            return True
        if image.shape[2] < 3:
            return True
        b, g, r = image[..., 0], image[..., 1], image[..., 2]
        return np.mean(np.abs(b.astype(np.int16) - g.astype(np.int16))) < 1.5 and np.mean(np.abs(g.astype(np.int16) - r.astype(np.int16))) < 1.5

    def _normalize(self, vectors):
        length = np.linalg.norm(vectors, axis=2, keepdims=True)
        return vectors / np.maximum(length, 1e-5)

    def _image_key(self, image):
        if image is None:
            return None
        return (id(image), image.shape, str(image.dtype))

class MainWindow(QMainWindow):
    def _on_preview_ready(self, result, generation):
        if generation == self._preview_generation and result is not None:
            self.image_viewer.set_after_image(result)
            if self.processed_normal_map is not None:
                self.image_viewer.set_map("Normal", self.processed_normal_map)

    def _on_preview_error(self, msg, generation):
        if generation == self._preview_generation:
            logger.warning("Preview skipped: %s", msg)

    def __init__(self):
        super().__init__()
        self.processor = SeamlessProcessor()
        self.current_file_path = None
        self.image_metadata = None
        self.loading_threads = []
        self.export_thread = None
        self.processing_thread = None
        self.preview_thread = PreviewThread()
        self.preview_thread.result_ready.connect(self._on_preview_ready)
        self.preview_thread.error.connect(self._on_preview_error)
        self.material_thread = MaterialMapThread(self)
        self.material_thread.maps_ready.connect(self._on_material_maps_ready)
        self.material_thread.error.connect(self._on_material_maps_error)
        self.image_np = None
        self.material_maps = {}
        self.processed_normal_map = None
        self._pending_reprocess = False
        self._ignore_next_processing_result = False
        self._load_generation = 0
        self._processing_generation = 0
        self._preview_generation = 0
        self._material_generation = 0
        self._active_mode = "seamless"
        # Undo / Redo stacks store numpy image snapshots.
        self._undo_stack = collections.deque(maxlen=20)
        self._redo_stack = collections.deque(maxlen=20)
        self.settings = load_settings()

        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._setup_shortcuts()
        self._connect_signals()
        self.control_panel.set_parameters(self.settings)
        self.setWindowTitle("SEAMS - Seamless Texture Studio")
        self.resize(self.settings.get("window_width", 1500), self.settings.get("window_height", 920))

    def _setup_ui(self):
        self.setStyleSheet(get_dark_theme())
        self.menuBar().setVisible(True)
        central = QWidget()
        central.setObjectName("AppRoot")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.rail = ToolRail()
        self.rail.navChanged.connect(self._on_nav_changed)
        self.rail.openClicked.connect(self._open_file)
        self.rail.exportClicked.connect(self._pbr_export_system)
        root.addWidget(self.rail)

        center = QWidget()
        center.setObjectName("CenterWorkspace")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        self.top_bar = None
        self.image_viewer = ImageViewer()
        self.image_viewer.fileDropped.connect(self._load_image)
        self.image_viewer.importRequested.connect(self._open_file)
        self.image_viewer.studioModeRequested.connect(self._on_studio_mode_requested)
        self.image_viewer.classicModeRequested.connect(self._on_classic_mode_requested)
        self.image_viewer.map_selector.mapChanged.connect(self._on_studio_map_changed)
        center_layout.addWidget(self.image_viewer, 1)
        root.addWidget(center, 1)

        right = QWidget()
        right.setObjectName("RightInspector")
        right.setFixedWidth(372)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self.control_stack = QStackedWidget()
        self.pre_panel = PreprocessingPanel()
        self.control_panel = ControlPanel()
        self.normal_panel = MaterialControlPanel()

        self.control_stack.addWidget(self.pre_panel)
        self.control_stack.addWidget(self.control_panel)
        self.control_stack.addWidget(self.normal_panel)
        right_layout.addWidget(self.control_stack)
        root.addWidget(right)

        self._set_active_nav("seamless")
        self.control_stack.setCurrentIndex(1)
        self._sync_inspector_footers("seamless")

    def _setup_menu(self):
        mb = self.menuBar()
        mb.clear()

        fm = mb.addMenu("&File")
        self._add_menu_action(fm, "&Open Texture...", "Ctrl+O", self._open_file)
        fm.addSeparator()
        self._add_menu_action(fm, "&Save Current Texture", "Ctrl+S", self._save_file)
        self._add_menu_action(fm, "Save &As...", "Ctrl+Shift+S", self._save_file_as)
        self._add_menu_action(fm, "Export Selected &Map...", "Ctrl+E", self._export_normal_map)
        self._add_menu_action(fm, "Export Renderer &Pipeline...", "Ctrl+Shift+E", self._pbr_export_system)
        fm.addSeparator()
        self._add_menu_action(fm, "E&xit", "Alt+F4", self.close)

        edit = mb.addMenu("&Edit")
        self._undo_action = self._add_menu_action(edit, "&Undo", "Ctrl+Z", self._undo)
        self._redo_action = self._add_menu_action(edit, "&Redo", "Ctrl+Y", self._redo)
        self._undo_action.setEnabled(False)
        self._redo_action.setEnabled(False)
        edit.addSeparator()
        self._add_menu_action(edit, "Reset &View", "Ctrl+0", self.image_viewer.fit_to_view)
        self._add_menu_action(edit, "Apply &Delight", "", self._apply_delight)

        view = mb.addMenu("&View")
        self._add_menu_action(view, "&Delight", "1", lambda: self._on_nav_changed("delight"))
        self._add_menu_action(view, "&Seamless", "2", lambda: self._on_nav_changed("seamless"))
        self._add_menu_action(view, "&Material Lab", "3", lambda: self._on_nav_changed("material"))
        view.addSeparator()
        self._add_menu_action(view, "Classic Mode", "", self._on_classic_mode_requested)
        self._add_menu_action(view, "Studio Mode", "", self._on_studio_mode_requested)

        help_menu = mb.addMenu("&Help")
        self._add_menu_action(help_menu, "&Keyboard Shortcuts", "F1", self._show_shortcuts)

        about_menu = mb.addMenu("&About")
        self._add_menu_action(about_menu, "About &SEAMS", "", self._show_about)

    def _add_menu_action(self, menu, label, shortcut, slot):
        action = QAction(label, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _not_implemented_action(self, name):
        return lambda: self.statusBar().showMessage(f"{name} is not available yet", 2200)

    def _show_shortcuts(self):
        from PyQt6.QtWidgets import QDialog, QScrollArea, QGridLayout, QPushButton, QFrame
        from PyQt6.QtCore import Qt

        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setFixedSize(580, 560)
        dialog.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.FramelessWindowHint
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # ── Root layout ───────────────────────────────────────────────────────
        root = QVBoxLayout(dialog)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Panel (opaque rounded card) ───────────────────────────────────────
        panel = QWidget()
        panel.setObjectName("ShortcutPanel")
        panel.setStyleSheet("""
            QWidget#ShortcutPanel {
                background: #0f0f14;
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 14px;
            }
        """)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(32, 28, 32, 28)
        panel_layout.setSpacing(0)
        root.addWidget(panel)

        # ── Header ────────────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("Keyboard Shortcuts")
        title_label.setStyleSheet("""
            color: #e8e4ff;
            font-family: 'Segoe UI';
            font-size: 15px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.06);
                color: #888;
                border: none;
                border-radius: 14px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255,80,80,0.25);
                color: #ff6060;
            }
        """)
        close_btn.clicked.connect(dialog.accept)

        header_row.addWidget(title_label)
        header_row.addStretch()
        header_row.addWidget(close_btn)
        panel_layout.addLayout(header_row)

        # Divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet("background: rgba(255,255,255,0.08); margin-top: 14px; margin-bottom: 18px;")
        panel_layout.addWidget(div)

        # ── Scroll area ───────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: rgba(255,255,255,0.03);
                width: 4px; border-radius: 2px;
            }
            QScrollBar::handle:vertical {
                background: rgba(140,80,255,0.5);
                border-radius: 2px; min-height: 24px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(20)

        # ── Shortcut data — grouped ───────────────────────────────────────────
        GROUPS = [
            ("File", [
                ("Ctrl+O",          "Open texture"),
                ("Ctrl+S",          "Save current texture"),
                ("Ctrl+Shift+S",    "Save As…"),
                ("Ctrl+E",          "Export selected map"),
                ("Ctrl+Shift+E",    "Export renderer pipeline"),
                ("Alt+F4",          "Exit"),
            ]),
            ("View & Navigation", [
                ("1",               "Switch to Delight"),
                ("2",               "Switch to Seamless"),
                ("3",               "Switch to Material Lab"),
                ("Ctrl+0",          "Reset viewport / Fit to view"),
            ]),
            ("Edit", [
                ("Ctrl+Z",          "Undo last image operation"),
                ("Ctrl+Y",          "Redo last undone operation"),
            ]),
            ("Help", [
                ("F1",              "Show keyboard shortcuts"),
            ]),
        ]

        def _make_key_badge(keys_str: str) -> QLabel:
            """Render  Ctrl+Shift+E  as styled pill badges joined by '+'."""
            parts = [k.strip() for k in keys_str.split("+")]
            html_parts = []
            for part in parts:
                html_parts.append(
                    f'<span style="'
                    f'background:#1e1b2e;'
                    f'color:#c4b8ff;'
                    f'border:1px solid rgba(140,80,255,0.45);'
                    f'border-radius:5px;'
                    f'padding:1px 7px;'
                    f'font-family:Consolas,monospace;'
                    f'font-size:11px;'
                    f'font-weight:600;'
                    f'">{part}</span>'
                )
            separator = '<span style="color:#555;font-size:10px;margin:0 2px;">+</span>'
            lbl = QLabel(separator.join(html_parts))
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setContentsMargins(0, 0, 0, 0)
            return lbl

        for group_name, shortcuts in GROUPS:
            # Group heading
            grp_label = QLabel(group_name.upper())
            grp_label.setStyleSheet("""
                color: rgba(140,80,255,0.85);
                font-family: 'Segoe UI';
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.8px;
                margin-bottom: 6px;
            """)
            content_layout.addWidget(grp_label)

            # Row container
            rows_widget = QWidget()
            rows_widget.setStyleSheet("""
                background: rgba(255,255,255,0.025);
                border-radius: 8px;
            """)
            rows_layout = QGridLayout(rows_widget)
            rows_layout.setContentsMargins(14, 8, 14, 8)
            rows_layout.setVerticalSpacing(6)
            rows_layout.setHorizontalSpacing(20)
            rows_layout.setColumnStretch(1, 1)

            for i, (keys, desc) in enumerate(shortcuts):
                badge = _make_key_badge(keys)
                desc_lbl = QLabel(desc)
                desc_lbl.setStyleSheet("""
                    color: #b0adc8;
                    font-family: 'Segoe UI';
                    font-size: 12px;
                """)
                rows_layout.addWidget(badge,    i, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                rows_layout.addWidget(desc_lbl, i, 1, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

            content_layout.addWidget(rows_widget)

        content_layout.addStretch()
        scroll.setWidget(content)
        panel_layout.addWidget(scroll)

        # ── Footer hint ───────────────────────────────────────────────────────
        footer = QLabel("Press  F1  or click  ✕  to close")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("""
            color: rgba(255,255,255,0.20);
            font-family: 'Segoe UI';
            font-size: 10px;
            margin-top: 10px;
        """)
        panel_layout.addWidget(footer)

        dialog.exec()

    def _show_about(self):
        from .credits_dialog import show_credits
        show_credits(self, APP_VERSION)


    # ── Undo / Redo ────────────────────────────────────────────────────────
    def _push_undo(self):
        """Snapshot current image_np onto the undo stack and clear redo."""
        if self.image_np is None:
            return
        self._undo_stack.append(self.image_np.copy())
        self._redo_stack.clear()
        self._update_undo_actions()

    def _update_undo_actions(self):
        if hasattr(self, "_undo_action"):
            self._undo_action.setEnabled(len(self._undo_stack) > 0)
        if hasattr(self, "_redo_action"):
            self._redo_action.setEnabled(len(self._redo_stack) > 0)

    def _undo(self):
        if not self._undo_stack or self.image_np is None:
            self.statusBar().showMessage("Nothing to undo", 2000)
            return
        # Save current state to redo stack
        self._redo_stack.append(self.image_np.copy())
        self.image_np = self._undo_stack.pop()
        self._restore_image_state("Undo")
        self._update_undo_actions()

    def _redo(self):
        if not self._redo_stack or self.image_np is None:
            self.statusBar().showMessage("Nothing to redo", 2000)
            return
        # Save current state to undo stack
        self._undo_stack.append(self.image_np.copy())
        self.image_np = self._redo_stack.pop()
        self._restore_image_state("Redo")
        self._update_undo_actions()

    def _restore_image_state(self, action_name: str):
        """Reload processor + viewer after an undo/redo step."""
        self.update_timer.stop()
        self.fullres_timer.stop()
        self._processing_generation += 1
        self._preview_generation += 1
        self._material_generation += 1
        self.processor.load_image(self.image_np)
        self.material_maps["Base Color"] = self.image_np.copy()
        self.image_viewer.set_before_image(self.image_np)
        self.image_viewer.set_after_image(self.image_np)
        self.image_viewer.set_map("Base Color", self.image_np)
        self.image_viewer.select_map("Base Color")
        self._on_normal_live_update()
        steps = len(self._undo_stack)
        self.statusBar().showMessage(
            f"{action_name}  —  {steps} step{'s' if steps != 1 else ''} remaining", 3000
        )
        # Kick off a full-res re-process in the background
        QTimer.singleShot(80, self._process_texture)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self._open_file)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_file)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self).activated.connect(self._save_file_as)
        QShortcut(QKeySequence("Ctrl+E"), self).activated.connect(self._export_normal_map)
        QShortcut(QKeySequence("Ctrl+Shift+E"), self).activated.connect(self._pbr_export_system)
        QShortcut(QKeySequence("Alt+F4"), self).activated.connect(self.close)
        
        QShortcut(QKeySequence("1"), self).activated.connect(lambda: self._on_nav_changed("delight"))
        QShortcut(QKeySequence("2"), self).activated.connect(lambda: self._on_nav_changed("seamless"))
        QShortcut(QKeySequence("3"), self).activated.connect(lambda: self._on_nav_changed("material"))
        QShortcut(QKeySequence("Ctrl+0"), self).activated.connect(self.image_viewer.fit_to_view)
        
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo)
        QShortcut(QKeySequence("F1"), self).activated.connect(self._show_shortcuts)


    def _setup_status_bar(self):
        native_sb = QStatusBar()
        native_sb.setSizeGripEnabled(False)
        native_sb.setMaximumHeight(0)
        native_sb.setVisible(False)
        self.setStatusBar(native_sb)
        self.bottom_bar = QWidget()
        self.bottom_bar.setObjectName("CustomStatusBar")
        self.bottom_bar.setFixedHeight(26)
        self.bottom_bar.setStyleSheet("""
            QWidget#CustomStatusBar {
                background: #05060a;
                border-top: 1px solid #141827;
            }
            QLabel { color: #85889f; font-size: 11px; }
        """)
        bb_layout = QHBoxLayout(self.bottom_bar)
        bb_layout.setContentsMargins(12, 0, 8, 0)
        bb_layout.setSpacing(10)
        self.file_label = QLabel("")
        bb_layout.addWidget(self.file_label)
        self.status_label = QLabel("")
        self.status_label.setMinimumWidth(220)
        bb_layout.addWidget(self.status_label)
        bb_layout.addStretch()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(180)
        self.progress.setMaximumHeight(5)
        self.progress.hide()
        bb_layout.addWidget(self.progress)
        self._monitor = StatusBarMonitor()
        bb_layout.addWidget(self._monitor)
        central = self.centralWidget()
        old_layout = central.layout()
        holder = QWidget()
        holder.setLayout(old_layout)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(holder, 1)
        vbox.addWidget(self.bottom_bar, 0)
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(24)
        self.update_timer.timeout.connect(self._request_live_preview)
        self.fullres_timer = QTimer()
        self.fullres_timer.setSingleShot(True)
        self.fullres_timer.setInterval(320)
        self.fullres_timer.timeout.connect(self._process_texture)

    def statusBar(self):
        class _FakeSB:
            def __init__(self_, lbl):
                self_._lbl = lbl
                self_._message_id = 0
            def showMessage(self_, msg, timeout=0):
                self_._message_id += 1
                message_id = self_._message_id
                self_._lbl.setText(msg)
                if timeout > 0:
                    QTimer.singleShot(timeout, lambda: self_._lbl.setText("") if message_id == self_._message_id else None)
            def clearMessage(self_):
                self_._message_id += 1
                self_._lbl.setText("")
        if not hasattr(self, "_fake_sb"):
            self._fake_sb = _FakeSB(self.status_label)
        return self._fake_sb

    def _connect_signals(self):
        self.control_panel.parametersChanged.connect(self._on_parameters_changed)
        self.control_panel.livePreviewRequested.connect(self._on_live_preview_requested)
        self.control_panel.exportClicked.connect(self._quick_export)
        self.pre_panel.parametersChanged.connect(self._on_parameters_changed)
        self.pre_panel.livePreviewRequested.connect(self._on_live_preview_requested)
        self.pre_panel.applyClicked.connect(self._apply_delight)
        self.normal_panel.livePreviewRequested.connect(self._on_normal_live_update)
        self.normal_panel.generateClicked.connect(self._generate_normal_map)
        self.normal_panel.exportClicked.connect(self._export_normal_map)

    def _set_active_nav(self, key):
        self.rail.set_active(key)
        self._active_mode = key

    def _on_nav_changed(self, mode):
        if mode == "recent":
            candidates = [self.settings.get("last_file", "")] + self.settings.get("recent_files", [])
            for path in candidates:
                if path and os.path.exists(path):
                    self._load_image(path)
                    return
            self.statusBar().showMessage("No recent texture is available", 2500)
            return
        self._set_active_nav(mode)
        self._sync_inspector_footers(mode)
        if mode == "delight":
            self.control_stack.setCurrentIndex(0)
            self.image_viewer.set_mode("side_by_side")
        elif mode == "material":
            self.control_stack.setCurrentIndex(2)
            if self.image_np is not None:
                self._on_normal_live_update()
        else:
            self.control_stack.setCurrentIndex(1)
            self.image_viewer.set_mode("seamless")

    def _sync_inspector_footers(self, mode):
        if hasattr(self.pre_panel, "footer"):
            self.pre_panel.footer.setVisible(mode == "delight")
        if hasattr(self.control_panel, "footer"):
            self.control_panel.footer.setVisible(mode == "seamless")
        if hasattr(self.normal_panel, "footer"):
            self.normal_panel.footer.setVisible(mode == "material")

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Texture", self.settings.get("last_directory", ""), get_format_filter())
        if path:
            self._load_image(path)

    def _load_image(self, path, **_kwargs):
        path = os.path.abspath(os.path.expanduser(str(path)))
        self._load_generation += 1
        generation = self._load_generation
        self.update_timer.stop()
        self.fullres_timer.stop()
        self.progress.setRange(0, 0)
        self.progress.show()
        self.statusBar().showMessage("Loading texture...")

        thread = ImageLoadThread(path, generation, self)
        self.loading_threads.append(thread)
        thread.loaded.connect(self._on_image_loaded)
        thread.error.connect(self._on_image_load_error)
        thread.finished.connect(lambda thread=thread: self._discard_load_thread(thread))
        thread.start()

    def _discard_load_thread(self, thread):
        if thread in self.loading_threads:
            self.loading_threads.remove(thread)
        thread.deleteLater()

    def _on_image_loaded(self, path, image, metadata, info, generation):
        if generation != self._load_generation:
            return
        try:
            self._preview_generation += 1
            self._material_generation += 1
            self._processing_generation += 1
            self.processor.load_image(image)
            self.current_file_path = path
            self.image_metadata = metadata
            # Push current state to undo before replacing
            if self.image_np is not None:
                self._push_undo()
            self._redo_stack.clear()
            self._update_undo_actions()
            self.image_np = image.copy()
            self.material_maps = {"Base Color": image.copy()}
            self.processed_normal_map = None
            self.image_viewer.set_before_image(image)
            self.image_viewer.set_after_image(None)
            self.image_viewer.fit_to_view()
            self.control_panel.set_image_loaded(True)
            self.normal_panel.set_image_loaded(True)
            self.control_panel.set_processed(False)
            info = info or get_file_info(path)
            name = os.path.basename(path)
            self.file_label.setText(f"{name}  |  {metadata.width} x {metadata.height}  |  {info['size_str']}")
            self.rail.update_info(metadata, info)
            self.settings["last_directory"] = os.path.dirname(path)
            self.settings["last_file"] = path
            self._remember_recent_file(path)
            save_settings(self.settings)
            self.setWindowTitle(f"SEAMS - {name}")
            self._on_nav_changed("seamless")
            self.progress.hide()
            QTimer.singleShot(120, self._process_texture)
        except Exception as exc:
            log_exception(logger, f"Failed to finalize loaded image {path}", exc)
            self.progress.hide()
            QMessageBox.critical(self, "Error", str(exc))

    def _remember_recent_file(self, path):
        recent = [p for p in self.settings.get("recent_files", []) if p and os.path.exists(p)]
        normalized = os.path.abspath(path)
        recent = [p for p in recent if os.path.abspath(p) != normalized]
        recent.insert(0, normalized)
        self.settings["recent_files"] = recent[:10]

    def _on_image_load_error(self, path, msg, generation):
        if generation != self._load_generation:
            return
        self.progress.hide()
        QMessageBox.critical(self, "Open Texture", f"Could not open this texture:\n{path}\n\n{msg}")

    def _on_parameters_changed(self):
        self.fullres_timer.stop()
        self.fullres_timer.start()

    def _on_live_preview_requested(self):
        if self._active_mode == "material":
            self._on_normal_live_update()
            return
        if self.processor.original_image is not None:
            self.fullres_timer.stop()
            if not self.update_timer.isActive():
                self.update_timer.start()

    def _request_live_preview(self):
        params = self.control_panel.get_parameters()
        params["preprocessing"] = self.pre_panel.get_parameters()
        if self.image_np is not None:
            self._preview_generation += 1
            self.preview_thread.request(self.image_np, params, self._preview_generation)

    def _apply_delight(self):
        if self.image_np is None:
            return
        self.update_timer.stop()
        self.fullres_timer.stop()
        if self.processing_thread and self.processing_thread.isRunning():
            self._ignore_next_processing_result = True
            self._pending_reprocess = False
            self._processing_generation += 1
        params = self.pre_panel.get_parameters()
        strength = float(params.get("delight", 0.0))
        flatness = float(params.get("flatness", 0.0))
        if strength > 0 or flatness > 0:
            from app.core.delighting import delight_image
            base_color = delight_image(self.image_np, strength=strength, flatness=flatness)
        elif self.processor.delighted_image is not None and self.processor.delighted_image.shape == self.image_np.shape:
            base_color = self.processor.delighted_image.copy()
        else:
            base_color = self.image_np.copy()

        self._push_undo()
        self.image_np = base_color.copy()
        self._processing_generation += 1
        self._preview_generation += 1
        self._material_generation += 1
        self.processor.load_image(self.image_np)
        
        self.pre_panel._on_reset()
        self.update_timer.stop()
        self.fullres_timer.stop()
        
        self.image_viewer.set_before_image(self.image_np)
        self.image_viewer.set_after_image(self.image_np)
        self.image_viewer.set_map("Base Color", self.image_np)
        self.material_maps["Base Color"] = self.image_np.copy()
        self.image_viewer.select_map("Base Color")
        self.image_viewer.set_delighted_image(None)
        self.control_panel.set_processed(True)
        self._on_normal_live_update()
        self.statusBar().showMessage("Delight applied to BaseColor", 3000)

    def _process_texture(self):
        if self.image_np is None:
            return
        if self.processing_thread and self.processing_thread.isRunning():
            self._pending_reprocess = True
            return
        self._pending_reprocess = False
        self.progress.setRange(0, 0)
        self.progress.show()
        params = self.control_panel.get_parameters()
        params["preprocessing"] = self.pre_panel.get_parameters()
        self.processor.set_parameters(**params)
        self._processing_generation += 1
        generation = self._processing_generation
        self.processing_thread = ProcessingThread(self.image_np, params, generation, self)
        self.processing_thread.finished.connect(self._on_processing_finished)
        self.processing_thread.error.connect(self._on_processing_error)
        self.processing_thread.start()

    def _on_processing_finished(self, result, elapsed, generation):
        self.progress.hide()
        if generation != self._processing_generation or self._ignore_next_processing_result:
            self._ignore_next_processing_result = False
            self.statusBar().showMessage("Skipped stale processing result", 1800)
            if self._pending_reprocess:
                self._process_texture()
            return
        self.processor.set_processed_image(result)
        self.image_viewer.set_after_image(result)
        self.image_viewer.set_map("Base Color", result)
        self.material_maps["Base Color"] = result.copy()
        self.control_panel.set_processed(True)
        
        # Update all material maps based on current Material Lab sliders
        self._on_normal_live_update()


        if self._active_mode == "delight":
            self.control_stack.setCurrentIndex(0)
        elif self._active_mode == "seamless":
            self.control_stack.setCurrentIndex(1)

        self.statusBar().showMessage(f"Done ({elapsed:.2f}s)", 3000)
        if self._pending_reprocess:
            self._process_texture()

    def _on_processing_error(self, msg, generation):
        if generation != self._processing_generation:
            return
        self.progress.hide()
        QMessageBox.critical(self, "Processing Error", msg)

    def _on_normal_live_update(self):
        if self.image_np is None:
            return
        params = self.normal_panel.get_parameters()
        base = self.processor.processed_image if self.processor.processed_image is not None else self.image_np
        self._material_generation += 1
        self.material_thread.request(base, params, self._material_generation)

    def _on_material_maps_ready(self, pbr_maps, generation):
        if generation != self._material_generation:
            return
        self.processed_normal_map = pbr_maps.get("Normal")
        for name, img in pbr_maps.items():
            if name != "Displacement":
                self.material_maps[name] = img.copy()
            self.image_viewer.set_map(name, img)

    def _on_material_maps_error(self, msg, generation):
        if generation == self._material_generation:
            self.statusBar().showMessage(f"Material map update failed: {msg}", 5000)

    def _generate_normal_map(self):
        self._on_normal_live_update()

    def _save_file(self):
        self._quick_export()

    def _save_file_as(self):
        if self._current_export_image() is None:
            return
        fmt = self.control_panel.get_export_format()
        default = get_output_path(self.current_file_path, "_seamless", fmt)
        path, _ = QFileDialog.getSaveFileName(self, "Save Texture", default, get_format_filter())
        if path:
            self._save_to_path(path)

    def _current_export_image(self):
        if self.processor.processed_image is not None:
            return self.processor.processed_image
        return self.image_np

    def _save_to_path(self, path):
        try:
            image = self._current_export_image()
            if image is None:
                return
            save_image(image, path, metadata=self.image_metadata)
            self.statusBar().showMessage(f"Saved: {path}", 5000)
        except Exception as exc:
            log_exception(logger, f"Save failed for {path}", exc)
            QMessageBox.critical(self, "Save Error", str(exc))

    def _quick_export(self):
        if self._current_export_image() is None:
            return
            
        mode = self.control_panel.get_save_mode()
        fmt = self.control_panel.get_export_format()
        
        if mode == "overwrite" and self.current_file_path:
            self._save_to_path(self.current_file_path)
        else:
            default = get_output_path(self.current_file_path, "_seamless", fmt)
            path, _ = QFileDialog.getSaveFileName(self, "Export Texture", default, get_format_filter())
            if path:
                self._save_to_path(path)

    def _pbr_export_system(self):
        if self.processor.processed_image is None:
            if self.image_np is None:
                return
            base_img = self.image_np
        else:
            base_img = self.processor.processed_image
            
        self.progress.setRange(0, 0)
        self.progress.show()

        params = self.normal_panel.get_parameters()
        try:
            # Generate lightweight thumbnails for the export dialog.
            h, w = base_img.shape[:2]
            thumb_size = 512
            scale = min(1.0, thumb_size / max(h, w))
            if scale < 1.0:
                thumb_img = cv2.resize(base_img, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
            else:
                thumb_img = base_img.copy()

            from app.gui.image_viewer import numpy_to_pixmap
            cached = self.image_viewer.maps
            maps_preview = {
                "BaseColor": numpy_to_pixmap(thumb_img),
                "Base Color": numpy_to_pixmap(thumb_img),
            }
            needed = ["Normal", "Roughness", "Metallic", "AO", "Height", "Opacity", "Emissive"]
            missing = [name for name in needed if cached.get(name) is None]
            if missing:
                preview_maps = NormalGenerator.process(thumb_img, **params)
                for name in needed:
                    maps_preview[name] = numpy_to_pixmap(preview_maps[name])
            else:
                for name in needed:
                    maps_preview[name] = cached[name]
            maps_preview["Ambient Occlusion"] = maps_preview["AO"]
        except Exception as exc:
            log_exception(logger, "Failed to prepare export dialog previews", exc)
            QMessageBox.critical(self, "Export Error", f"Could not prepare export previews:\n{exc}")
            return
        finally:
            self.progress.hide()
        
        from app.gui.export_dialog import PBRExportDialog
        default_dir = self.settings.get("last_directory", os.path.expanduser("~"))
        
        dialog = PBRExportDialog(base_img, maps_preview, default_dir, self)
        if dialog.exec():
            data = dialog.get_export_data()
            self._run_pbr_export(base_img, params, data)

    def _run_pbr_export(self, base_img, gen_params, data):
        if self.export_thread and self.export_thread.isRunning():
            QMessageBox.information(self, "Export In Progress", "A renderer export is already running.")
            return

        self.settings["last_directory"] = data["directory"]
        self.progress.setRange(0, 0)
        self.progress.show()

        self.export_thread = PBRExportThread(base_img, gen_params, data, self.image_metadata, self)
        self.export_thread.finished.connect(self._on_pbr_export_finished)
        self.export_thread.error.connect(self._on_pbr_export_error)
        self.export_thread.finished.connect(self.export_thread.deleteLater)
        self.export_thread.error.connect(self.export_thread.deleteLater)
        self.export_thread.start()

    def _on_pbr_export_finished(self, result):
        self.progress.hide()
        written = result.get("written", {})
        engine = result.get("engine", "PBR")
        self.statusBar().showMessage(f"{engine} export complete: {len(written)} files", 6000)
        save_settings(self.settings)
        self.export_thread = None

    def _on_pbr_export_error(self, msg):
        self.progress.hide()
        self.export_thread = None
        QMessageBox.critical(self, "Export Error", f"Failed to export PBR maps:\n{msg}")

    def _export_normal_map(self):
        channel = self.image_viewer.map_selector.checked_name()
        image = self._material_channel_image(channel)
        if image is None:
            QMessageBox.information(self, "Export Map", f"No {channel} map is available yet.")
            return

        suffix = self._channel_export_suffix(channel)
        default = get_output_path(self.current_file_path, suffix, self.normal_panel.get_export_format())
        path, _ = QFileDialog.getSaveFileName(self, f"Save {channel} Map", default, get_format_filter())
        if path:
            try:
                save_image(image, path, metadata=self.image_metadata)
                self.statusBar().showMessage(f"Saved {channel}: {path}", 5000)
            except Exception as exc:
                log_exception(logger, f"Map export failed for {path}", exc)
                QMessageBox.critical(self, "Export Map Error", str(exc))

    def _material_channel_image(self, channel):
        if channel == "Displacement":
            channel = "Height"
        if channel == "Base Color":
            return self._current_export_image()
        image = self.material_maps.get(channel)
        if image is not None:
            return image
        base = self._current_export_image()
        if base is None:
            return None
        generated = NormalGenerator.process(base, **self.normal_panel.get_parameters())
        for name, img in generated.items():
            if name != "Displacement":
                self.material_maps[name] = img.copy()
        self.processed_normal_map = generated.get("Normal")
        return self.material_maps.get(channel)

    def _channel_export_suffix(self, channel):
        suffixes = {
            "Base Color": "_basecolor",
            "Normal": "_normal",
            "Roughness": "_roughness",
            "Metallic": "_metallic",
            "AO": "_ao",
            "Height": "_height",
            "Displacement": "_height",
            "Opacity": "_opacity",
            "Emissive": "_emissive",
        }
        return suffixes.get(channel, f"_{channel.lower().replace(' ', '_')}")

    def closeEvent(self, event):
        if self.export_thread and self.export_thread.isRunning():
            QMessageBox.warning(
                self,
                "Export In Progress",
                "A renderer export is still running. Wait for it to finish before closing SEAMS.",
            )
            event.ignore()
            return

        self.settings["window_width"] = self.width()
        self.settings["window_height"] = self.height()
        self.settings["active_tool"] = self._active_mode
        self.settings.update(self.control_panel.get_parameters())
        save_settings(self.settings)
        if hasattr(self, "_monitor"):
            self._monitor.stop()
        if self.preview_thread.isRunning():
            self.preview_thread.stop()
            self.preview_thread.wait(2000)
        for thread in list(self.loading_threads):
            if thread.isRunning():
                thread.wait(2000)
                if thread.isRunning():
                    QMessageBox.warning(
                        self,
                        "Loading In Progress",
                        "A texture is still loading. Wait for it to finish before closing SEAMS.",
                    )
                    event.ignore()
                    return
        if self.material_thread.isRunning():
            self.material_thread.stop()
            self.material_thread.wait(3000)
            if self.material_thread.isRunning():
                QMessageBox.warning(
                    self,
                    "Material Update In Progress",
                    "Material maps are still updating. Wait for the update to finish before closing SEAMS.",
                )
                event.ignore()
                return
        if self.processing_thread and self.processing_thread.isRunning():
            self._ignore_next_processing_result = True
            self.processing_thread.wait(3000)
            if self.processing_thread.isRunning():
                QMessageBox.warning(
                    self,
                    "Processing In Progress",
                    "Texture processing is still running. Wait for it to finish before closing SEAMS.",
                )
                event.ignore()
                return
        if hasattr(self, "image_viewer"):
            self.image_viewer.cleanup()
        event.accept()

    def _on_studio_map_changed(self, name):
        self._set_active_nav("material")
        self._sync_inspector_footers("material")
        self.control_stack.setCurrentIndex(2)
        self.normal_panel.set_active_map(name)
        self.statusBar().showMessage(f"Material Lab: {name}", 1800)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            ext = os.path.splitext(event.mimeData().urls()[0].toLocalFile())[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".tga", ".exr"]:
                event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self._load_image(urls[0].toLocalFile())

    def _on_studio_mode_requested(self):
        self.image_viewer.set_workspace(1)
        self._on_nav_changed("material")
        
        if "Normal" not in self.image_viewer.maps or self.image_viewer.maps["Normal"] is None:
            self._on_normal_live_update()
            
        self.image_viewer.select_map("Normal")


    def _on_classic_mode_requested(self):
        self.image_viewer.set_workspace(0)
        self._on_nav_changed("seamless")
