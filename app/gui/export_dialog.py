import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.utils.image_io import ensure_writable_directory, sanitize_filename_component


RENDERERS = {
    "corona": {
        "name": "Corona",
        "title": "Corona Renderer Export",
        "badge": "CORONA",
        "quick": "Archviz optimized",
        "description": "Optimized for 3ds Max + Corona Physical Material workflow.",
        "workflow": "Metallic / Roughness",
        "format": "PNG",
        "normal": "OpenGL (Y+)",
        "bit_depth": "16-bit",
        "colorspace": "Renderer Preset",
        "token": "CORONA",
        "formats": ["PNG", "TIFF", "EXR"],
        "options": [
            "Convert Roughness to Glossiness",
            "Generate CoronaPhysicalMtl preset",
            "Gamma-safe export",
            "Use CoronaNormal compatibility",
            "Clamp displacement values",
            "Auto-generate 3ds Max material",
        ],
        "material_option": "Generate Corona material",
    },
    "vray": {
        "name": "V-Ray",
        "title": "V-Ray Export",
        "badge": "VRAY",
        "quick": "V-Ray 6+ material workflow",
        "description": "Optimized for V-Ray 6+ material workflow.",
        "workflow": "Metallic / Roughness",
        "format": "TIFF",
        "normal": "DirectX (Y-)",
        "bit_depth": "16-bit",
        "colorspace": "Renderer Preset",
        "token": "VRAY",
        "formats": ["PNG", "TIFF", "EXR"],
        "options": [
            "Convert Roughness to Glossiness",
            "VRayBitmap optimized",
            "Generate VRayMtl preset",
            "Linear workflow support",
            "Reflection glossiness mode",
            "VRayDisplacementMod support",
        ],
        "material_option": "Generate VRay material",
    },
    "ue5": {
        "name": "Unreal Engine 5",
        "title": "Unreal Engine 5 Export",
        "badge": "UE5",
        "quick": "Game-ready packed export",
        "description": "Game-ready optimized UE5 material export pipeline.",
        "workflow": "Metallic / Roughness",
        "format": "PNG",
        "normal": "DirectX (Y-)",
        "bit_depth": "8-bit",
        "colorspace": "Renderer Preset",
        "token": "UE5",
        "formats": ["PNG", "TGA", "EXR"],
        "options": [
            "ORM Texture Packing",
            "Nanite displacement ready",
            "Virtual texture ready",
            "Compression optimized",
            "Auto power-of-two validation",
            "Generate UE5 material instance",
            "sRGB auto assignment",
        ],
        "material_option": "Generate UE5 material",
    },
    "blender": {
        "name": "Blender",
        "title": "Blender Cycles Export",
        "badge": "BLEND",
        "quick": "Cycles/Eevee workflow",
        "description": "Optimized for Blender Principled BSDF workflow.",
        "workflow": "Metallic / Roughness",
        "format": "PNG",
        "normal": "OpenGL (Y+)",
        "bit_depth": "16-bit",
        "colorspace": "Renderer Preset",
        "token": "BLENDER",
        "formats": ["PNG", "EXR"],
        "options": [
            "Generate Principled BSDF setup",
            "Auto node linking",
            "Cycles optimized",
            "Eevee compatible",
            "Non-color data assignment",
            "Height to displacement node setup",
        ],
        "material_option": "Generate Blender shader",
    },
    "generic": {
        "name": "Generic PBR",
        "title": "Generic PBR Export",
        "badge": "PBR",
        "quick": "Universal clean naming",
        "description": "Universal renderer-independent PBR texture export.",
        "workflow": "Metallic / Roughness",
        "format": "PNG",
        "normal": "OpenGL (Y+)",
        "bit_depth": "16-bit",
        "colorspace": "Manual",
        "token": "PBR",
        "formats": ["PNG", "TIFF", "EXR"],
        "options": [
            "Metallic/Roughness workflow",
            "Specular/Glossiness workflow",
            "OpenGL normal support",
            "DirectX normal support",
            "Clean universal naming",
        ],
        "material_option": "Generate .mat",
    },
}


MAPS = [
    ("BaseColor", "sRGB", True),
    ("Roughness", "Linear", True),
    ("Metallic", "Linear", True),
    ("Normal", "Linear", True),
    ("AO", "Linear", True),
    ("Height", "Linear", True),
    ("Opacity", "Linear", False),
    ("Emissive", "sRGB", False),
]


class RendererCard(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, key, preset, parent=None):
        super().__init__(parent)
        self.key = key
        self.setObjectName("RendererCard")
        self.setProperty("active", False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(96)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(5)
        top = QHBoxLayout()
        badge = QLabel(preset["badge"])
        badge.setObjectName("RendererBadge")
        top.addWidget(badge)
        top.addStretch()
        workflow = QLabel(preset["workflow"])
        workflow.setObjectName("RendererWorkflow")
        top.addWidget(workflow)
        layout.addLayout(top)
        name = QLabel(preset["name"])
        name.setObjectName("RendererName")
        layout.addWidget(name)
        quick = QLabel(preset["quick"])
        quick.setObjectName("RendererQuick")
        layout.addWidget(quick)

    def set_active(self, active):
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.key)


class MapToggleCard(QFrame):
    toggled = pyqtSignal()

    def __init__(self, name, colorspace, checked=True, parent=None):
        super().__init__(parent)
        self.name = name
        self.colorspace = colorspace
        self.checked = checked
        self.packed = False
        self.bit_depth = "16-bit"
        self.setObjectName("MapToggleCard")
        self.setProperty("checked", checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(186, 104)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.thumb = QLabel()
        self.thumb.setObjectName("MapThumb")
        self.thumb.setFixedSize(62, 62)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.thumb)
        body = QVBoxLayout()
        body.setSpacing(3)
        self.title = QLabel(name)
        self.title.setObjectName("MapTitle")
        self.meta = QLabel("")
        self.meta.setObjectName("MapMeta")
        self.state = QLabel("")
        self.state.setObjectName("MapMeta")
        body.addWidget(self.title)
        body.addWidget(self.meta)
        body.addWidget(self.state)
        body.addStretch()
        layout.addLayout(body, 1)
        self._refresh()

    def set_thumbnail(self, pixmap):
        if pixmap is None or pixmap.isNull():
            self.thumb.setText("No map")
            return
        crop = QPixmap(self.thumb.size())
        crop.fill(QColor("#07090f"))
        scaled = pixmap.scaled(
            self.thumb.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter = QPainter(crop)
        painter.drawPixmap((crop.width() - scaled.width()) // 2, (crop.height() - scaled.height()) // 2, scaled)
        painter.end()
        self.thumb.setPixmap(crop)

    def set_export_state(self, bit_depth, packed=False):
        self.bit_depth = bit_depth
        self.packed = packed
        self._refresh()

    def _refresh(self):
        self.setProperty("checked", self.checked)
        self.meta.setText(f"{self.colorspace} / {self.bit_depth}")
        self.state.setText("Packed" if self.packed else "Unpacked")
        self.style().unpolish(self)
        self.style().polish(self)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.checked = not self.checked
            self._refresh()
            self.toggled.emit()


class PBRExportDialog(QDialog):
    def __init__(self, base_image, maps, default_dir, parent=None):
        super().__init__(parent)
        self.base_image = base_image
        self.maps = maps
        self.default_dir = default_dir
        self.renderer_key = "corona"
        self.renderer_cards = {}
        self.option_checks = {}
        self.map_cards = {}

        self.setWindowTitle("Production Export Pipeline")
        self.resize(1040, 820)
        self.setMinimumSize(940, 720)
        self.setStyleSheet(self._stylesheet())

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 26, 28, 24)
        layout.setSpacing(18)

        title = QLabel("PRODUCTION EXPORT PIPELINES")
        title.setObjectName("DialogTitle")
        subtitle = QLabel("Renderer-specific presets with workflow conversion, naming, color management and material setup sidecars.")
        subtitle.setObjectName("DialogSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        renderer_grid = QGridLayout()
        renderer_grid.setSpacing(10)
        for index, (key, preset) in enumerate(RENDERERS.items()):
            card = RendererCard(key, preset)
            card.clicked.connect(self._select_renderer)
            self.renderer_cards[key] = card
            renderer_grid.addWidget(card, index // 3, index % 3)
        layout.addLayout(renderer_grid)

        self.pipeline_title = QLabel("")
        self.pipeline_title.setObjectName("PipelineTitle")
        self.pipeline_desc = QLabel("")
        self.pipeline_desc.setObjectName("DialogSubtitle")
        self.pipeline_desc.setWordWrap(True)
        layout.addWidget(self.pipeline_title)
        layout.addWidget(self.pipeline_desc)

        settings = QFrame()
        settings.setObjectName("SettingsPanel")
        settings_layout = QGridLayout(settings)
        settings_layout.setContentsMargins(16, 16, 16, 16)
        settings_layout.setSpacing(12)
        self.workflow_combo = self._combo(["Metallic / Roughness", "Specular / Glossiness"])
        self.format_combo = self._combo(["PNG", "TIFF", "EXR"])
        self.normal_combo = self._combo(["OpenGL (Y+)", "DirectX (Y-)"])
        self.bit_depth_combo = self._combo(["8-bit", "16-bit", "32-bit"])
        self.resolution_combo = self._combo(["1K", "2K", "4K", "8K"])
        self.resolution_combo.setCurrentText("4K")
        self.compression_combo = self._combo(["Balanced", "Lossless", "Render Farm Safe"])
        self.padding_combo = self._combo(["0 px", "8 px", "16 px", "32 px"])
        self.colorspace_combo = self._combo(["Renderer Preset", "sRGB", "Linear", "ACES"])
        self.texture_scale_combo = self._combo(["1.0", "0.5", "2.0", "Real-world"])
        rows = [
            ("Workflow", self.workflow_combo),
            ("Texture Format", self.format_combo),
            ("Normal Format", self.normal_combo),
            ("Bit Depth", self.bit_depth_combo),
            ("Resolution", self.resolution_combo),
            ("Compression", self.compression_combo),
            ("Padding / Dilation", self.padding_combo),
            ("Color Management", self.colorspace_combo),
            ("Texture Scale", self.texture_scale_combo),
        ]
        for i, (label, widget) in enumerate(rows):
            settings_layout.addWidget(self._field_label(label), i // 3 * 2, i % 3)
            settings_layout.addWidget(widget, i // 3 * 2 + 1, i % 3)
        layout.addWidget(settings)

        layout.addWidget(self._section("RENDERER MATERIAL LOGIC"))
        self.options_panel = QFrame()
        self.options_panel.setObjectName("SettingsPanel")
        self.options_layout = QGridLayout(self.options_panel)
        self.options_layout.setContentsMargins(16, 14, 16, 14)
        self.options_layout.setSpacing(8)
        layout.addWidget(self.options_panel)

        layout.addWidget(self._section("MAP EXPORT"))
        map_grid = QGridLayout()
        map_grid.setSpacing(10)
        for i, (name, colorspace, checked) in enumerate(MAPS):
            card = MapToggleCard(name, colorspace, checked)
            card.set_thumbnail(self._map_pixmap(name))
            self.map_cards[name] = card
            map_grid.addWidget(card, i // 4, i % 4)
        layout.addLayout(map_grid)

        layout.addWidget(self._section("AUTO MATERIAL GENERATION"))
        material_panel = QFrame()
        material_panel.setObjectName("SettingsPanel")
        mat_layout = QGridLayout(material_panel)
        mat_layout.setContentsMargins(16, 14, 16, 14)
        self.material_checks = {}
        for i, label in enumerate([
            "Generate .mat",
            "Generate Blender shader",
            "Generate UE5 material",
            "Generate Corona material",
            "Generate VRay material",
        ]):
            chk = QCheckBox(label)
            self.material_checks[label] = chk
            mat_layout.addWidget(chk, i // 3, i % 3)
        layout.addWidget(material_panel)

        layout.addWidget(self._section("OUTPUT"))
        output = QFrame()
        output.setObjectName("SettingsPanel")
        out_layout = QGridLayout(output)
        out_layout.setContentsMargins(16, 14, 16, 14)
        out_layout.setSpacing(10)
        self.folder_input = QLineEdit(default_dir)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_folder)
        self.name_input = QLineEdit(self._default_name())
        self.subfolder_chk = QCheckBox("Create renderer subfolder")
        self.subfolder_chk.setChecked(True)
        out_layout.addWidget(self._field_label("Export Folder"), 0, 0)
        out_layout.addWidget(self.folder_input, 0, 1)
        out_layout.addWidget(browse, 0, 2)
        out_layout.addWidget(self._field_label("Material Name"), 1, 0)
        out_layout.addWidget(self.name_input, 1, 1, 1, 2)
        out_layout.addWidget(self.subfolder_chk, 2, 1, 1, 2)
        layout.addWidget(output)

        self.export_btn = QPushButton("EXPORT RENDER PIPELINE")
        self.export_btn.setObjectName("ExportBtn")
        self.export_btn.setFixedHeight(52)
        self.export_btn.clicked.connect(self.accept)
        layout.addWidget(self.export_btn)

        scroll.setWidget(content)
        root.addWidget(scroll)
        self._select_renderer("corona")

    def _stylesheet(self):
        return """
        QDialog, QWidget { background: #080a10; color: #f2efff; font-family: "Segoe UI", Arial; font-size: 12px; }
        QLabel#DialogTitle { color: #ffffff; font-size: 22px; font-weight: 950; letter-spacing: 1px; }
        QLabel#DialogSubtitle { color: #8c91a8; line-height: 1.35; }
        QLabel#PipelineTitle { color: #ffffff; font-size: 17px; font-weight: 950; }
        QLabel#SectionTitle { color: #31e6bd; font-size: 10px; font-weight: 950; letter-spacing: 1.4px; }
        QLabel#FieldLabel { color: #8c91a8; font-size: 10px; font-weight: 850; letter-spacing: 1px; }
        QFrame#RendererCard, QFrame#MapToggleCard, QFrame#SettingsPanel {
            background: #10131d; border: 1px solid #232a3d; border-radius: 8px;
        }
        QFrame#RendererCard:hover, QFrame#MapToggleCard:hover { border-color: #6d59e8; background: #141827; }
        QFrame#RendererCard[active="true"], QFrame#MapToggleCard[checked="true"] {
            border: 1px solid #31e6bd; background: #111a24;
        }
        QLabel#RendererBadge { color: #31e6bd; font-size: 10px; font-weight: 950; letter-spacing: 1.2px; }
        QLabel#RendererWorkflow, QLabel#MapMeta { color: #858aa0; font-size: 10px; font-weight: 700; }
        QLabel#RendererName, QLabel#MapTitle { color: #ffffff; font-size: 13px; font-weight: 950; }
        QLabel#RendererQuick { color: #aeb4ca; font-size: 11px; }
        QLabel#MapThumb { background: #06070b; border: 1px solid #252c40; border-radius: 5px; color: #60677c; }
        QComboBox, QLineEdit {
            background: #0b0e16; border: 1px solid #2a3248; border-radius: 6px;
            color: #ffffff; padding: 7px 10px; min-height: 24px;
        }
        QComboBox:hover, QLineEdit:hover { border-color: #8f70ff; }
        QCheckBox { color: #d6d7e6; font-weight: 700; spacing: 8px; }
        QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #3a435b; background: #0b0e16; border-radius: 4px; }
        QCheckBox::indicator:checked { background: #31e6bd; border-color: #9fffee; }
        QPushButton { background: #121725; border: 1px solid #303950; color: #f2efff; border-radius: 7px; padding: 8px 12px; font-weight: 850; }
        QPushButton:hover { border-color: #31e6bd; }
        QPushButton#ExportBtn {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #5b3dff, stop:1 #31e6bd);
            border: 1px solid rgba(255,255,255,45); color: white; font-weight: 950; letter-spacing: 1.3px;
        }
        QScrollArea { border: none; }
        QScrollBar:vertical { width: 8px; background: #080a10; }
        QScrollBar::handle:vertical { background: #252c40; border-radius: 4px; min-height: 30px; }
        """

    def _combo(self, items):
        combo = QComboBox()
        combo.addItems(items)
        return combo

    def _section(self, text):
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    def _field_label(self, text):
        label = QLabel(text.upper())
        label.setObjectName("FieldLabel")
        return label

    def _default_name(self):
        return "Material_01"

    def _map_pixmap(self, name):
        aliases = {
            "BaseColor": ["BaseColor", "Base Color", "Albedo / Base Color"],
            "AO": ["AO", "Ambient Occlusion"],
        }
        for key in aliases.get(name, [name]):
            pix = self.maps.get(key)
            if pix is not None:
                return pix
        return None

    def _select_renderer(self, key):
        self.renderer_key = key
        preset = RENDERERS[key]
        for card_key, card in self.renderer_cards.items():
            card.set_active(card_key == key)
        self.pipeline_title.setText(preset["title"])
        self.pipeline_desc.setText(preset["description"])
        self.workflow_combo.setCurrentText(preset["workflow"])
        self.format_combo.clear()
        self.format_combo.addItems(preset["formats"])
        self.format_combo.setCurrentText(preset["format"])
        self.normal_combo.setCurrentText(preset["normal"])
        self.bit_depth_combo.setCurrentText(preset["bit_depth"])
        self.colorspace_combo.setCurrentText(preset["colorspace"])
        self._rebuild_options(preset)
        self._sync_material_checks(preset)
        for name, card in self.map_cards.items():
            card.set_export_state(preset["bit_depth"], packed=(key == "ue5" and name in ("AO", "Roughness", "Metallic")))

    def _rebuild_options(self, preset):
        while self.options_layout.count():
            item = self.options_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.option_checks = {}
        for i, label in enumerate(preset["options"]):
            chk = QCheckBox(label)
            chk.setChecked(True)
            self.option_checks[label] = chk
            self.options_layout.addWidget(chk, i // 3, i % 3)

    def _sync_material_checks(self, preset):
        for label, chk in self.material_checks.items():
            chk.setChecked(label == preset["material_option"])

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Directory", self.folder_input.text())
        if folder:
            self.folder_input.setText(folder)

    def accept(self):
        folder = self.folder_input.text().strip()
        if not folder:
            QMessageBox.warning(self, "Export Folder Required", "Choose a folder for the renderer export.")
            return
        try:
            ensure_writable_directory(folder)
        except Exception as exc:
            QMessageBox.warning(self, "Export Folder Unavailable", str(exc))
            return

        if not any(card.checked for card in self.map_cards.values()):
            QMessageBox.warning(self, "No Maps Selected", "Select at least one texture map to export.")
            return

        safe_name = sanitize_filename_component(self.name_input.text(), "Material_01")
        if safe_name != self.name_input.text().strip():
            self.name_input.setText(safe_name)

        super().accept()

    def get_export_data(self):
        preset = RENDERERS[self.renderer_key]
        maps = {name: card.checked for name, card in self.map_cards.items()}
        options = {label: chk.isChecked() for label, chk in self.option_checks.items()}
        material_options = {label: chk.isChecked() for label, chk in self.material_checks.items()}
        return {
            "renderer_key": self.renderer_key,
            "engine": preset["name"],
            "renderer_token": preset["token"],
            "directory": self.folder_input.text(),
            "name": self.name_input.text().strip() or "Material_01",
            "resolution": {"1K": 1024, "2K": 2048, "4K": 4096, "8K": 8192}[self.resolution_combo.currentText()],
            "format": self.format_combo.currentText().lower(),
            "bit_depth": self.bit_depth_combo.currentText(),
            "workflow": self.workflow_combo.currentText(),
            "normal_format": self.normal_combo.currentText(),
            "colorspace": self.colorspace_combo.currentText(),
            "compression": self.compression_combo.currentText(),
            "padding": self.padding_combo.currentText(),
            "texture_scale": self.texture_scale_combo.currentText(),
            "maps": maps,
            "options": options,
            "material_options": material_options,
            "packing": self.renderer_key == "ue5" and options.get("ORM Texture Packing", True),
            "create_subfolder": self.subfolder_chk.isChecked(),
            "subfolder_name": f"{self.name_input.text().strip() or 'Material_01'}_{preset['token']}",
        }
