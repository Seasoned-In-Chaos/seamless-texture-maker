"""Right-side modular control panels for SEAMS."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPainter, QColor, QPen
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def _title_font(size=22):
    font = QFont("Segoe UI Variable", size, QFont.Weight.Black)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
    return font


class MiniGraph(QWidget):
    """Small visual readout used as plugin-card texture analysis chrome."""

    def __init__(self, accent="#8f70ff", parent=None):
        super().__init__(parent)
        self.accent = QColor(accent)
        self.setFixedHeight(78)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        p.setPen(QPen(QColor(123, 105, 255, 52), 1))
        p.setBrush(QColor(7, 9, 16, 180))
        p.drawRoundedRect(r, 8, 8)
        p.setPen(QPen(QColor(255, 255, 255, 18), 1))
        for i in range(1, 5):
            x = r.left() + r.width() * i // 5
            p.drawLine(x, r.top() + 8, x, r.bottom() - 8)
        p.setPen(QPen(self.accent, 2))
        points = []
        for i in range(0, 36):
            x = r.left() + 12 + (r.width() - 24) * i / 35
            wave = 0.5 + 0.32 * __import__("math").sin(i * 0.58)
            y = r.top() + 14 + (r.height() - 28) * wave
            points.append((x, y))
        for a, b in zip(points, points[1:]):
            p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        p.end()


class PluginCard(QFrame):
    def __init__(self, title, subtitle=None, parent=None):
        super().__init__(parent)
        self.setObjectName("PluginCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)
        head = QHBoxLayout()
        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        t = QLabel(title.upper())
        t.setObjectName("CardTitle")
        s = QLabel(subtitle or "")
        s.setObjectName("CardSubtitle")
        s.setWordWrap(True)
        text_col.addWidget(t)
        if subtitle:
            text_col.addWidget(s)
        head.addLayout(text_col)
        head.addStretch()
        dot = QLabel("")
        dot.setObjectName("CardDot")
        dot.setFixedSize(8, 8)
        head.addWidget(dot)
        layout.addLayout(head)
        self.body = QVBoxLayout()
        self.body.setSpacing(12)
        layout.addLayout(self.body)


class LabeledSlider(QWidget):
    """Tactile slider row with compact numeric chip."""

    valueChanged = pyqtSignal(float)
    sliderPressed = pyqtSignal()
    sliderReleased = pyqtSignal()
    sliderMoved = pyqtSignal(float)

    def __init__(self, label, min_val=0, max_val=100, default=50, suffix="", parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.suffix = suffix

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        top = QHBoxLayout()
        self.label = QLabel(label.upper())
        self.label.setObjectName("ParamLabel")
        self.value_label = QLabel(self._format(default))
        self.value_label.setObjectName("ValueChip")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setFixedWidth(42)
        top.addWidget(self.label)
        top.addStretch()
        top.addWidget(self.value_label)
        layout.addLayout(top)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(self._to_raw(default))
        self.slider.valueChanged.connect(self._on_value_changed)
        self.slider.sliderPressed.connect(self.sliderPressed.emit)
        self.slider.sliderReleased.connect(self.sliderReleased.emit)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        layout.addWidget(self.slider)

    def _to_raw(self, val):
        if self.max_val == self.min_val:
            return 0
        return int((val - self.min_val) / (self.max_val - self.min_val) * 100)

    def _from_raw(self, value):
        return self.min_val + (value / 100.0) * (self.max_val - self.min_val)

    def _format(self, val):
        if isinstance(self.min_val, int) and isinstance(self.max_val, int):
            return f"{int(round(val))}{self.suffix}"
        return f"{val:.1f}{self.suffix}"

    def _on_value_changed(self, value):
        mapped = self._from_raw(value)
        self.value_label.setText(self._format(mapped))
        self.valueChanged.emit(mapped)

    def _on_slider_moved(self, value):
        self.sliderMoved.emit(self._from_raw(value))

    def value(self):
        return self._from_raw(self.slider.value())

    def setValue(self, val):
        self.slider.setValue(self._to_raw(val))


class ChipRow(QWidget):
    changed = pyqtSignal(str)

    def __init__(self, labels, default=None, parent=None):
        super().__init__(parent)
        self._buttons = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for label in labels:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setObjectName("Chip")
            b.clicked.connect(lambda _=False, text=label: self.set_value(text))
            layout.addWidget(b)
            self._buttons[label] = b
        layout.addStretch()
        self.set_value(default or labels[0])

    def set_value(self, label):
        for text, button in self._buttons.items():
            button.setChecked(text == label)
        self.changed.emit(label)


class PanelShell(QWidget):
    def __init__(self, title, kicker, parent=None):
        super().__init__(parent)
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)
        scroll = QScrollArea()
        self.scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName("PanelScroll")
        body = QWidget()
        body.setObjectName("PanelBody")
        self.layout = QVBoxLayout(body)
        self.layout.setContentsMargins(18, 18, 18, 18)
        self.layout.setSpacing(14)

        title_label = QLabel(title.upper())
        title_label.setFont(_title_font())
        title_label.setObjectName("PanelTitle")
        kicker_label = QLabel(kicker)
        kicker_label.setObjectName("PanelKicker")
        kicker_label.setWordWrap(True)
        self.layout.addWidget(title_label)
        self.layout.addWidget(kicker_label)

        scroll.setWidget(body)
        self._outer.addWidget(scroll, 1)

    def add_bottom_button(self, text, signal):
        footer = QWidget()
        self.footer = footer
        footer.setObjectName("PanelFooter")
        fl = QVBoxLayout(footer)
        fl.setContentsMargins(18, 14, 18, 16)
        button = QPushButton(text)
        button.setObjectName("PrimaryAction")
        button.setFixedHeight(48)
        button.clicked.connect(signal)
        fl.addWidget(button)
        self._outer.addWidget(footer)
        return button


class PreprocessingPanel(PanelShell):
    parametersChanged = pyqtSignal()
    livePreviewRequested = pyqtSignal()
    applyClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(
            "Delight",
            "Remove baked lighting, gradients, ambient occlusion and color cast before seamless processing.",
            parent,
        )

        insight = PluginCard("Light Field Analysis", "Balance shadows while preserving usable surface grain.")
        insight.body.addWidget(MiniGraph("#31e6bd"))
        chips = ChipRow(["Soft", "Balanced", "Aggressive"], "Balanced")
        insight.body.addWidget(chips)
        self.layout.addWidget(insight)

        adjust = PluginCard("Adjustments", "Live parameters feed the viewport preview.")
        self._sliders = {}
        for label, key, lo, hi, default in [
            ("Shadow Removal", "shadow", 0, 100, 0),
            ("Highlight Reduction", "highlight", 0, 100, 0),
            ("Contrast Recovery", "contrast", 0, 100, 0),
            ("Detail Preservation", "detail", 0, 100, 0),
            ("Color Preservation", "color", 0, 100, 0),
            ("Flatness Strength", "flatness", 0, 100, 0),
            ("AO Removal", "ao", 0, 100, 0),
            ("Edge Consistency", "edge", 0, 100, 0),
        ]:
            slider = LabeledSlider(label, lo, hi, default)
            slider.valueChanged.connect(self.parametersChanged.emit)
            slider.sliderMoved.connect(self.livePreviewRequested.emit)
            self._sliders[key] = slider
            adjust.body.addWidget(slider)
        self.layout.addWidget(adjust)

        output = PluginCard("Output Intent", "Texture-space flattening settings.")
        output.body.addWidget(self._combo_row("Output Type", ["Albedo / Diffuse", "Base Color", "Diffuse", "Raw"], "output_type_combo"))
        output.body.addWidget(self._combo_row("Bit Depth", ["8-bit", "16-bit", "32-bit"], "bit_depth_combo", current="16-bit"))
        self.color_temp_toggle = QCheckBox("Keep color temperature")
        self.color_temp_toggle.setChecked(True)
        output.body.addWidget(self.color_temp_toggle)
        self.layout.addWidget(output)
        self.layout.addStretch()

        self.apply_btn = self.add_bottom_button("APPLY DELIGHT", self.applyClicked.emit)

    def _combo_row(self, label, items, attr, current=None):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label))
        layout.addStretch()
        combo = QComboBox()
        combo.addItems(items)
        if current:
            combo.setCurrentText(current)
        setattr(self, attr, combo)
        layout.addWidget(combo)
        return row

    def _on_reset(self):
        for slider in self._sliders.values():
            slider.setValue(0)
        self.parametersChanged.emit()

    def get_parameters(self):
        shadow = self._sliders["shadow"].value() / 100.0
        flatness = self._sliders["flatness"].value() / 100.0
        delight = max(shadow, self._sliders["ao"].value() / 100.0, self._sliders["highlight"].value() / 100.0)
        return {
            "delight": delight,
            "flatness": flatness,
            "shadow_removal": shadow,
            "highlight_reduction": self._sliders["highlight"].value() / 100.0,
            "contrast_recovery": self._sliders["contrast"].value() / 100.0,
            "detail_preservation": self._sliders["detail"].value() / 100.0,
            "color_preservation": self._sliders["color"].value() / 100.0,
            "ao_removal": self._sliders["ao"].value() / 100.0,
            "edge_consistency": self._sliders["edge"].value() / 100.0,
        }


class ControlPanel(PanelShell):
    parametersChanged = pyqtSignal()
    livePreviewRequested = pyqtSignal()
    processClicked = pyqtSignal()
    exportClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(
            "Seamless Maker",
            "Edge-aware synthesis controls for production-ready tiling textures.",
            parent,
        )

        analysis = PluginCard("Seam Intelligence", "Edge balance, overlap energy and repeat risk.")
        analysis.body.addWidget(MiniGraph("#8f70ff"))
        rec = QLabel("Smart recommendation: 20% overlap with low falloff for structured materials.")
        rec.setObjectName("Recommendation")
        rec.setWordWrap(True)
        analysis.body.addWidget(rec)
        self.layout.addWidget(analysis)

        method = PluginCard("Technique", "Choose the synthesis behavior for this material.")
        self.method_combo = QComboBox()
        self.method_combo.addItem("Overlap Blend", "overlap")
        self.method_combo.addItem("Splat Synthesis", "splat")
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method.body.addWidget(self.method_combo)
        method.body.addWidget(ChipRow(["Stone", "Fabric", "Organic", "Hard Surface"], "Stone"))
        self.layout.addWidget(method)

        self.overlap_card = PluginCard("Overlap Controls", "Precise seam replacement and edge feathering.")
        self.overlap_x_slider = LabeledSlider("Overlap X", 0, 50, 0, "%")
        self.overlap_y_slider = LabeledSlider("Overlap Y", 0, 50, 0, "%")
        self.ov_falloff_slider = LabeledSlider("Edge Falloff", 0, 100, 0, "%")
        for slider in [self.overlap_x_slider, self.overlap_y_slider, self.ov_falloff_slider]:
            slider.valueChanged.connect(self._on_param_changed)
            slider.sliderMoved.connect(self._on_live_update)
            self.overlap_card.body.addWidget(slider)
        self.layout.addWidget(self.overlap_card)

        self.splat_card = PluginCard("Splat Controls", "Patch-based material synthesis for irregular sources.")
        self.sp_falloff_slider = LabeledSlider("Edge Falloff", 0, 100, 0, "%")
        self.splat_scale = LabeledSlider("Splat Scale", 0, 5, 0)
        self.splat_rot = LabeledSlider("Splat Rotation", 0, 360, 0)
        self.splat_rand_rot = LabeledSlider("Random Rotation", 0, 100, 0, "%")
        self.splat_wobble = LabeledSlider("Splat Wobble", 0, 100, 0, "%")
        for slider in [self.sp_falloff_slider, self.splat_scale, self.splat_rot, self.splat_rand_rot, self.splat_wobble]:
            slider.valueChanged.connect(self._on_param_changed)
            slider.sliderMoved.connect(self._on_live_update)
            self.splat_card.body.addWidget(slider)
        self.layout.addWidget(self.splat_card)
        self.splat_card.hide()
        self.current_random_seed = 0

        export = PluginCard("Export", "Production texture output.")
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("Format"))
        rl.addStretch()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "TIFF"])
        rl.addWidget(self.format_combo)
        export.body.addWidget(row)
        self.save_mode_group = QButtonGroup(self)
        self.new_file_radio = QRadioButton("Save as new file (_seamless)")
        self.new_file_radio.setChecked(True)
        self.overwrite_radio = QRadioButton("Overwrite original")
        self.save_mode_group.addButton(self.new_file_radio, 0)
        self.save_mode_group.addButton(self.overwrite_radio, 1)
        export.body.addWidget(self.new_file_radio)
        export.body.addWidget(self.overwrite_radio)
        self.info_label = QLabel("")
        self.info_label.setObjectName("InfoLabel")
        export.body.addWidget(self.info_label)
        self.layout.addWidget(export)
        self.layout.addStretch()

        self.export_btn = self.add_bottom_button("EXPORT TEXTURE", self.exportClicked.emit)
        self.export_btn.setEnabled(False)

    def _on_method_changed(self, _index):
        is_splat = self.method_combo.currentData() == "splat"
        self.overlap_card.setVisible(not is_splat)
        self.splat_card.setVisible(is_splat)
        self.parametersChanged.emit()

    def _on_param_changed(self, *_args):
        self.parametersChanged.emit()

    def _on_live_update(self, *_args):
        self.livePreviewRequested.emit()

    def get_parameters(self):
        method = self.method_combo.currentData()
        edge_falloff = (self.sp_falloff_slider.value() if method == "splat" else self.ov_falloff_slider.value()) / 100.0
        return {
            "method": method,
            "overlap_x": self.overlap_x_slider.value() / 100.0,
            "overlap_y": self.overlap_y_slider.value() / 100.0,
            "edge_falloff": edge_falloff,
            "splat_scale": self.splat_scale.value(),
            "splat_rotation": self.splat_rot.value(),
            "splat_random_rotation": self.splat_rand_rot.value() / 100.0,
            "splat_wobble": self.splat_wobble.value() / 100.0,
            "splat_randomize": self.current_random_seed,
        }

    def set_parameters(self, params):
        if "method" in params:
            idx = self.method_combo.findData(params["method"])
            if idx >= 0:
                self.method_combo.setCurrentIndex(idx)

    def get_export_format(self):
        return self.format_combo.currentText().lower()

    def get_save_mode(self):
        return "new_file" if self.new_file_radio.isChecked() else "overwrite"

    def set_image_loaded(self, _loaded):
        pass

    def set_processed(self, processed):
        self.export_btn.setEnabled(processed)

    def set_info(self, text):
        self.info_label.setText(text)
