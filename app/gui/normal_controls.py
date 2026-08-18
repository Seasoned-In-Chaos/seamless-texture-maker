from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QWidget,
)

from app.gui.controls import LabeledSlider, MiniGraph, PanelShell, PluginCard


class MaterialControlPanel(PanelShell):
    """Premium plugin panel for full PBR material map generation."""

    parametersChanged = pyqtSignal()
    livePreviewRequested = pyqtSignal()
    generateClicked = pyqtSignal()
    exportClicked = pyqtSignal()


    def __init__(self, parent=None):
        super().__init__(
            "Material Lab",
            "Fine-tune the individual PBR channels for production-ready outputs.",
            parent,
        )

        # 1. NORMAL
        self.normal_card = PluginCard("Normal Map Settings", "Advanced normal generation matching NVTT.")
        
        self.height_source = QComboBox()
        self.height_source.addItems(["Average RGB", "Luminance", "Max RGB", "Red", "Green", "Blue", "Alpha Channel"])
        self.height_source.currentTextChanged.connect(self._on_live_update)

        self.normal_filter = QComboBox()
        self.normal_filter.addItem("4 Sample", "4_sample")
        self.normal_filter.addItem("3×3", "sobel_3x3")
        self.normal_filter.addItem("5×5", "sobel_5x5")
        self.normal_filter.addItem("7×7", "sobel_7x7")
        self.normal_filter.addItem("9×9", "sobel_9x9")
        self.normal_filter.addItem("dUdV", "dudv")
        self.normal_filter.currentIndexChanged.connect(self._on_live_update)

        self.normal_wrap = QCheckBox("Wrap")
        self.normal_wrap.setChecked(True)
        self.normal_wrap.toggled.connect(self._on_live_update)
        
        self.normal_invert_x = QCheckBox("Invert X")
        self.normal_invert_x.toggled.connect(self._on_live_update)

        self.normal_invert_y = QCheckBox("Invert Y")
        self.normal_invert_y.toggled.connect(self._on_live_update)

        self.normal_invert_height = QCheckBox("Invert Height")
        self.normal_invert_height.toggled.connect(self._on_live_update)

        self.normal_format = QComboBox()
        self.normal_format.addItems(["OpenGL", "DirectX"])
        self.normal_format.currentTextChanged.connect(self._on_live_update)
        
        self.normal_min_z = LabeledSlider("Min Z", 0.0, 1.0, 0.0, "", steps=1000, decimals=3)
        self.normal_min_z.valueChanged.connect(self._on_live_update)

        self.normal_scale = LabeledSlider("Scale", 0.1, 8.0, 4.266, "", steps=1000, decimals=3)
        self.normal_scale.valueChanged.connect(self._on_live_update)

        self.normal_card.body.addWidget(QLabel("Height Source"))
        self.normal_card.body.addWidget(self.height_source)

        filter_layout = QGridLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setHorizontalSpacing(10)
        filter_layout.addWidget(QLabel("Filter Type"), 0, 0)
        filter_layout.addWidget(self.normal_filter, 0, 1)
        filter_layout.addWidget(self.normal_wrap, 1, 1)
        self.normal_card.body.addLayout(filter_layout)
        
        # A single horizontal row overflows the inspector on compact screens.
        # Keep the controls in a responsive two-column grid instead.
        flags_layout = QGridLayout()
        flags_layout.setContentsMargins(0, 0, 0, 0)
        flags_layout.setHorizontalSpacing(10)
        flags_layout.setVerticalSpacing(8)
        flags_layout.setColumnStretch(0, 1)
        flags_layout.setColumnStretch(1, 1)
        flags_layout.addWidget(self.normal_invert_x, 0, 0)
        flags_layout.addWidget(self.normal_invert_y, 0, 1)
        flags_layout.addWidget(self.normal_invert_height, 1, 0)
        self.normal_card.body.addLayout(flags_layout)

        self.normal_card.body.addWidget(QLabel("Format"))
        self.normal_card.body.addWidget(self.normal_format)

        self.normal_card.body.addWidget(self.normal_min_z)
        self.normal_card.body.addWidget(self.normal_scale)
        
        self.layout.addWidget(self.normal_card)

        # 2. ROUGHNESS
        self.roughness_card = PluginCard("Roughness", "Surface micro-roughness and gloss levels.")
        self.rough_intensity = LabeledSlider("Intensity", 0, 100, 50, "%")
        self.rough_contrast = LabeledSlider("Contrast", 0, 100, 0, "%")
        self.rough_invert = QCheckBox("Invert Roughness (Glossiness)")
        for w in [self.rough_intensity, self.rough_contrast]:
            w.valueChanged.connect(self._on_live_update)
            self.roughness_card.body.addWidget(w)
        self.rough_invert.toggled.connect(self._on_live_update)
        self.roughness_card.body.addWidget(self.rough_invert)
        self.layout.addWidget(self.roughness_card)

        # 3. AMBIENT OCCLUSION
        self.ao_card = PluginCard("Ambient Occlusion", "Micro-shadow depth and spread.")
        self.ao_intensity = LabeledSlider("Depth", 0, 100, 50, "%")
        self.ao_spread = LabeledSlider("Spread", 0, 100, 30, "%")
        self.ao_invert = QCheckBox("Invert AO")
        for w in [self.ao_intensity, self.ao_spread]:
            w.valueChanged.connect(self._on_live_update)
            self.ao_card.body.addWidget(w)
        self.ao_invert.toggled.connect(self._on_live_update)
        self.ao_card.body.addWidget(self.ao_invert)
        self.layout.addWidget(self.ao_card)

        # 5. DISPLACEMENT
        self.height_card = PluginCard("Displacement", "Surface displacement for tessellation and parallax.")
        self.height_depth = LabeledSlider("Depth Scale", 0, 100, 50, "%")
        # Keep the default height signal intact; NVIDIA's normal conversion
        # applies its derivative filter directly instead of pre-blurring it.
        self.height_smooth = LabeledSlider("Smoothing", 0, 100, 0, "%")
        self.height_invert = QCheckBox("Invert Displacement")
        self.height_contrast = QComboBox()
        self.height_contrast.addItems(["Balanced", "Auto", "Soft", "Sharp"])
        for w in [self.height_depth, self.height_smooth]:
            w.valueChanged.connect(self._on_live_update)
            self.height_card.body.addWidget(w)
        self.height_invert.toggled.connect(self._on_live_update)
        self.height_contrast.currentTextChanged.connect(self._on_live_update)
        self.height_card.body.addWidget(self.height_invert)
        self.height_card.body.addWidget(QLabel("Contrast Handling"))
        self.height_card.body.addWidget(self.height_contrast)
        self.layout.addWidget(self.height_card)

        # 6. OPACITY
        self.opacity_card = PluginCard("Opacity", "Transparency and alpha masking.")
        self.alpha_threshold = LabeledSlider("Threshold", 0, 100, 100, "%")
        self.alpha_softness = LabeledSlider("Edge Softness", 0, 100, 0, "%")
        for w in [self.alpha_threshold, self.alpha_softness]:
            w.valueChanged.connect(self._on_live_update)
            self.opacity_card.body.addWidget(w)
        self.layout.addWidget(self.opacity_card)

        self.layout.addStretch()

        self.export_btn = self.add_bottom_button("EXPORT MAP", self.exportClicked.emit)
        self.export_btn.setEnabled(False)

    def _on_live_update(self, *_args):
        self.parametersChanged.emit()
        self.livePreviewRequested.emit()

    def set_active_map(self, name):
        """Show controls for the selected material channel only."""
        n = name.lower()
        cards = [
            self.normal_card,
            self.roughness_card,
            self.ao_card,
            self.height_card,
            self.opacity_card,
        ]
        show_all = "base" in n or not n
        for card in cards:
            card.setVisible(show_all)
        active = None
        if "normal" in n:
            active = self.normal_card
        elif "roughness" in n:
            active = self.roughness_card
        elif n == "ao":
            active = self.ao_card
        elif "displacement" in n:
            active = self.height_card
        elif "opacity" in n:
            active = self.opacity_card
        elif "base" in n:
            active = self.normal_card
        if active is not None:
            for card in cards:
                card.setVisible(card is active if not show_all else True)
            self.layout.removeWidget(active)
            self.layout.insertWidget(2, active)
            QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(0))

    def get_parameters(self):
        return {
            "height_source": self.height_source.currentText().lower().replace(" ", "_"),
            "normal_filter": self.normal_filter.currentData(),
            "normal_wrap": self.normal_wrap.isChecked(),
            "normal_invert_x": self.normal_invert_x.isChecked(),
            "normal_invert_y": self.normal_invert_y.isChecked(),
            "normal_map_type": "normal",
            "normal_format": self.normal_format.currentText().lower(),
            "normal_invert_height": self.normal_invert_height.isChecked(),
            "normal_min_z": self.normal_min_z.value(),
            "normal_scale": self.normal_scale.value(),
            
            "rough_intensity": self.rough_intensity.value() / 100.0,
            "rough_contrast": self.rough_contrast.value() / 100.0,
            "rough_invert": self.rough_invert.isChecked(),
            "ao_intensity": self.ao_intensity.value() / 100.0,
            "ao_spread": self.ao_spread.value() / 100.0,
            "ao_invert": self.ao_invert.isChecked(),
            "height_depth": self.height_depth.value() / 100.0,
            "height_smooth": self.height_smooth.value() / 100.0,
            "height_invert": self.height_invert.isChecked(),
            "height_contrast": self.height_contrast.currentText().lower(),
            "alpha_threshold": self.alpha_threshold.value() / 100.0,
            "alpha_softness": self.alpha_softness.value() / 100.0,
        }

    def set_image_loaded(self, loaded):
        self.export_btn.setEnabled(loaded)

    def get_export_format(self):
        return "png"

    def get_save_mode(self):
        return "new_file"
