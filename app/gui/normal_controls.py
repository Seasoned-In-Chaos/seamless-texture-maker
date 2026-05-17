"""Material map generation controls."""
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QRadioButton, QWidget, QVBoxLayout

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

        # 1. NORMAL / BUMP
        self.normal_card = PluginCard("Normal & Bump", "Surface direction and detail intensity.")
        self.normal_intensity = LabeledSlider("Intensity", 0, 100, 50, "%")
        self.normal_detail = LabeledSlider("Detail Scale", 0, 100, 50, "%")
        self.normal_smooth = LabeledSlider("Smoothness", 0, 100, 0, "%")
        self.normal_invert_height = QCheckBox("Invert Height")
        self.normal_invert = QCheckBox("Invert Y (DirectX)")
        self.normal_map_type = QComboBox()
        self.normal_map_type.addItems(["Normal Map (RGB)", "Bump Map (Grayscale)"])
        self.normal_contrast = QComboBox()
        self.normal_contrast.addItems(["Balanced", "Auto", "Soft", "Sharp"])
        for w in [self.normal_intensity, self.normal_smooth, self.normal_detail]:
            w.valueChanged.connect(self._on_live_update)
            self.normal_card.body.addWidget(w)
        self.normal_invert_height.toggled.connect(self._on_live_update)
        self.normal_invert.toggled.connect(self._on_live_update)
        self.normal_map_type.currentTextChanged.connect(self._on_live_update)
        self.normal_contrast.currentTextChanged.connect(self._on_live_update)
        self.normal_card.body.addWidget(self.normal_invert_height)
        self.normal_card.body.addWidget(QLabel("Map Type"))
        self.normal_card.body.addWidget(self.normal_map_type)
        self.normal_card.body.addWidget(QLabel("Format"))
        self.normal_card.body.addWidget(self.normal_invert)
        self.normal_card.body.addWidget(QLabel("Contrast Handling"))
        self.normal_card.body.addWidget(self.normal_contrast)
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

        # 3. METALLIC
        self.metallic_card = PluginCard("Metallic", "Surface conductivity and reflection type.")
        self.metal_intensity = LabeledSlider("Metalness", 0, 100, 0, "%")
        self.metal_edge = LabeledSlider("Edge Softness", 0, 100, 20, "%")
        for w in [self.metal_intensity, self.metal_edge]:
            w.valueChanged.connect(self._on_live_update)
            self.metallic_card.body.addWidget(w)
        self.layout.addWidget(self.metallic_card)

        # 4. AMBIENT OCCLUSION
        self.ao_card = PluginCard("Ambient Occlusion", "Micro-shadow depth and spread.")
        self.ao_intensity = LabeledSlider("Depth", 0, 100, 50, "%")
        self.ao_spread = LabeledSlider("Spread", 0, 100, 30, "%")
        for w in [self.ao_intensity, self.ao_spread]:
            w.valueChanged.connect(self._on_live_update)
            self.ao_card.body.addWidget(w)
        self.layout.addWidget(self.ao_card)

        # 5. HEIGHT / DISPLACEMENT
        self.height_card = PluginCard("Height & Displacement", "Vertical depth and tessellation scale.")
        self.height_depth = LabeledSlider("Depth Scale", 0, 100, 50, "%")
        self.height_smooth = LabeledSlider("Smoothing", 0, 100, 10, "%")
        self.displacement_strength = LabeledSlider("Displacement Strength", 0, 100, 20, "%")
        self.height_invert = QCheckBox("Invert Height / Displacement")
        self.height_contrast = QComboBox()
        self.height_contrast.addItems(["Balanced", "Auto", "Soft", "Sharp"])
        for w in [self.height_depth, self.height_smooth, self.displacement_strength]:
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

        # 7. EMISSIVE
        self.emissive_card = PluginCard("Emissive", "Surface glow and light emission.")
        self.glow_intensity = LabeledSlider("Glow Power", 0, 100, 0, "%")
        self.glow_tint = QComboBox()
        self.glow_tint.addItems(["White", "Warm", "Cool", "Custom"])
        self.glow_intensity.valueChanged.connect(self._on_live_update)
        self.glow_tint.currentTextChanged.connect(self._on_live_update)
        self.emissive_card.body.addWidget(self.glow_intensity)
        self.emissive_card.body.addWidget(self.glow_tint)
        self.layout.addWidget(self.emissive_card)

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
            self.metallic_card,
            self.ao_card,
            self.height_card,
            self.opacity_card,
            self.emissive_card,
        ]
        show_all = "base" in n or not n
        for card in cards:
            card.setVisible(show_all)
        active = None
        if "normal" in n:
            active = self.normal_card
        elif "roughness" in n:
            active = self.roughness_card
        elif "metallic" in n:
            active = self.metallic_card
        elif n == "ao":
            active = self.ao_card
        elif "height" in n or "displacement" in n:
            active = self.height_card
        elif "opacity" in n:
            active = self.opacity_card
        elif "emissive" in n:
            active = self.emissive_card
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
            "normal_intensity": self.normal_intensity.value() / 100.0,
            "normal_smooth": self.normal_smooth.value() / 100.0,
            "normal_detail": self.normal_detail.value() / 100.0,
            "normal_invert_height": self.normal_invert_height.isChecked(),
            "normal_map_type": "bump" if "Bump" in self.normal_map_type.currentText() else "normal",
            "normal_format": "directx" if self.normal_invert.isChecked() else "opengl",
            "normal_contrast": self.normal_contrast.currentText().lower(),
            "rough_intensity": self.rough_intensity.value() / 100.0,
            "rough_contrast": self.rough_contrast.value() / 100.0,
            "rough_invert": self.rough_invert.isChecked(),
            "metal_intensity": self.metal_intensity.value() / 100.0,
            "metal_edge": self.metal_edge.value() / 100.0,
            "ao_intensity": self.ao_intensity.value() / 100.0,
            "ao_spread": self.ao_spread.value() / 100.0,
            "height_depth": self.height_depth.value() / 100.0,
            "height_smooth": self.height_smooth.value() / 100.0,
            "height_invert": self.height_invert.isChecked(),
            "height_contrast": self.height_contrast.currentText().lower(),
            "displacement_strength": self.displacement_strength.value() / 100.0,
            "alpha_threshold": self.alpha_threshold.value() / 100.0,
            "alpha_softness": self.alpha_softness.value() / 100.0,
            "glow_intensity": self.glow_intensity.value() / 100.0,
            "glow_tint": self.glow_tint.currentText().lower(),
        }

    def set_image_loaded(self, loaded):
        self.export_btn.setEnabled(loaded)

    def get_export_format(self):
        return "png"

    def get_save_mode(self):
        return "new_file"
