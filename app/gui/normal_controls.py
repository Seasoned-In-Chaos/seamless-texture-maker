"""
Control panel for Normal Map Generator.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QGroupBox, QPushButton, QRadioButton,
    QButtonGroup, QFrame, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from app.gui.controls import LabeledSlider

class NormalControlPanel(QWidget):
    """Control panel for Normal Map parameters."""
    
    parametersChanged = pyqtSignal()
    livePreviewRequested = pyqtSignal()
    generateClicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # 1. Main Parameters Group
        params_group = QGroupBox("Normal Parameters")
        params_layout = QVBoxLayout(params_group)
        
        self.intensity = LabeledSlider("Intensity", 1, 500, 100, suffix="")
        self.intensity.valueChanged.connect(self._on_live_update)
        params_layout.addWidget(self.intensity)
        
        self.detail_scale = LabeledSlider("Detail Scale", 1, 100, 50)
        self.detail_scale.valueChanged.connect(self._on_live_update)
        params_layout.addWidget(self.detail_scale)
        
        self.smoothness = LabeledSlider("Smoothness", 0, 100, 0)
        self.smoothness.valueChanged.connect(self._on_live_update)
        params_layout.addWidget(self.smoothness)
        
        self.invert_height = QCheckBox("Invert Height")
        self.invert_height.toggled.connect(self._on_live_update)
        params_layout.addWidget(self.invert_height)
        
        layout.addWidget(params_group)
        
        # 2. Format Group
        format_group = QGroupBox("Format")
        format_layout = QHBoxLayout(format_group)
        
        self.format_group = QButtonGroup(self)
        self.opengl_radio = QRadioButton("OpenGL (Y+)")
        self.directx_radio = QRadioButton("DirectX (Y-)")
        self.opengl_radio.setChecked(True)
        
        self.format_group.addButton(self.opengl_radio)
        self.format_group.addButton(self.directx_radio)
        
        format_layout.addWidget(self.opengl_radio)
        format_layout.addWidget(self.directx_radio)
        self.format_group.buttonClicked.connect(self._on_live_update)
        
        layout.addWidget(format_group)
        
        # 3. Contrast Handling
        contrast_group = QGroupBox("Contrast Handling")
        contrast_layout = QVBoxLayout(contrast_group)
        
        self.contrast_combo = QComboBox()
        self.contrast_combo.addItems(["Balanced", "Auto", "Soft", "Sharp"])
        self.contrast_combo.currentIndexChanged.connect(self._on_live_update)
        contrast_layout.addWidget(self.contrast_combo)
        
        layout.addWidget(contrast_group)
        
        # 4. Preview System
        preview_group = QGroupBox("Preview System")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_mode = QComboBox()
        self.preview_mode.addItems(["Flat Plane", "Sphere"])
        self.preview_mode.currentIndexChanged.connect(self.parametersChanged)
        preview_layout.addWidget(self.preview_mode)
        
        layout.addWidget(preview_group)
        
        layout.addStretch()
        
        # 5. Actions
        self.generate_btn = QPushButton("Generate Normal Map")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setStyleSheet("background-color: #0098ff; color: white; font-weight: bold;")
        self.generate_btn.clicked.connect(self.generateClicked)
        layout.addWidget(self.generate_btn)

    def _on_live_update(self, *args):
        self.livePreviewRequested.emit()
        
    def get_parameters(self):
        return {
            'intensity': self.intensity.value() / 100.0,
            'detail_scale': self.detail_scale.value() / 100.0,
            'smoothness': self.smoothness.value() / 100.0,
            'invert_height': self.invert_height.isChecked(),
            'format': 'directx' if self.directx_radio.isChecked() else 'opengl',
            'contrast_mode': self.contrast_combo.currentText().lower(),
            'preview_shape': self.preview_mode.currentText().lower()
        }
        
    def set_image_loaded(self, loaded):
        self.generate_btn.setEnabled(loaded)
