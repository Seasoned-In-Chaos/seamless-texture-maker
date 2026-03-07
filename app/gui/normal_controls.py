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
    exportClicked = pyqtSignal()
    
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
        
        self.intensity = LabeledSlider("Intensity", 0, 100, 50)
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
        
        # 2. Map Type Group
        type_group = QGroupBox("Map Type")
        type_layout = QHBoxLayout(type_group)
        
        self.type_group = QButtonGroup(self)
        self.normal_radio = QRadioButton("Normal Map (RGB)")
        self.bump_radio = QRadioButton("Bump Map (Grayscale)")
        self.normal_radio.setChecked(True)
        
        self.type_group.addButton(self.normal_radio)
        self.type_group.addButton(self.bump_radio)
        
        type_layout.addWidget(self.normal_radio)
        type_layout.addWidget(self.bump_radio)
        self.type_group.buttonClicked.connect(self._on_live_update)
        
        layout.addWidget(type_group)
        
        # 3. Format Group
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
        
        # 4. Export Options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(10)
        
        # Format selector
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["PNG", "JPG", "TIFF"])
        format_layout.addWidget(self.export_format_combo)
        export_layout.addLayout(format_layout)
        
        # Save mode radio buttons
        self.save_mode_group = QButtonGroup(self)
        
        self.new_file_radio = QRadioButton("Save as new file (_normal/_bump)")
        self.new_file_radio.setChecked(True)
        self.save_mode_group.addButton(self.new_file_radio, 0)
        export_layout.addWidget(self.new_file_radio)
        
        self.overwrite_radio = QRadioButton("Overwrite original")
        self.save_mode_group.addButton(self.overwrite_radio, 1)
        export_layout.addWidget(self.overwrite_radio)
        
        # Export button
        self.export_btn = QPushButton("Export Map")
        self.export_btn.clicked.connect(self.exportClicked.emit)
        export_layout.addWidget(self.export_btn)
        
        layout.addWidget(export_group)

        layout.addStretch()
        
        # Status label
        self.status_label = QLabel("Maps update automatically")
        self.status_label.setStyleSheet("color: #888; font-size: 11px; padding: 10px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    def _on_live_update(self, *args):
        self.livePreviewRequested.emit()
        
    def get_parameters(self):
        intensity_val = self.intensity.value() / 100.0
        return {
            'intensity': intensity_val,
            'detail_scale': self.detail_scale.value() / 100.0,
            'smoothness': self.smoothness.value() / 100.0,
            'invert_height': self.invert_height.isChecked(),
            'format': 'directx' if self.directx_radio.isChecked() else 'opengl',
            'contrast_mode': self.contrast_combo.currentText().lower(),
            'map_type': 'bump' if self.bump_radio.isChecked() else 'normal',
            'height_intensity': intensity_val  # Same slider controls both modes
        }
        
    def set_image_loaded(self, loaded):
        # No generate button anymore - everything is automatic
        self.export_btn.setEnabled(loaded)
        
    def get_export_format(self):
        """Get selected export format."""
        return self.export_format_combo.currentText().lower()
        
    def get_save_mode(self):
        """Get save mode: 'new_file' or 'overwrite'."""
        if self.new_file_radio.isChecked():
            return 'new_file'
        return 'overwrite'
