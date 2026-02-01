"""
Main application window for Seamless Texture Maker.
"""
import os
import time
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QSplitter, QLabel,
    QProgressBar, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex
from PyQt6.QtGui import QAction, QIcon

from .image_viewer import ImageViewer
from .controls import ControlPanel
from .styles import get_dark_theme
from ..core.seamless import SeamlessProcessor
from ..core.gpu_utils import is_cuda_available
from ..utils.image_io import load_image, save_image, get_output_path, get_format_filter, get_file_info
from ..utils.config import APP_NAME, APP_VERSION, load_settings, save_settings


class ProcessingThread(QThread):
    """Background thread for processing to keep UI responsive."""
    
    finished = pyqtSignal(object, float)  # result image, processing time
    error = pyqtSignal(str)
    
    def __init__(self, processor, parent=None):
        super().__init__(parent)
        self.processor = processor
    
    def run(self):
        try:
            start_time = time.time()
            result = self.processor.process()
            elapsed = time.time() - start_time
            self.finished.emit(result, elapsed)
        except Exception as e:
            self.error.emit(str(e))


class PreviewThread(QThread):
    """Background thread specifically for high-speed live previews."""
    result_ready = pyqtSignal(object)
    
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.params = None
        self._mutex = QMutex()
        self._restart = False
        
    def request(self, params):
        self._mutex.lock()
        self.params = params.copy()
        self._mutex.unlock()
        
        if not self.isRunning():
            self.start()
        else:
            self._restart = True
            
    def run(self):
        while True:
            self._mutex.lock()
            params = self.params
            self._restart = False
            self._mutex.unlock()
            
            if params:
                try:
                    # Thread-safe preview generation using cached preview image
                    result = self.processor.process(preview=True, params=params)
                    if not self._restart:
                        self.result_ready.emit(result)
                except Exception:
                    pass
            
            self._mutex.lock()
            if not self._restart:
                self._mutex.unlock()
                break
            self._mutex.unlock()


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.processor = SeamlessProcessor()
        self.current_file_path = None
        self.image_metadata = None
        self.processing_thread = None
        self.preview_thread = PreviewThread(self.processor)
        self.preview_thread.result_ready.connect(self._on_preview_ready)
        
        # Load settings
        self.settings = load_settings()
        
        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._connect_signals()
        
        # Apply settings
        self.control_panel.set_parameters(self.settings)
        
        # Set window properties
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(self.settings.get('window_width', 1200), 
                   self.settings.get('window_height', 800))
    
    def _setup_ui(self):
        """Set up the main UI layout."""
        # Apply dark theme
        self.setStyleSheet(get_dark_theme())
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Image viewer (left/center)
        self.image_viewer = ImageViewer()
        splitter.addWidget(self.image_viewer)
        
        # Control panel (right)
        self.control_panel = ControlPanel()
        splitter.addWidget(self.control_panel)
        
        # Set splitter sizes (image viewer gets more space)
        splitter.setSizes([900, 300])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        
        layout.addWidget(splitter)
    
    def _setup_menu(self):
        """Set up menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_file)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._save_file_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        fit_action = QAction("&Fit to View", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self._fit_to_view)
        view_menu.addAction(fit_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_status_bar(self):
        """Set up status bar."""
        self.status_bar = self.statusBar()
        
        # File info
        self.file_label = QLabel("No file loaded")
        self.status_bar.addWidget(self.file_label)
        
        self.status_bar.addPermanentWidget(QLabel("Made by Shubham Panchasara | © 2026"), 1)
        
        # Processing indicator
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(150)
        self.progress.setMaximumHeight(16)
        self.progress.hide()
        self.status_bar.addPermanentWidget(self.progress)

        # Update timer for live preview - 16ms for 60fps feel
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(16)  # 16ms = ~60fps
        self.update_timer.timeout.connect(self._request_live_preview)
        
        # Full resolution timer (when slider released)
        self.fullres_timer = QTimer()
        self.fullres_timer.setSingleShot(True)
        self.fullres_timer.setInterval(300)  # Reduced to 300ms
        self.fullres_timer.timeout.connect(self._process_texture)
    
    def _connect_signals(self):
        """Connect UI signals."""
        self.control_panel.parametersChanged.connect(self._on_parameters_changed)
        self.control_panel.livePreviewRequested.connect(self._on_live_preview_requested)
        self.control_panel.processClicked.connect(self._process_texture)
        self.control_panel.exportClicked.connect(self._export_texture)
    
    def _open_file(self):
        """Open an image file."""
        last_dir = self.settings.get('last_directory', '')
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Texture",
            last_dir,
            get_format_filter()
        )
        
        if file_path:
            self._load_image(file_path)
    
    def _load_image(self, file_path):
        """Load an image from file."""
        try:
            image, metadata = load_image(file_path)
            
            self.processor.load_image(image)
            self.current_file_path = file_path
            self.image_metadata = metadata
            
            # Update UI
            self.image_viewer.set_before_image(image)
            self.image_viewer.set_after_image(None)
            self.image_viewer.fit_to_view()
            
            self.control_panel.set_image_loaded(True)
            self.control_panel.set_processed(False)
            
            # Update status
            file_info = get_file_info(file_path)
            self.file_label.setText(
                f"{file_info['name']} | {metadata.width}x{metadata.height} | {file_info['size_str']}"
            )
            
            # Save last directory
            self.settings['last_directory'] = os.path.dirname(file_path)
            
            self.setWindowTitle(f"{APP_NAME} - {os.path.basename(file_path)}")
            
            # Automatically process texture on load so canvas isn't blank
            # (Deferred slightly to allow UI to update first)
            QTimer.singleShot(100, self._process_texture)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
    
    def _on_parameters_changed(self):
        """Handle parameter changes (slider released or final value)."""
        # Schedule full resolution update
        self.fullres_timer.stop()
        self.fullres_timer.start()
    
    def _on_live_preview_requested(self):
        """Handle live preview request while slider is being dragged."""
        params = self.control_panel.get_parameters()
        
        # If image loaded, show live preview via background thread
        if self.processor._original_image is not None:
            # Stop any pending full update
            self.fullres_timer.stop()
            
            # Throttle preview requests  
            if not self.update_timer.isActive():
                self.update_timer.start()
    
    def _request_live_preview(self):
        """Actually request preview after throttle delay."""
        params = self.control_panel.get_parameters()
        if self.processor._original_image is not None:
            self.preview_thread.request(params)
             
    def _on_preview_ready(self, preview_result):
        """Update the viewer with the background-processed preview."""
        if preview_result is not None:
            self.image_viewer.set_after_image(preview_result)
    
    def _process_texture(self):
        """Process the texture to make it seamless."""
        if self.processor.original_image is None:
            return
        
        # Show progress
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.show()
        self.control_panel.process_btn.setEnabled(False)
        
        # Update parameters
        params = self.control_panel.get_parameters()
        self.processor.set_parameters(**params)
        
        # Process in background thread
        if self.processing_thread is not None:
            self.processing_thread.wait()
        self.processing_thread = ProcessingThread(self.processor, self)
        self.processing_thread.finished.connect(self._on_processing_finished)
        self.processing_thread.error.connect(self._on_processing_error)
        self.processing_thread.start()
    
    def _on_processing_finished(self, result, elapsed_time):
        """Handle processing completion."""
        self.progress.hide()
        self.control_panel.process_btn.setEnabled(True)
        
        # Update viewer
        self.image_viewer.set_after_image(result)
        
        # Enable export
        self.control_panel.set_processed(True)
        
        # Update status
        self.control_panel.set_info(f"Processed in {elapsed_time:.2f}s")
        self.status_bar.showMessage(f"Processing complete ({elapsed_time:.2f}s)", 3000)
    
    def _on_processing_error(self, error_msg):
        """Handle processing error."""
        self.progress.hide()
        self.control_panel.process_btn.setEnabled(True)
        QMessageBox.critical(self, "Processing Error", f"Failed to process texture:\n{error_msg}")
    
    def _save_file(self):
        """Save with current settings."""
        if self.processor.processed_image is None:
            return
        
        save_mode = self.control_panel.get_save_mode()
        
        if save_mode == 'overwrite':
            # Confirm overwrite
            reply = QMessageBox.question(
                self, 'Confirm Overwrite',
                'Are you sure you want to overwrite the original file? This cannot be undone.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._save_to_path(self.current_file_path)
        else:
            # "Save as new file" - Ask user where to save
            self._save_file_as()
    
    def _save_file_as(self):
        """Save with file dialog."""
        if self.processor.processed_image is None:
            return
        
        export_format = self.control_panel.get_export_format()
        default_name = get_output_path(self.current_file_path, '_seamless', export_format)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Seamless Texture",
            default_name,
            get_format_filter()
        )
        
        if file_path:
            self._save_to_path(file_path)
    
    def _save_to_path(self, file_path):
        """Save the processed image to a file."""
        try:
            save_image(
                self.processor.processed_image,
                file_path,
                metadata=self.image_metadata
            )
            self.status_bar.showMessage(f"Saved: {file_path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save image:\n{str(e)}")
    
    def _export_texture(self):
        """Export the processed texture."""
        self._save_file()
    
    def _fit_to_view(self):
        """Fit image to view."""
        self.image_viewer.fit_to_view()
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<h3>{APP_NAME}</h3>"
            f"<p>Version {APP_VERSION}</p>"
            f"<p>Create perfectly seamless textures for 3D workflows.</p>"
            f"<p><b>Created by Shubham Panchasara</b></p>"
            f"<p><a href='https://www.instagram.com/panchasarashubham/'>Instagram: @panchasarashubham</a></p>"
            f"<p>Supports: 3ds Max, Corona, V-Ray, Unreal, Blender</p>"
            f"<hr>"
            f"<p>GPU: {'CUDA Available' if is_cuda_available() else 'CPU Mode'}</p>"
        )
    
    def closeEvent(self, event):
        """Save settings on close."""
        self.settings['window_width'] = self.width()
        self.settings['window_height'] = self.height()
        self.settings.update(self.control_panel.get_parameters())
        save_settings(self.settings)
        
        # Wait for processing thread if running (with timeout)
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.quit()
            if not self.processing_thread.wait(3000):  # 3 second timeout
                self.processing_thread.terminate()
        
        event.accept()
    
    def dragEnterEvent(self, event):
        """Handle drag enter for file drops."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                    event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle file drop."""
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            self._load_image(file_path)
