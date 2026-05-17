import re

with open(r'e:\seamless-texture-maker\app\gui\image_viewer.py', 'r') as f:
    content = f.read()

# 1. Fix TextureViewport.paintEvent split mode QRectF
content = content.replace(
    'target_rect = QRect(ox + split_x, oy, sw - split_x, sh)',
    'target_rect = QRectF(float(ox + split_x), float(oy), float(sw - split_x), float(sh))'
)

# 2. Refactor ImageViewer layout to extract the toggle button
# Remove switch_btn from ViewportToolbar
toolbar_pattern = r'self\.setStyleSheet\("background: #0d0e14; border-top: 1px solid #1c1e26;"\).*?layout\.addSpacing\(20\)'
content = re.sub(toolbar_pattern, 'self.setStyleSheet("background: #0d0e14; border-top: 1px solid #1c1e26;")', content, flags=re.DOTALL)

# Remove top bar from StudioWorkspace
studio_pattern = r'top = QHBoxLayout\(\); top\.setContentsMargins\(10,10,10,10\).*?layout\.addLayout\(top\)'
content = re.sub(studio_pattern, '', content, flags=re.DOTALL)

# Inject the toggle button into ImageViewer.__init__
iv_init_search = r'self\.bottom_stack = QStackedWidget\(\).*?layout\.addWidget\(self\.bottom_stack, 0\)'

iv_init_replace = """self.bottom_layout = QHBoxLayout()
        self.bottom_layout.setContentsMargins(10, 5, 10, 5)
        self.bottom_layout.setSpacing(20)
        self.bottom_layout.setStyleSheet("background: #0d0e14; border-top: 1px solid #1c1e26;")
        
        self.toggle_btn = QPushButton("STUDIO MODE")
        self.toggle_btn.setStyleSheet(\"\"\"
            QPushButton { 
                color: #ffffff; background: #5c5edc; font-weight: bold; padding: 6px 15px; 
                border: 1px solid #7c7ee2; border-radius: 4px; font-size: 11px;
            }
            QPushButton:hover { background: #6c6ee2; }
        \"\"\")
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.clicked.connect(self._toggle_mode)
        self.bottom_layout.addWidget(self.toggle_btn)
        
        self.bottom_stack = QStackedWidget()
        self.classic_toolbar = ViewportToolbar()
        self.classic_toolbar.setStyleSheet("background: transparent; border: none;") # remove duplicate background
        self.map_selector = MapSelector()
        self.map_selector.setStyleSheet("background: transparent; border: none;") # remove duplicate background
        self.bottom_stack.addWidget(self.classic_toolbar)
        self.bottom_stack.addWidget(self.map_selector)
        
        self.bottom_layout.addWidget(self.bottom_stack, 1)
        
        bottom_widget = QWidget()
        bottom_widget.setLayout(self.bottom_layout)
        layout.addWidget(bottom_widget, 0)"""

content = re.sub(iv_init_search, iv_init_replace, content, flags=re.DOTALL)

# Remove old connections
content = re.sub(r'self\.classic_toolbar\.switch_btn\.clicked\.connect\(self\.studioModeRequested\.emit\)', '', content)
content = re.sub(r'self\.studio\.switch_btn\.clicked\.connect\(self\.classicModeRequested\.emit\)', '', content)

# Remove the extra line from MapSelector and ViewportToolbar styling
content = content.replace('self.setStyleSheet("background: #0d0e14; border-top: 1px solid #1c1e26;")', '')

# Add _toggle_mode method to ImageViewer
toggle_method = """
    def _toggle_mode(self):
        if self.stack.currentIndex() == 0:
            self.studioModeRequested.emit()
        else:
            self.classicModeRequested.emit()

    def set_workspace(self, index):"""

content = content.replace('def set_workspace(self, index):', toggle_method)

# Update button text on workspace switch
workspace_method = """def set_workspace(self, index):
        self.stack.setCurrentIndex(index)
        self.bottom_stack.setCurrentIndex(index)
        self.toggle_btn.setText("CLASSIC MODE" if index == 1 else "STUDIO MODE")"""

content = content.replace("""def set_workspace(self, index):
        self.stack.setCurrentIndex(index)
        self.bottom_stack.setCurrentIndex(index)""", workspace_method)

with open(r'e:\seamless-texture-maker\app\gui\image_viewer.py', 'w') as f:
    f.write(content)

print("DONE")
