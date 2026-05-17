# Seamless Texture Maker

A powerful, GPU-accelerated desktop application for creating perfectly seamless textures for 3D workflows. Built with Python, PyQt6, and OpenCV with CUDA support.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features (New in v2.0)

### 🎨 Seamless Texture Generation & Delighting
- **Delighting Algorithm** - Remove directional light and shadows from photos to create flat base colors.
- **Standard Method** - Offset + Inpaint with edge blending.
- **Overlap Technique** - Tile overlap with smooth falloff.
- **Splat Technique** - Texture bombing for organic patterns.

### 🔮 PBR Material Studio
- **Full PBR Map Generation** - Automatically derive Normal, Roughness, Metallic, AO, Height, and Displacement maps from a single image.
- **Material Lab Controls** - Tweak intensity, blur, and contrast for each PBR channel in real-time.

### 🌐 3D Viewport & Preview
- **Real-Time 3D Preview** - View your seamless PBR material on 3D meshes (Sphere, Cube, Plane).
- **HDRI Lighting** - Test your materials in various lighting environments (Studio, Outdoor, Archviz).
- **Live Tiling & Displacement** - Visualize tessellation and displacement directly in the 3D viewport.
- **Workspace Modes** - Toggle between "Classic Mode" (2D) and "Studio Mode" (split 2D/3D workspace).

### 🚀 Performance & Export Pipelines
- **Renderer-Specific Export** - One-click export for Unreal Engine 5 (ORM packing), Blender (Node setup), V-Ray, and Corona (MaxScript generation).
- **GPU Acceleration** - CUDA-optimized operations and Numba JIT compilation for heavy map generation.
- **Multi-threaded Architecture** - Background processing keeps the UI fully responsive.

## 📸 Screenshots

*Coming soon*

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional, falls back to CPU)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Seasoned-In-Chaos/seamless-texture-maker.git
cd seamless-texture-maker

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## 🎮 Usage

### Quick Start

1. **Load Image** - Drag & drop an image or use File → Open.
2. **Delight** - (Optional) Use the Delight tab to remove uneven lighting.
3. **Seamless** - Make the texture tileable using Standard, Overlap, or Splat modes.
4. **Material Lab** - Generate and tweak PBR maps (Normal, Roughness, AO, etc.).
5. **Studio Mode** - Switch to Studio Mode to preview your material on a 3D mesh.
6. **Export** - Use `Ctrl+Shift+E` to export the full PBR package for your target renderer.

### Keyboard Shortcuts

- `Ctrl+O` - Open image
- `Ctrl+S` - Save current map
- `Ctrl+Shift+E` - Export PBR Pipeline
- `1 / 2 / 3` - Switch between Delight, Seamless, and Material Lab modes.
- `F1` - Show Shortcuts
- `Escape` - Exit Fullscreen Mode

## 🏗️ Architecture

### Core Modules

```text
seamless-texture-maker/
├── app/
│   ├── core/              # Processing algorithms
│   │   ├── seamless.py    # Main seamless processor
│   │   ├── delighting.py  # Delighting algorithm
│   │   ├── normal_generator.py # PBR Map generation
│   │   └── gpu_utils.py   # CUDA/Numba acceleration
│   ├── gui/               # User interface
│   │   ├── main_window.py # Main application window
│   │   ├── image_viewer.py# 2D Viewport & Workspace Splitter
│   │   ├── pbr_viewport.py# OpenGL 3D Viewport
│   │   └── styles.py      # Premium dark theme
│   └── utils/             # Utilities
├── main.py                # Entry point
└── build.bat              # PyInstaller build script
```

## 🔧 Technical Details

### Edge Blending Algorithm

Traditional approaches use Gaussian blur which destroys edge details. Our implementation uses:

1. **Distance-based gradient** - Calculate distance from seam
2. **Smoothstep interpolation** - Smooth falloff curve: `t² × (3 - 2t)`
3. **Neighbor blending** - Blend with wrapped opposite edge
4. **Vectorized operations** - Process all offsets at once

**Result:** Sharp seams with proper falloff, ~10x faster than loop-based approach.

### GPU Acceleration

When CUDA is available:
- Image resizing on GPU
- Alpha blending operations
- Gaussian blur (when needed)
- Automatic CPU fallback

### Live Preview System

**Dual-Timer Architecture:**
```python
# Live preview: 50ms throttle for instant feel
update_timer.setInterval(50)
update_timer.timeout → request_live_preview()

# Full resolution: 400ms after slider release  
fullres_timer.setInterval(400)
fullres_timer.timeout → process_texture()
```

## 🚀 Performance

### Benchmarks

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Edge Blend | 120ms | 12ms | **10x faster** |
| Live Preview | N/A | 50ms | **20fps** |
| Splat (512px) | 850ms | 180ms | **4.7x faster** |
| GPU vs CPU | Baseline | 3-5x | **CUDA** |

*Tested on: i7-10700K, RTX 3070, 2048×2048 texture*

## 🗺️ Roadmap

### Phase 1: Quick Wins ✅
- [x] Distance-based edge falloff
- [x] Live preview system
- [x] GPU acceleration
- [x] Vectorized operations

### Phase 2: Real-Time (Planned)
- [ ] Numba JIT compilation
- [ ] Result caching
- [ ] Parallel channel processing
- [ ] Remove throttling

### Phase 3: GPU Shaders (Future)
- [ ] GLSL compute shaders
- [ ] Direct GPU texture processing
- [ ] 60fps full-resolution preview
- [ ] Real-time 4K support

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Inspired by [Materialize](https://github.com/BoundingBoxSoftware/Materialize)
- OpenCV for image processing
- PyQt6 for the GUI framework

## 📧 Contact

For issues and questions, please use the [GitHub Issues](https://github.com/Seasoned-In-Chaos/seamless-texture-maker/issues) page.

---

**Made with ❤️ for the 3D artist community**
