# Seamless Texture Maker

A powerful, GPU-accelerated desktop application for creating perfectly seamless textures for 3D workflows. Built with Python, PyQt6, and OpenCV with CUDA support.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

### 🎨 Multiple Processing Techniques

- **Standard Method** - Offset + Inpaint with edge blending
- **Overlap Technique** - Tile overlap with smooth falloff
- **Splat Technique** - Texture bombing for organic patterns

### ⚡ Real-Time Live Preview

- **Instant slider updates** - 50ms response time (~20fps)
- **Background processing** - Non-blocking UI
- **Dual-resolution system** - Fast preview + high-quality export

### 🚀 Performance Optimizations

- **GPU Acceleration** - CUDA-optimized operations
- **Vectorized CPU operations** - NumPy-based processing
- **Smart caching** - Preview resolution optimization
- **Multi-threaded** - Background processing threads

### 🎯 Quality Enhancements

- **Distance-based edge falloff** - No blur, preserves details
- **Smoothstep interpolation** - Smooth gradients
- **Detail preservation** - Maintains texture sharpness

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

### Dependencies

```
PyQt6>=6.4.0
opencv-python>=4.7.0
numpy>=1.24.0
Pillow>=9.5.0
```

## 🎮 Usage

### Quick Start

1. **Load Image** - Drag & drop or use File → Open
2. **Choose Technique** - Standard, Overlap, or Splat
3. **Adjust Parameters** - Use sliders for real-time preview
4. **Export** - Save seamless texture

### Keyboard Shortcuts

- `Ctrl+O` - Open image
- `Ctrl+S` - Save texture
- `Ctrl+0` - Fit to view

### Parameter Guide

#### Standard Method
- **Edge Blend** (0-100%) - Width of seam blending zone
- **Detail Preservation** (0-100%) - Controls inpainting quality

#### Overlap Technique
- **Overlap X/Y** (0-50%) - Amount of tile overlap
- **Edge Falloff** (0-100%) - Softness of blending

#### Splat Technique
- **Edge Falloff** (0-100%) - Splat edge smoothness
- **Splat Scale** (1-5x) - Size multiplier
- **Rotation** (0-360°) - Base rotation angle
- **Random Rotation** (0-100%) - Rotation variation
- **Wobble** (0-100%) - Position randomness

## 🏗️ Architecture

### Core Modules

```
seamless-texture-maker/
├── app/
│   ├── core/              # Processing algorithms
│   │   ├── seamless.py    # Main processor
│   │   ├── edge_blending.py    # Distance-based falloff
│   │   ├── materialize_methods.py  # Overlap & Splat
│   │   ├── inpainting.py  # Seam filling
│   │   ├── gpu_utils.py   # CUDA acceleration
│   │   └── offset_mapping.py   # Image offsetting
│   ├── gui/               # User interface
│   │   ├── main_window.py # Main application window
│   │   ├── controls.py    # Parameter controls
│   │   ├── image_viewer.py    # Split-view display
│   │   └── styles.py      # Dark theme
│   └── utils/             # Utilities
│       ├── image_io.py    # Image loading/saving
│       └── config.py      # App configuration
├── main.py                # Entry point
└── requirements.txt       # Dependencies
```

### Processing Pipeline

```
Load Image → Offset to Center → Process Seams → Blend Edges → Reverse Offset → Export
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
