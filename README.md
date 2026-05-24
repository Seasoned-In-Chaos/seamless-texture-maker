# SEAMS - Seamless Texture Studio

A powerful, GPU-accelerated desktop application for creating perfectly seamless textures and PBR materials for 3D workflows. Built with Python, PyQt6, OpenCV with CUDA support, and a Rust extension module.

![Version](https://img.shields.io/badge/version-3.0.0-blue)
![Python](https://img.shields.io/badge/python-3.12-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![Platform](https://img.shields.io/badge/platform-Windows-blue)

## Features (New in v3.0)

### Seamless Texture Generation & Delighting
- **Delighting Algorithm** - Remove directional light and shadows from photos to create flat base colors.
- **Standard Method** - Offset + Inpaint with edge blending.
- **Overlap Technique** - Tile overlap with smooth falloff.
- **Splat Technique** - Texture bombing for organic patterns.
- **Chunked Blending** - 2D fade at tile corners eliminates seam artifacts.

### PBR Material Studio
- **Full PBR Map Generation** - Automatically derive Normal, Roughness, Metallic, AO, Height, and Displacement maps from a single image.
- **Material Lab Controls** - Tweak intensity, blur, and contrast for each PBR channel in real-time.

### 3D Viewport & Preview
- **Real-Time 3D Preview** - View your seamless PBR material on 3D meshes (Sphere, Cube, Plane).
- **HDRI Lighting** - Test your materials in various lighting environments (Studio, Outdoor, Archviz).
- **Live Tiling & Displacement** - Visualize tessellation and displacement directly in the 3D viewport.
- **Workspace Modes** - Toggle between "Classic Mode" (2D) and "Studio Mode" (split 2D/3D workspace).

### Performance & Export Pipelines
- **Renderer-Specific Export** - One-click export for Unreal Engine 5 (ORM packing), Blender (Node setup), V-Ray, and Corona (MaxScript generation).
- **GPU Acceleration** - CUDA-optimized operations and Numba JIT compilation for heavy map generation.
- **Rust Extension** - Critical path operations (edge blending, gradient computation, splat) in native Rust via PyO3.
- **Multi-threaded Architecture** - Background processing keeps the UI fully responsive.
- **Result Caching** - LRU cache with pipeline and PBR buckets avoids redundant recomputation.

### Auto-Update & Security
- **Built-in Auto-Updater** - Check for new versions, download, and apply updates on restart.
- **SHA256 Verification** - All update downloads are integrity-checked before applying.

## Screenshots

*Coming soon*

## Installation

### System Requirements

- Windows 10/11 (64-bit)
- Python 3.12
- NVIDIA GPU with CUDA support (optional, falls back to CPU)
- Rust toolchain (for building `seams_core` extension)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Seasoned-In-Chaos/seamless-texture-maker.git
cd seamless-texture-maker

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build the Rust extension
pip install maturin
cd seams_core && maturin develop --release && cd ..

# Run the application
python main.py
```

### Building the Installer

```bash
# Run the full build pipeline (venv, deps, MSVC, Rust, PyInstaller, signing, Inno Setup)
build.bat

# Or build for Microsoft Store (MSIX package)
build_store.bat
```

The build produces:
- **Standard build**: Single-file EXE via PyInstaller + Inno Setup installer (~131 MB)
- **Store build**: One-directory build for MSIX packaging

## Usage

### Quick Start

1. **Load Image** - Drag & drop an image or use File > Open.
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

## Architecture

### Project Structure

```text
seamless-texture-maker/
├── app/
│   ├── core/                    # Processing algorithms
│   │   ├── seamless.py          # Main seamless processor
│   │   ├── delighting.py        # Delighting algorithm
│   │   ├── normal_generator.py  # PBR map generation
│   │   ├── ao_generator.py      # Ambient occlusion generation
│   │   ├── edge_blending.py     # Edge blending (Python)
│   │   ├── edge_blending_jit.py # Edge blending (Numba JIT)
│   │   ├── inpainting.py        # Inpainting algorithm
│   │   ├── materialize_methods.py    # Materialize (Python)
│   │   ├── materialize_methods_jit.py # Materialize (Numba JIT)
│   │   ├── offset_mapping.py    # Offset mapping
│   │   ├── gpu_utils.py         # CUDA/Numba acceleration
│   │   ├── cache.py             # LRU result cache
│   │   ├── assertions.py        # Runtime assertions
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── warmup.py            # JIT warmup on startup
│   ├── gui/                     # User interface
│   │   ├── main_window.py       # Main application window
│   │   ├── image_viewer.py      # 2D Viewport & Workspace Splitter
│   │   ├── pbr_viewport.py      # OpenGL 3D Viewport
│   │   ├── splash_screen.py     # Splash screen
│   │   ├── controls.py          # Main controls panel
│   │   ├── normal_controls.py   # Normal map controls
│   │   ├── export_dialog.py     # Export dialog
│   │   ├── credits_dialog.py    # Credits dialog
│   │   ├── system_monitor.py    # System resource monitor
│   │   └── styles.py            # Premium dark theme
│   └── utils/                   # Utilities
│       ├── config.py            # Settings persistence
│       ├── image_io.py          # Image read/write
│       ├── app_logging.py       # Logging setup
│       ├── perf.py              # Performance monitoring
│       └── updater.py           # Auto-updater with SHA256 verification
├── seams_core/                  # Rust extension (PyO3)
│   ├── src/
│   │   ├── lib.rs               # Module entry point
│   │   ├── edge_blend.rs        # Edge blending (Rust)
│   │   ├── gradients.rs         # Gradient computation (Rust)
│   │   └── splat.rs             # Splat algorithm (Rust)
│   ├── Cargo.toml
│   └── pyproject.toml
├── tests/                       # Test suite
│   ├── test_cache.py
│   ├── test_edge_blending.py
│   ├── test_image_io.py
│   ├── test_normal_generator.py
│   └── test_splat.py
├── benchmarks/                  # Performance benchmarks
├── store/                       # Microsoft Store assets
├── .github/workflows/build.yml  # CI pipeline
├── build.bat                    # Standard build script
├── build_store.bat              # MSIX store build script
├── build.spec                   # PyInstaller spec (onefile)
├── build_store.spec             # PyInstaller spec (onedir)
├── installer.iss                # Inno Setup config
├── main.py                      # Entry point
└── version_info.txt             # Windows version info
```

## Technical Details

### Edge Blending Algorithm

Traditional approaches use Gaussian blur which destroys edge details. Our implementation uses:

1. **Distance-based gradient** - Calculate distance from seam
2. **Smoothstep interpolation** - Smooth falloff curve: `t^2 * (3 - 2t)`
3. **Neighbor blending** - Blend with wrapped opposite edge
4. **Chunked 2D fade** - At tile corners, `fade = fade_y * fade_x` for artifact-free blending
5. **Vectorized operations** - Process all offsets at once

**Result:** Sharp seams with proper falloff, ~10x faster than loop-based approach.

### Rust Extension (seams_core)

Performance-critical operations are implemented in Rust via PyO3:
- **Edge blending** - Direct numpy array access for zero-copy processing
- **Gradient computation** - Sobel gradient fields for normal map generation
- **Splat** - Texture bombing with gaussian splat distribution

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
update_timer.timeout -> request_live_preview()

# Full resolution: 400ms after slider release
fullres_timer.setInterval(400)
fullres_timer.timeout -> process_texture()
```

### Result Cache

LRU cache with separate buckets for pipeline and PBR results:
- `get_pipeline(key)` / `set_pipeline(key, result)` - Seamless processing cache
- `get_pbr(key)` / `set_pbr(key, result)` - PBR map generation cache
- Automatic eviction when cache size exceeds limits

## Performance

### Benchmarks

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Edge Blend | 120ms | 12ms | **10x faster** |
| Live Preview | N/A | 50ms | **20fps** |
| Splat (512px) | 850ms | 180ms | **4.7x faster** |
| GPU vs CPU | Baseline | 3-5x | **CUDA** |
| Build Size | 185 MB | 131 MB | **30% smaller** |

*Tested on: i7-10700K, RTX 3070, 2048x2048 texture*

## Roadmap

### Phase 1: Quick Wins (Done)
- [x] Distance-based edge falloff
- [x] Live preview system
- [x] GPU acceleration
- [x] Vectorized operations
- [x] Numba JIT compilation
- [x] Result caching
- [x] Rust extension for critical paths

### Phase 2: Polish & Ship (In Progress)
- [x] Auto-updater with SHA256 verification
- [x] DPI-aware window sizing (1080p+)
- [x] PyInstaller packaging with selective Qt6 imports
- [x] Inno Setup installer
- [ ] Microsoft Store (MSIX) distribution
- [ ] Parallel channel processing

### Phase 3: GPU Shaders (Future)
- [ ] GLSL compute shaders
- [ ] Direct GPU texture processing
- [ ] 60fps full-resolution preview
- [ ] Real-time 4K support

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- Inspired by [Materialize](https://github.com/BoundingBoxSoftware/Materialize)
- OpenCV for image processing
- PyQt6 for the GUI framework
- PyO3 for Python-Rust interop

## Contact

For issues and questions, please use the [GitHub Issues](https://github.com/Seasoned-In-Chaos/seamless-texture-maker/issues) page.

---

**Made with care for the 3D artist community**
