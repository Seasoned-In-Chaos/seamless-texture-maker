# SEAMS - Seamless Texture Studio

A powerful desktop application for creating perfectly seamless textures and PBR materials for 3D workflows. Built with Python, PyQt6, and OpenCV, with Numba JIT-accelerated map generation.

![Version](https://img.shields.io/badge/version-3.2.0-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![Platform](https://img.shields.io/badge/platform-Windows-blue)

## Features (New in v3.0)

### Seamless Texture Generation & Delighting
- **Delighting Algorithm** - Remove directional light and shadows from photos to create flat base colors.
- **Offset + Cross-Fade** - 50% offset with linear center-seam feathering.
- **Mirror Tiling** - Exact 2×2 reflection for zero-break edge repetition.
- **Overlap Technique** - Tile overlap with smooth falloff.
- **Splat Technique** - Texture bombing for organic patterns.
- **Chunked Blending** - 2D fade at tile corners eliminates seam artifacts.

### PBR Material Studio
- **PBR Map Generation** - Automatically derive Normal, Roughness, AO, Displacement, and Opacity maps from a single image.
- **Material Lab Controls** - Tweak intensity, blur, and contrast for each PBR channel in real-time.

### 3D Viewport & Preview
- **Real-Time 3D Preview** - View your seamless PBR material on 3D meshes (Sphere, Cube, Plane).
- **HDRI Lighting** - Test your materials in various lighting environments (Studio, Outdoor, Archviz).
- **Live Tiling & Displacement** - Visualize tessellation and displacement directly in the 3D viewport.
- **Workspace Modes** - Toggle between "Classic Mode" (2D) and "Studio Mode" (split 2D/3D workspace).

### Performance & Export Pipelines
- **Texture & Map Export** - Save the seamless texture or any individual PBR map (Base Color, Normal, Roughness, AO, Displacement, Opacity) via quick-save, save-as, or per-map export.
- **Numba JIT Acceleration** - JIT-compiled hot paths for heavy map generation, warmed up in the background at startup.
- **Multi-threaded Architecture** - Background processing keeps the UI fully responsive.
- **Result Caching** - LRU cache with pipeline and PBR buckets avoids redundant recomputation.

## Screenshots

*Coming soon*

## Installation

### System Requirements

- Windows 10/11 (64-bit)
- Python 3.11 or newer

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

# Run the application
python main.py
```

### Building the Installer

```bash
# Run the full build pipeline (venv, deps, PyInstaller, signing, Inno Setup)
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
3. **Seamless** - Make the texture tileable using Offset + Cross-Fade, Mirror Tiling, Overlap, or Splat modes.
4. **Material Lab** - Generate and tweak PBR maps (Normal, Roughness, AO, etc.).
5. **Studio Mode** - Switch to Studio Mode to preview your material on a 3D mesh.
6. **Export** - Use `Ctrl+E` to export the active map, or `Ctrl+S` / `Ctrl+Shift+S` to save the seamless texture.

### Keyboard Shortcuts

- `Ctrl+O` - Open image
- `Ctrl+S` - Save current map
- `Ctrl+E` - Export selected map
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
│   │   ├── normal_generator.py  # PBR map generation (Normal, Roughness, AO, Displacement, Opacity)
│   │   ├── materialize_methods.py    # Materialize (Python)
│   │   ├── materialize_methods_jit.py # Materialize (Numba JIT)
│   │   ├── offset_mapping.py    # Offset mapping
│   │   ├── texture_mipmaps.py   # Viewport mipmap generation
│   │   ├── gpu_utils.py         # GPU detection (inactive -- see Technical Details)
│   │   ├── cache.py             # LRU result cache
│   │   ├── assertions.py        # Runtime assertions
│   │   └── exceptions.py        # Custom exceptions
│   ├── gui/                     # User interface
│   │   ├── main_window.py       # Main application window
│   │   ├── image_viewer.py      # 2D Viewport & Workspace Splitter
│   │   ├── pbr_viewport.py      # OpenGL 3D Viewport
│   │   ├── splash_screen.py     # Splash screen
│   │   ├── controls.py          # Main controls panel
│   │   ├── normal_controls.py   # Normal map controls
│   │   ├── credits_dialog.py    # Credits dialog
│   │   ├── system_monitor.py    # System resource monitor
│   │   └── styles.py            # Premium dark theme
│   └── utils/                   # Utilities
│       ├── config.py            # Settings persistence
│       ├── image_io.py          # Image read/write
│       ├── app_logging.py       # Logging setup
│       ├── perf.py              # Performance monitoring
├── tests/                       # Test suite
│   ├── test_cache.py
│   ├── test_image_io.py
│   ├── test_live_preview.py
│   ├── test_normal_generator.py
│   ├── test_overlap_blend.py
│   ├── test_seamless_methods.py
│   ├── test_splat.py
│   └── test_texture_mipmaps.py
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

### Seam Blending

Each seamless method uses its own vectorized, numpy-based feathering rather than a plain Gaussian blur (which destroys edge detail): Offset + Cross-Fade uses a linear falloff from the center seam, while Overlap and Splat use smoothstep-based masks tuned to avoid ghosting on high-detail content. No per-pixel Python loops are involved.

### GPU Acceleration (currently inactive)

`gpu_utils.py` provides CUDA detection and a `GPUAccelerator` wrapper (resize, Gaussian blur, alpha blend, inpaint) with automatic CPU fallback on any failure. It isn't wired into the processing pipeline today because the pinned `opencv-python-headless` package ships without CUDA support compiled in — `is_cuda_available()` returns `False` on every install regardless of the user's GPU. Enabling this for real would require switching to a CUDA-enabled OpenCV build and re-verifying on real hardware; everything currently ships on CPU/Numba, which is fast enough that this hasn't been a priority.

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
| Live Preview | N/A | 50ms | **20fps** |
| Splat (512px) | 850ms | 180ms | **4.7x faster** |
| Build Size | 185 MB | 131 MB | **30% smaller** |

*Tested on: i7-10700K, RTX 3070, 2048x2048 texture*

## Roadmap

### Phase 1: Quick Wins (Done)
- [x] Distance-based edge falloff
- [x] Live preview system
- [x] Vectorized operations
- [x] Numba JIT compilation
- [x] Result caching

### Phase 2: Polish & Ship (In Progress)
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

## Contact

For issues and questions, please use the [GitHub Issues](https://github.com/Seasoned-In-Chaos/seamless-texture-maker/issues) page.

---

**Made with care for the 3D artist community**
