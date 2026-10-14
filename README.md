# SEAMS - Seamless Texture Studio

A native desktop application for turning photos into seamless textures and PBR material maps. Written in Rust, with an egui interface rendered on the GPU.

![Version](https://img.shields.io/badge/version-4.0.0-blue)
![Rust](https://img.shields.io/badge/rust-1.90%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows-blue)

## Features

### Seamless tiling

Four methods, each with a live preview:

- **Offset cross-fade**: shifts the image by half and cross-fades the seams at the centre.
- **Mirror tiling**: an exact 2x2 reflected tile. The result is twice the source size.
- **Overlap blend**: overlaps opposite edges and blends the join.
- **Splat synthesis**: scatters rotated patches of the source (texture bombing) for irregular materials.

### Delighting

Removes baked lighting from a photographed base colour: shadow, highlight and ambient occlusion removal, flattening, and contrast, detail and colour recovery. Soft, Balanced and Aggressive presets. Applying a delight is undoable.

### PBR maps

Normal, roughness, ambient occlusion, displacement and opacity maps generated from one height source. Normal maps offer the 4-sample, Sobel 3x3 to 9x9 and dUdV filters, OpenGL or DirectX orientation, and tiling-aware sampling.

### Viewing and export

- Result, split, side-by-side and tiled views, with pan and zoom.
- Live previews on the GPU: the material maps and three of the four seamless methods follow a slider at full resolution.
- PNG, JPG, TIFF, TGA and EXR import and export.
- Normal maps export at 16 bits per channel in PNG and TIFF. A 16-bit source keeps its depth.
- Files are written atomically: an interrupted save never leaves a corrupt file.

## Installation

Download the installer or the standalone `SEAMS.exe` from the [releases page](https://github.com/Seasoned-In-Chaos/seamless-texture-maker/releases/latest). Requires 64-bit Windows 10 or 11.

## Usage

1. **Open** a texture with `Ctrl+O`, or drop an image on the window.
2. **Delight** (optional): even out the lighting, then apply it.
3. **Seamless**: pick a method and tune it. The tiled view shows how it repeats.
4. **Material**: select a map in the strip below the canvas and adjust it.
5. **Export** the selected image with `Ctrl+E`, or save the base colour with `Ctrl+S`.

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Open texture |
| `Ctrl+S` / `Ctrl+Shift+S` | Save base colour / save as |
| `Ctrl+E` | Export the selected map |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo |
| `1` / `2` / `3` | Delight / Seamless / Material |
| `Ctrl+0` / `Ctrl+1` | Fit to window / actual size |
| `Ctrl+G` | Toggle tile guides |
| `F1` | Keyboard shortcuts |

## Building from source

Requirements: [Rust](https://rustup.rs) 1.90 or newer. With the GNU toolchain (`x86_64-pc-windows-gnu`), also MinGW-w64: `winget install BrechtSanders.WinLibs.POSIX.UCRT`. The MSVC toolchain needs the Visual Studio C++ build tools instead.

```bash
git clone https://github.com/Seasoned-In-Chaos/seamless-texture-maker.git
cd seamless-texture-maker

cargo run --release -p seams      # run the app
cargo test --workspace            # run the tests
build.bat                         # tests, release exe and installer
```

`build.bat` writes `dist\SEAMS.exe` and, when [Inno Setup](https://jrsoftware.org/isinfo.php) is installed, `dist\SEAMS_Setup_<version>.exe`. `build_store.bat` packages an MSIX and needs the Windows 10 SDK.

## Architecture

A Cargo workspace in layers. A crate depends only on the ones above it, and only the app uses a UI toolkit.

```text
crates/
  core/      seams-core    Pixel processing: seamless methods, delight, PBR maps, filters, cache.
                           Pure functions on f32 images. No files, no UI, no global state.
  io/        seams-io      Reading and writing image files; validation; atomic saves.
  engine/    seams-engine  The document and its undo history, the render pipeline and its
                           caches, background workers, persisted settings.
  gpu/       seams-gpu     Live previews with wgpu: the material maps and the per-pixel
                           seamless methods as shaders, checked against seams-core.
apps/
  seams/                   The egui desktop shell: theme, panels, canvas, dialogs.
```

How it fits together:

- **Everything is an action.** Menus, shortcuts and buttons produce an `Action` (`apps/seams/src/actions.rs`), executed in one place.
- **The UI thread never touches pixels.** Rendering runs on worker threads. Each edit starts a new generation, and results from older generations are discarded when they arrive.
- **Live preview, then full resolution.** A reduced-size preview updates while a control moves; the full-resolution render starts once input pauses.
- **The GPU previews, the CPU decides.** The material maps and the offset, mirror and overlap methods are also drawn by shaders at full resolution, so a slider is followed without waiting. Export always uses the CPU code, and parity tests hold the shaders within one 8-bit step of it. Splat, delight, an image beyond the GPU's limits or a lost device fall back to the CPU preview; `SEAMS_CPU_PREVIEW=1` forces it.
- **No panics in non-test code.** `unsafe` is forbidden and `unwrap`/`expect`/`panic!` are denied by lint. Malformed files and out-of-range settings return errors or are clamped. A panic inside a worker is caught and reported as a failed job.
- **One design system.** Colours, radii and type sizes come from `apps/seams/src/theme.rs`; contrast ratios are checked by tests.

## Performance

Measured in a release build on a 4096 x 4096 RGB texture, on the development machine:

| Step | Time |
|---|---|
| Offset cross-fade | 0.21 s |
| Overlap blend | 0.31 s |
| Mirror tiling (8192 x 8192 result) | 0.42 s |
| Splat synthesis | 0.64 s |
| All five maps | 0.35 s |

Processing uses every CPU core through [rayon](https://github.com/rayon-rs/rayon). Images are held as 32-bit float, so a 4K texture with its maps uses a few gigabytes of memory, more with mirror tiling.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

MIT License - see [LICENSE](LICENSE).

## Acknowledgments

- Map generation follows the conventions of [Materialize](https://github.com/BoundingBoxSoftware/Materialize) and NormalMap-Online.
- [egui](https://github.com/emilk/egui) for the interface, the [image](https://github.com/image-rs/image) crate for file formats.
- [Geist and Geist Mono](https://github.com/vercel/geist-font) typefaces, under the SIL Open Font License (`apps/seams/assets/fonts/OFL.txt`).
- [Lucide](https://lucide.dev) icons (lucide-static 1.57.0), under the ISC License (`apps/seams/assets/fonts/Lucide-LICENSE.txt`).

For issues and questions, use [GitHub Issues](https://github.com/Seasoned-In-Chaos/seamless-texture-maker/issues).
