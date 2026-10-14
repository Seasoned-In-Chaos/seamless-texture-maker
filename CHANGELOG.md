# Changelog

All notable changes to SEAMS are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [4.0.0] - 2026-10-14

SEAMS is rewritten in Rust. The Python/PyQt6 application is gone; the four seamless methods, delighting and the five PBR maps are ported with the same algorithms and defaults.

### Added
- A Crop tool (`C`, Edit › Crop, or the crop button in the view bar), after Photoshop's Classic Mode. Drag inside the untouched frame to draw a new one, inside a frame to move it, on an edge or corner to resize it, and outside it to turn it. Shift keeps proportions (or snaps the turn to 15°), Alt works from the centre. The inspector sets an aspect ratio (free, original, 1:1, 4:3, 3:4, 16:9, 9:16, 2:1), the exact size and the angle. Enter applies and Esc cancels; applying is undoable. An upright crop copies pixels exactly; a turned one is resampled, repeating edge pixels where it reaches past the image.
- Perspective crop (`Shift+C`, Edit › Perspective crop, or Mode › Perspective in the Crop tool), after Photoshop's Perspective Crop tool. Drag each corner onto a corner of a surface shot at an angle; applying straightens it into a rectangle. Edges move both their corners, the inside moves the whole shape, and a grid follows the perspective. A move that would fold the shape inward is refused. The result size is measured from the edges unless set.
- The Crop tool shows a turning-arrow cursor where a drag will turn the frame, with the angle beside it while turning.
- Icons from Lucide (ISC license) for the crop, pan and fit buttons, tile guides, dropdowns and the empty canvas.
- A pan tool button in the view bar (`H`), and hold `Space` to pan with any tool. The cursor shows a hand while panning.
- Zoom in and Zoom out (`Ctrl+=` / `Ctrl++`, `Ctrl+-`) through Photoshop's zoom levels, also in the View menu and as − / + buttons beside the zoom percentage.
- A Zoom tool (`Z`, or the magnifier in the view bar), as in Photoshop with Scrubby Zoom: click to zoom in at the pointer, Alt-click to zoom out, or drag right and left to zoom continuously about where the drag began. The pan and zoom tools show Lucide icons as the pointer.
- A "Replace original file" option above the export button, for the base color. The button then saves over the opened file in its own format, after a confirmation. It is the same setting as File › Quick save replaces the original.
- `Ctrl+Q` exits. Redo is `Ctrl+Shift+Z`, as in Photoshop; `Ctrl+Y` still works.
- Keyboard focus rings on toggles, segmented controls, checkboxes and the primary button.
- A last-resort guard around every menu, button and shortcut action: an unexpected failure is reported in the status bar instead of closing the window.
- A native Rust application: a Cargo workspace of `seams-core` (processing), `seams-io` (files), `seams-engine` (document, pipeline, workers, settings) and the `seams` egui app.
- TGA and EXR in the export format list, alongside PNG, JPG and TIFF.
- Sixteen-bit export for 16-bit sources in PNG and TIFF. Previously every base colour was written at 8 bits.
- A confirmation before a quick save replaces the original file.
- Open recent in the File menu, opening a file passed on the command line, and a Shuffle control for the splat layout.
- `Ctrl+1` for actual size and `Ctrl+G` for tile guides.
- Live previews on the GPU (`seams-gpu`, wgpu on DirectX 12, Vulkan or Metal): the five material maps and the seamless result for offset cross-fade, mirror tiling and overlap blend follow a slider at full resolution. Export still uses the CPU code, and tests hold the two within one 8-bit step. Splat synthesis, delighting, images beyond the GPU's limits and machines without a usable GPU are previewed on the CPU; `SEAMS_CPU_PREVIEW=1` forces that. Help › About shows which is in use.

### Changed
- A plain left drag no longer moves the view, as in Photoshop: pan with the pan tool (`H`), by holding `Space`, or with the middle button.
- Panning stops at the image: a large image cannot be dragged past its edges, and a small one stays inside the window, as in Photoshop.
- The scroll wheel zooms in exact 10% steps on each notch, without drifting on after the wheel stops. Shift + wheel scrolls sideways, Alt + wheel scrolls up and down, and a pinch or Ctrl + wheel zooms. A middle-button drag always pans.
- The stage switcher, the map thumbnails, the image and the view bar share one centre line over the canvas.
- Tile guides are dashed.
- Slider values are drawn in the secondary text colour, so labels lead. Spacing in the inspector and dialogs follows the 4 px scale.
- `Ctrl+=`, `Ctrl+-` and `Ctrl+0` no longer rescale the whole interface.
- The interface is redesigned to be quiet and monochrome: one top bar with the menus and the workflow stage, a view bar floating over the canvas, one inspector for the current stage with export pinned below it, a map strip under the canvas, and a status bar. Controls are a tone on their surface with a one-pixel outline. One accent colour, a light primary button, one small corner-radius scale, sentence-case labels, and an empty state that asks for a texture. Text is Geist and Geist Mono, fitted to the pixel grid so it stays sharp at 100% display scale. Hover and selection changes are animated. Images show exact pixels when enlarged.
- Whether a quick save writes a new file or replaces the original is now a File menu option.
- The standalone executable is about 19 MB, down from about 198 MB.
- Splat patch size now uses its whole 10-150% range. Values below 25% were silently raised to 25% before.
- Scalar maps (roughness, AO, displacement, opacity) export as single-channel grayscale unless the source has alpha.
- Material maps are always generated from the seamless base colour. A generated map can no longer be run through a seamless method on its own.
- Undo history keeps up to 20 steps within a 1.5 GB memory budget.
- More than one SEAMS window can be open at once.

### Removed
- The 3D viewport and Studio mode, with their mesh, HDRI, tessellation and displacement preview controls.
- NVIDIA CUDA acceleration for Splat. Splat runs on all CPU cores.
- The decorative "Seam Intelligence" and "Light Field Analysis" graphs, which did not analyse the image.
- The Output Type, Bit Depth and "Keep color temperature" controls in Delight, which had no effect.
- The CPU, memory and GPU readout in the status bar, the animated splash screen and the illustrated credits page.

### Fixed
- Results are cached under a hash of every pixel. The hash used to sample about 1,300 pixels, so opening a texture of the same size that differed only in a small area could reuse, and export, the previous texture's result.
- Perspective crop: a drag just outside the region no longer replaces corners placed by hand, and corners grab from a little further out.
- The Crop tool closes when another texture is opened, so a frame drawn for one texture cannot crop the next.
- Esc closing a dropdown in the Crop tool no longer cancels the crop.
- Two SEAMS windows saving settings at the same moment no longer collide on one temporary file.
- Crop: pressing on a corner and pulling outward turned the frame instead of resizing it. A gesture now acts on what was under the pointer when the button went down, and corners grab from a little further out.
- `Shift` + a letter shortcut no longer fires while typing in a value field.
- The application icon and version information were missing from `SEAMS.exe` when the project folder's path contained a space.
- The status bar shows the base color's current size, not the size it was opened at.
- Delight: Highlight reduction and AO removal no longer also lift shadows. Each of Shadow removal, Highlight reduction and AO removal now acts only on its own part of the lighting.
- Delight: Contrast recovery no longer brings back the shadows that Shadow removal took out.
- Delight: Color preservation now keeps lifted shadows saturated instead of only undoing Flatness. On its own it changes nothing, so Apply stays off.
- Delight: Edge consistency and Detail are sized relative to the image, so they are visible on large textures and the live preview matches the full-resolution result.

## Python application: changes after 3.2.0 that were never released

Superseded by 4.0.0, kept for the record.

### Added
- Real GPU acceleration for the Splat method on NVIDIA hardware: its hottest loop now runs on the GPU via Numba's `@cuda.jit` (`materialize_methods_cuda.py`), dispatched automatically above a work-volume threshold and skipped for live-preview canvases where transfer/launch overhead isn't worth it. `nvidia-cuda-nvcc-cu12` and `nvidia-cuda-runtime-cu12` are bundled so end users never install NVIDIA's CUDA Toolkit themselves; any detection or runtime failure falls back to CPU and is never retried for the rest of the session. Adds roughly 20MB to the installer.

### Changed
- Every machine without a working CUDA setup gets a faster Splat path too: the accumulate kernel is now parallelized across CPU cores with `numba.prange` (row-band decomposition) instead of running single-threaded.
- `build.spec` / `build_store.spec` no longer strip Numba's CUDA target from the build (previously dead weight since GPU was inactive); both now also bundle just the CUDA redistributable files Numba actually loads, not the full pip packages.

## [3.2.0] - 2026-08-23

### Added
- A Cross-Fade Controls card (Fade Amount) for Offset + Cross-Fade, which previously exposed no parameters at all.
- Generation and quality-tier gating shared across all three live-preview mechanisms, so a slower low-resolution result can no longer land after and overwrite an already-displayed full-resolution one.
- Soft / Balanced / Aggressive Delight presets are now wired up, plus a Reset to Default button.

### Changed
- Reworked AO and Displacement generation to match the NormalMap-Online reference algorithm, with matching Strength/Mean/Range/Blur-Sharp controls.
- The medium-quality (stage-2) preview now runs through the background `PreviewThread` instead of blocking the GUI thread.
- Rebuilt the Credits dialog for new artwork, sized relative to the main window.
- Reordered the seamless technique dropdown: Offset + Cross-Fade, Mirror, Overlap, Splat.
- Chunked processing of large textures now runs tiles in parallel across a thread pool instead of sequentially, and no longer holds every processed tile in memory at once before reassembling them.
- The offset + cross-fade seam blend is now vectorized instead of looping per pixel offset — noticeably faster live preview when that method is active.
- GPU/CUDA documentation now accurately reflects that acceleration isn't active in this build: the pinned `opencv-python-headless` package has no CUDA support compiled in, so `is_cuda_available()` returns `False` regardless of the user's hardware.
- Company/publisher identity is now consistent across the installer, Windows version resource, and Qt application metadata.
- Dependency versions in `requirements.txt` now have upper bounds instead of floor-only pins.

### Fixed
- Editing a non-Base-Color channel on the Seamless tab silently processed Base Color data instead of the active channel.
- The full-resolution settle timer could get permanently cancelled after a slider drag.
- Switching to an ungenerated material channel silently fell back to Base Color instead of generating it on demand.
- A crash when Delight was active while editing a non-Base-Color channel.
- A crash in the 3D viewport's Split view mode from an uninitialized split ratio.
- Contrast Recovery, Color Preservation, and Edge Consistency did nothing when adjusted alone in the Delight panel.
- Detail Preservation only worked if Shadow/Highlight/AO Removal were also raised.
- Overlap and Splat synthesis didn't normalize uint8 material-channel input to their expected float32 contract.
- A full-resolution processing thread was never released after each edit settled, orphaning a full-resolution image copy in memory for the app's lifetime on every change.
- The shared result cache had no locking; per-channel PBR caching turned a latent race into one two real call sites (a background material-map recompute, a synchronous export) could actually hit.
- Chunked processing of large textures could crash on a trailing or leading tile smaller than the minimum tile size.
- The per-channel PBR cache's capacity wasn't rescaled when caching moved from one entry per image to up to five, silently shrinking effective capacity roughly 5x.
- `NormalGenerator.process()` could silently divide its input image in place, corrupting the live image when switching to a not-yet-generated material channel.
- `hash_image()` rebuilt its random sampling state on every call, costing ~18-20x more than necessary on a path hit every live-preview tick.
- The result cache had no memory ceiling — only an entry-count limit — so a session working with large textures could grow without bound.
- An exception raised while starting a processing job (before the worker thread launched) was silently swallowed, leaving the progress indicator stuck on "Processing..." with no visible error.
- The rough (stage-1) live preview could stutter on large source images, since its 0.125x scale factor alone didn't bound the work for very large sources.
- The GPU/CPU/RAM status bar polled `nvidia-smi` synchronously on the GUI thread every 1.5 seconds for the whole session.
- The Shortcuts and Credits dialogs were never released after closing, and Credits re-decoded its artwork from disk on every open.
- A cold-cache first launch where JIT warmup outlasted the splash animation could let application shutdown destroy a still-running background thread.
- Documentation referenced a "Production Export Pipeline" dialog and renderer presets that had already been removed, including a `Ctrl+Shift+E` shortcut that was never actually bound to anything.

### Removed
- The Production Export Pipelines dialog and renderer-preset system; export now uses the existing Format/Save-mode panel directly.
- `ao_generator.py`, superseded by the reworked AO generation.
- The Rust extension (`seams_core`) and its build/CI steps — none of its exported functions had any remaining caller.
- Several modules with no callers anywhere in the app: `inpainting.py`, `edge_blending.py`, `edge_blending_jit.py`, and a dead `SystemMonitorWidget` class.

## [3.1.0] - 2026-06-28

### Fixed
- Rust gradient computation's radius parameter typing.
- Release tag title formatting.

## [3.0.0] - 2026-06-20

### Added
- Advanced normal map generation and general UI enhancements.
- A full documentation site covering all SEAMS features.
