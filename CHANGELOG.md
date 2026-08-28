# Changelog

All notable changes to SEAMS are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
