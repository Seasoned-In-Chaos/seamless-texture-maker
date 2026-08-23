"""Seamless texture generation algorithms and processing orchestration."""
import numpy as np
import cv2
from .offset_mapping import offset_image, reverse_offset
from .materialize_methods import synthesis_overlap, synthesis_splat
from .delighting import delight_image
from .gpu_utils import is_cuda_available
from .cache import ResultCache, hash_image, make_pipeline_key
from .exceptions import ProcessingError, ImageLoadError

class SeamlessProcessor:
    """
    Main processor for creating seamless textures.
    """
    
    def __init__(self):
        self.use_gpu = is_cuda_available()
        self._original_image = None
        self._preview_image = None
        self._processed_image = None
        self._delighted_image = None
        self._image_hash = None
        
        # Performance optimizations
        self._cache = ResultCache(max_size=50)
        self._splat_cache = {} # Cache for rotated patches (huge speedup)
        self.use_jit = True  # Use JIT-compiled functions
        
        # Default parameters.  The two deterministic methods deliberately do
        # not expose extra controls: their behavior is fixed and repeatable.
        self.method = 'overlap'  # overlap, splat, offset_crossfade, mirror
        
        # Offset + cross-fade params
        self.blend_strength = 0.5
        self.seam_smoothness = 1.0
        self.detail_preservation = 0.75
        self.symmetric_blending = True
        
        # Overlap/Splat params (some shared)
        self.overlap_x = 0.2
        self.overlap_y = 0.2
        self.edge_falloff = 0.1
        self.splat_scale = 1.0
        self.splat_rotation = 0
        self.splat_random_rotation = 0
        self.splat_wobble = 0.2
        self.splat_randomize = 0

        # Delighting/Flattening params
        self.preprocessing_params = {}
    
    def set_parameters(self, **kwargs):
        """Update processing parameters."""
        if 'method' in kwargs:
            method = str(kwargs['method']).strip().lower().replace('-', '_').replace(' ', '_')
            method_aliases = {
                'standard': 'offset_crossfade',
                'offset': 'offset_crossfade',
                'offset_blend': 'offset_crossfade',
                'crossfade': 'offset_crossfade',
                'offset_cross_fade': 'offset_crossfade',
                'mirror_tiling': 'mirror',
                'mirror_tile': 'mirror',
                'mirror_tiles': 'mirror',
            }
            self.method = method_aliases.get(method, method)

        # Handle flattened params (direct set)
        for key, value in kwargs.items():
            if key == 'method':
                continue
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Handle nested keys from GUI (controls.py structure)
        if 'standard' in kwargs:
            std = kwargs['standard']
            if 'blend' in std: self.blend_strength = std['blend']
            # if 'smoothness' in std: self.seam_smoothness = std['smoothness'] # Removed
            
        if 'overlap' in kwargs:
            ov = kwargs['overlap']
            if 'x' in ov: self.overlap_x = ov['x']
            if 'y' in ov: self.overlap_y = ov['y']
            if 'falloff' in ov: self.edge_falloff = ov['falloff']
            
        if 'splat' in kwargs:
            sp = kwargs['splat']
            if 'scale' in sp: self.splat_scale = sp['scale']
            if 'rotation' in sp: self.splat_rotation = sp['rotation']
            if 'rand_rot' in sp: self.splat_random_rotation = sp['rand_rot']
            if 'wobble' in sp: self.splat_wobble = sp['wobble']
            if 'randomize' in sp: self.splat_randomize = sp['randomize']
            if 'falloff' in sp: self.edge_falloff = sp['falloff']

        if 'preprocessing' in kwargs:
            self.preprocessing_params = kwargs['preprocessing']

        # Keep direct/API callers safe as well as the GUI.  A zero-sized
        # splat is an empty patch and produces an apparent flat result.
        self.splat_scale = max(0.25, float(self.splat_scale))
        self.splat_random_rotation = float(np.clip(self.splat_random_rotation, 0.0, 1.0))
        self.splat_wobble = float(np.clip(self.splat_wobble, 0.0, 1.0))
        self.edge_falloff = float(np.clip(self.edge_falloff, 0.0, 1.0))

    def load_image(self, image):
        """
        Set the input image for processing.
        
        Args:
            image: numpy array (BGR format) or path string
            
        Raises:
            ImageLoadError: If the image is invalid.
        """
        if isinstance(image, str):
            self._original_image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            if self._original_image is None:
                raise ImageLoadError(f"Failed to read image: {image}")
        else:
            # Input validation
            if not isinstance(image, np.ndarray):
                raise ImageLoadError(f"Expected ndarray, got {type(image).__name__}")
            if image.size == 0:
                raise ImageLoadError("Image is empty")
            h, w = image.shape[:2]
            if h < 64 or w < 64:
                raise ImageLoadError(f"Image too small: {w}x{h} (minimum 64x64)")
            if h > 8192 or w > 8192:
                raise ImageLoadError(f"Image too large: {w}x{h} (maximum 8192x8192)")
            # Check for NaN/Inf
            if np.issubdtype(image.dtype, np.floating):
                if not np.all(np.isfinite(image)):
                    import logging
                    logging.getLogger("seams").warning("Image contains NaN/Inf, replacing with 0")
                    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            self._original_image = image.copy()
        
        self._processed_image = None
        self._delighted_image = None
        self._splat_cache = {} # Clear patch cache

        # Cache preview image for live updates (smaller for maximum speed)
        if self._original_image is not None:
             # Hash image for cache key
             self._image_hash = hash_image(self._original_image)
             
             h, w = self._original_image.shape[:2]
             max_dim = 600  # Higher resolution for sharper live previews
             if max(h, w) > max_dim:
                 scale = max_dim / max(h, w)
                 new_w = int(w * scale)
                 new_h = int(h * scale)
                 self._preview_image = cv2.resize(self._original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
             else:
                 self._preview_image = self._original_image.copy()
    
    def process(self, image=None, preview=False, params=None, use_cache: bool = True,
                chunked: bool = True):
        """
        Process the image to create a seamless texture with caching.

        Args:
            image: Optional input image. If None, uses previously loaded image.
            preview (bool): If True, process at lower resolution for speed.
            params (dict): Optional parameter overrides. A ``delight_only``
                key (bool) returns the delighted image before the seamless
                method runs, so the Delight step's live preview matches what
                "Apply Delight" actually commits instead of being obscured
                by tiling/blend artifacts from the seamless method.
            use_cache (bool): If True, check/store result in cache (default True).
            chunked (bool): If True, auto-select chunked path for large images.

        Returns:
            Processed seamless texture
        """
        if params:
            self.set_parameters(**params)

        delight_only = bool((params or {}).get('delight_only', False))

        if image is not None:
            self.load_image(image)

        if self._original_image is None:
            raise ProcessingError("No image loaded. Call load_image() first.")

        # Auto-select chunked path for large images
        h, w = self._original_image.shape[:2]
        if self.method == 'mirror' and max(h, w) > 4096:
            raise ProcessingError(
                "Mirror Tiling doubles both dimensions; use an input up to 4096px "
                "or choose another seamless method."
            )
        # A mirror tile intentionally produces a 2x2 canvas, so splitting the
        # source into independent chunks would destroy the exact reflection
        # at its four outer edges.  The offset cross-fade also needs one
        # global center seam.  Keep both methods on their full-image path.
        if (chunked and not preview and max(h, w) > 2048
                and self.method not in {'mirror', 'offset_crossfade'} and not delight_only):
            return self.run_pipeline_chunked(self._original_image, chunk_size=1024, overlap=64)

        # Check cache (both preview and full-res when use_cache is True).
        # Skipped for delight_only requests: the cache key doesn't encode
        # this flag, so a prior full-pipeline result could otherwise be
        # returned in place of the plain delighted image (or vice versa).
        if use_cache and self._image_hash and not delight_only:
            cache_key = make_pipeline_key(
                self._preview_image if preview else self._original_image,
                self._get_cache_params(),
            )
            cached_result = self._cache.get_pipeline(cache_key)
            if cached_result is not None:
                return cached_result

        # Prepare source image
        if preview:
            if self._preview_image is None:
                # Fallback if somehow not loaded
                self.load_image(self._original_image)
            img = self._preview_image.copy()
        else:
            img = self._original_image.copy()

        # Apply delighting/flattening. Delight corrects baked lighting in a
        # photographic Base Color image (always float32 in this pipeline);
        # it has no meaning for a generated material channel like Normal
        # (uint16) or Roughness/AO/Displacement/Opacity (uint8), and
        # delight_image requires float32 input, so running a material
        # channel through it here would raise instead of silently doing
        # something meaningless.
        if (img.dtype == np.float32 and self.preprocessing_params
                and any(v > 0 for v in self.preprocessing_params.values())):
            delight_kwargs = self.preprocessing_params.copy()
            delight_kwargs["strength"] = delight_kwargs.pop("delight", 0.0)
            img = delight_image(img, **delight_kwargs)

        # Store for UI display
        self._delighted_image = img.copy()

        if delight_only:
            return img

        # Choose method
        if self.method == 'splat':
            result = self._process_splat(img)
        elif self.method == 'offset_crossfade':
            result = self._process_offset_crossfade(img)
        elif self.method == 'mirror':
            result = self._process_mirror_tiling(img)
        else:  # overlap (default)
            result = self._process_overlap(img)
        
        # Cache result (both preview and full-res when use_cache is True)
        if use_cache and self._image_hash:
            cache_key = make_pipeline_key(
                self._preview_image if preview else self._original_image,
                self._get_cache_params(),
            )
            self._cache.set_pipeline(cache_key, result)
        
        return result
    
    def _get_cache_params(self):
        """Get current parameters for cache key."""
        params = {
            'method': self.method,
            'blend_strength': round(self.blend_strength, 3),
            'seam_smoothness': round(self.seam_smoothness, 3),
            'overlap_x': round(self.overlap_x, 3),
            'overlap_y': round(self.overlap_y, 3),
            'edge_falloff': round(self.edge_falloff, 3),
            'splat_scale': round(self.splat_scale, 2),
            'splat_rotation': int(self.splat_rotation),
            'splat_random_rotation': round(self.splat_random_rotation, 3),
            'splat_wobble': round(self.splat_wobble, 3),
            'splat_randomize': int(self.splat_randomize),
        }
        if self.preprocessing_params:
            for k, v in self.preprocessing_params.items():
                params[f'pre_{k}'] = round(float(v), 3) if isinstance(v, (int, float)) else v
        return params
            
    def _process_overlap(self, img):
        """Process using Overlap method."""
        work, value_range = self._materialize_input(img)
        result = synthesis_overlap(
            work,
            overlap_x=self.overlap_x,
            overlap_y=self.overlap_y,
            falloff=self.edge_falloff
        )
        result = self._materialize_output(result, img.dtype, value_range)
        self._processed_image = result
        return result
        
    def _process_splat(self, img):
        """Process using Splat method with patch caching."""
        h, w = img.shape[:2]
        work, value_range = self._materialize_input(img)

        # KEY OPTIMIZATION: Cache rotated patches.
        # Only re-generate patches if appearance-affecting params change.
        # Coordinate layout is re-computed each call (fast, vectorized).
        # Wobble and the seed shape the patch masks, so they belong in the
        # key too -- otherwise a stale bank is reused and those sliders look
        # dead until some other parameter changes.
        cache_key = (
            "sampled-splat-v2",
            self._image_hash,
            img.shape,
            round(self.splat_scale, 3),
            int(self.splat_rotation),
            round(self.splat_random_rotation, 3),
            round(self.splat_wobble, 3),
            round(self.edge_falloff, 3),
            int(self.splat_randomize),
        )

        cached_batches = self._splat_cache.get(cache_key)

        result, batches = synthesis_splat(
            work,
            new_size=(h, w),
            scale=self.splat_scale,
            rotation=self.splat_rotation,
            rand_rot=self.splat_random_rotation,
            wobble=self.splat_wobble,
            falloff=self.edge_falloff,
            cached_batches=cached_batches,
            seed=int(self.splat_randomize),
        )

        # Store in cache if newly generated
        if cached_batches is None:
            self._splat_cache[cache_key] = batches
            # Simple LRU eviction
            if len(self._splat_cache) > 8:
                try:
                    first_key = next(iter(self._splat_cache))
                    del self._splat_cache[first_key]
                except (StopIteration, RuntimeError):
                    pass

        self._processed_image = self._materialize_output(result, img.dtype, value_range)
        return self._processed_image

    @staticmethod
    def _materialize_input(image):
        """Convert map data to the float32/0-255 contract of Materialize methods."""
        if np.issubdtype(image.dtype, np.integer):
            value_range = float(np.iinfo(image.dtype).max)
        else:
            observed = float(np.nanmax(image)) if image.size else 255.0
            value_range = 1.0 if observed <= 1.0 else 255.0
        work = image.astype(np.float32, copy=False)
        if value_range != 255.0:
            work = work * (255.0 / value_range)
        return np.ascontiguousarray(work), value_range

    @staticmethod
    def _materialize_output(result, dtype, value_range):
        restored = result.astype(np.float32, copy=False)
        if value_range != 255.0:
            restored = restored * (value_range / 255.0)
        if np.issubdtype(dtype, np.integer):
            restored = np.clip(restored, 0, np.iinfo(dtype).max)
        return restored.astype(dtype, copy=False)

    def _process_standard(self, img):
        """Backward-compatible name for the offset cross-fade method."""
        return self._process_offset_crossfade(img)

    def _process_offset_crossfade(self, img):
        """Create a tile with a 50% offset and linear center-seam fades.

        The offset moves the source's four original borders to the center
        axes.  At each center seam, matching pixels on either side are
        blended into both sides with a linear alpha feather.  Reversing the
        offset then places the equalized seam pixels on the outside borders,
        making the result repeat cleanly without an inpainting pass.
        """
        h, w = img.shape[:2]
        offset = offset_image(img, 0.5, 0.5)

        # Use a local feather: large enough to hide hard source borders but
        # bounded so the method does not wash out the whole texture.
        strength = float(np.clip(self.blend_strength, 0.1, 1.0))
        radius = max(1, int(round(min(h, w) * 0.16 * strength)))
        blended = self._linear_crossfade_center_seams(offset, radius)
        result = reverse_offset(blended, 0.5, 0.5)

        self._processed_image = result.astype(img.dtype, copy=False)
        return self._processed_image

    @staticmethod
    def _linear_crossfade_center_seams(image, radius):
        """Feather both center seam axes using symmetric linear weights."""
        result = image.astype(np.float32, copy=True)
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        radius = max(1, int(radius))

        # At the seam the two sides are averaged equally.  The influence
        # decreases linearly to zero at the outer edge of the feather.
        source = result.copy()
        for distance in range(radius):
            weight = 0.5 * (1.0 - (distance / radius))
            left, right = cx - 1 - distance, cx + distance
            if left >= 0 and right < w:
                left_value = source[:, left].copy()
                right_value = source[:, right].copy()
                result[:, left] = left_value * (1.0 - weight) + right_value * weight
                result[:, right] = right_value * (1.0 - weight) + left_value * weight

        # Keep the horizontal seam work while processing the vertical seam.
        # This makes the center crossing obey both equalized boundaries.
        source = result.copy()
        for distance in range(radius):
            weight = 0.5 * (1.0 - (distance / radius))
            top, bottom = cy - 1 - distance, cy + distance
            if top >= 0 and bottom < h:
                top_value = source[top, :].copy()
                bottom_value = source[bottom, :].copy()
                result[top, :] = top_value * (1.0 - weight) + bottom_value * weight
                result[bottom, :] = bottom_value * (1.0 - weight) + top_value * weight

        if np.issubdtype(image.dtype, np.integer):
            upper = np.iinfo(image.dtype).max
        else:
            upper = 1.0 if np.nanmax(image) <= 1.0 else 255.0
        return np.clip(result, 0, upper).astype(image.dtype, copy=False)

    def _process_mirror_tiling(self, img):
        """Build an exact 2x2 mirrored tile with seamless outer borders."""
        horizontal = np.concatenate((img, np.flip(img, axis=1)), axis=1)
        result = np.concatenate((horizontal, np.flip(horizontal, axis=0)), axis=0)
        self._processed_image = result
        return result
    
    def get_preview(self, max_size=1024):
        """
        Get a resized preview for real-time display.
        
        Args:
            max_size: Maximum dimension for preview
        
        Returns:
            Tuple of (original_preview, processed_preview)
        """
        if self._original_image is None:
            return None, None
        
        h, w = self._original_image.shape[:2]
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            orig_preview = cv2.resize(self._original_image, (new_w, new_h), 
                                       interpolation=cv2.INTER_AREA)
            
            if self._processed_image is not None:
                proc_preview = cv2.resize(self._processed_image, (new_w, new_h),
                                          interpolation=cv2.INTER_AREA)
            else:
                proc_preview = None
        else:
            orig_preview = self._original_image.copy()
            proc_preview = self._processed_image.copy() if self._processed_image is not None else None
        
        return orig_preview, proc_preview
    
    def get_tiled_preview(self, image=None, tiles=2, max_size=1024):
        """
        Create a tiled preview to verify seamlessness.
        
        Args:
            image: Image to tile (uses processed if None)
            tiles: Number of tiles in each direction
            max_size: Maximum total preview size
        
        Returns:
            Tiled preview image
        """
        if image is None:
            image = self._processed_image
        
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # Resize for preview if needed
        tile_size = max_size // tiles
        if max(h, w) > tile_size:
            scale = tile_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            tile = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            tile = image
            new_w, new_h = w, h
        
        # Create tiled image
        tiled = np.tile(tile, (tiles, tiles, 1) if len(tile.shape) == 3 else (tiles, tiles))
        
        return tiled
    
    @property
    def original_image(self):
        return self._original_image
    
    @property
    def processed_image(self):
        return self._processed_image

    def set_processed_image(self, image):
        """Synchronize an externally processed image back into this processor."""
        self._processed_image = None if image is None else image.copy()
    
    @property
    def delighted_image(self):
        return self._delighted_image
    
    @property
    def gpu_available(self):
        return self.use_gpu

    def run_pipeline_chunked(self, img: 'np.ndarray', chunk_size: int = 1024,
                              overlap: int = 64, **kwargs) -> 'np.ndarray':
        """Process large images in overlapping tiles to avoid OOM.

        Splits the image into (chunk_size x chunk_size) tiles with *overlap*
        pixels of overlap on each edge, processes each tile through the
        standard pipeline, then reassembles with linear gradient blending
        over the overlap zone.

        Falls back to :meth:`process` directly for images <= 2048px.
        """
        import logging
        import time
        logger = logging.getLogger("seams.chunked")

        h, w = img.shape[:2]

        if max(h, w) <= 2048:
            return self.process(image=img, preview=False, chunked=False, **kwargs)

        t0 = time.perf_counter()
        result = img.copy()
        c = img.shape[2] if img.ndim == 3 else 1

        # Compute tile positions
        tiles_y = list(range(0, h, chunk_size))
        tiles_x = list(range(0, w, chunk_size))

        total_tiles = len(tiles_y) * len(tiles_x)
        logger.info("Chunked processing: %dx%d → %d tiles (chunk=%d, overlap=%d)",
                     w, h, total_tiles, chunk_size, overlap)

        tile_idx = 0
        for y0 in tiles_y:
            for x0 in tiles_x:
                tile_idx += 1
                y1 = min(y0 + chunk_size + overlap, h)
                x1 = min(x0 + chunk_size + overlap, w)
                # Also expand backwards for overlap
                y_start = max(0, y0 - overlap)
                x_start = max(0, x0 - overlap)

                tile = img[y_start:y1, x_start:x1].copy()

                t_tile = time.perf_counter()
                tile_processor = SeamlessProcessor()
                tile_processor.set_parameters(**self._get_cache_params())
                tile_processor.load_image(tile)
                processed = tile_processor.process(preview=False, chunked=False)
                tile_ms = (time.perf_counter() - t_tile) * 1000.0
                logger.debug("  tile %d/%d: %dms", tile_idx, total_tiles, int(tile_ms))

                # Compute blend weights for overlap regions
                th, tw = processed.shape[:2]
                fade_y = np.linspace(0, 1, min(overlap, th)).astype(np.float32)
                fade_x = np.linspace(0, 1, min(overlap, tw)).astype(np.float32)

                if y_start > 0:
                    fade = fade_y[:, np.newaxis, np.newaxis] if img.ndim == 3 else fade_y[:, np.newaxis]
                    fh = len(fade_y)
                    processed[:fh] = (processed[:fh] * fade +
                                      result[y_start:y_start+fh, x_start:x1] * (1 - fade))

                if x_start > 0:
                    fade = fade_x[np.newaxis, :, np.newaxis] if img.ndim == 3 else fade_x[np.newaxis, :]
                    fw = len(fade_x)
                    processed[:, :fw] = (processed[:, :fw] * fade +
                                         result[y_start:y1, x_start:x_start+fw] * (1 - fade))

                # Write tile into result
                result[y_start:y1, x_start:x1] = processed

        total_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("Chunked processing done: %.0f ms, %d tiles", total_ms, total_tiles)
        return result


def make_seamless(image, blend_strength=0.5, seam_smoothness=0.5, 
                  detail_preservation=0.75, symmetric=True):
    """
    Convenience function to make an image seamless.
    
    Args:
        image: Input image (numpy array or path)
        blend_strength: Edge blend strength (0.0-1.0)
        seam_smoothness: Seam smoothness (0.0-1.0)
        detail_preservation: Detail preservation (0.0-1.0)
        symmetric: Use symmetric blending
    
    Returns:
        Seamless texture
    """
    processor = SeamlessProcessor()
    processor.set_parameters(
        blend_strength=blend_strength,
        seam_smoothness=seam_smoothness,
        detail_preservation=detail_preservation,
        symmetric_blending=symmetric
    )
    processor.load_image(image)
    return processor.process()


def precompile_jit_functions():
    """Trigger Numba compilation for all seamless method paths.

    Runs on a background QThread (PrecompileThread in splash_screen.py)
    while the splash animation plays, so the first user action is
    stall-free. Each function is called once with a minimal 64x64 array.
    Returns a dict mapping each component name to its elapsed compile
    time in ms.
    """
    import logging
    import time

    logger = logging.getLogger("seams.precompile")
    timings = {}

    tiny_3ch = np.zeros((64, 64, 3), dtype=np.float32)
    tiny_weights = np.zeros(8, dtype=np.float32)

    # -- Overlap Blend JIT (edge blending) --------------------------------
    try:
        from .edge_blending_jit import (
            blend_seam_horizontal_jit,
            blend_seam_vertical_jit,
            calculate_blend_weights,
        )

        t0 = time.perf_counter()
        calculate_blend_weights(8, 0.5)
        result_h = tiny_3ch.copy()
        blend_seam_horizontal_jit(result_h, tiny_3ch, 32, 4, tiny_weights)
        result_v = tiny_3ch.copy()
        blend_seam_vertical_jit(result_v, tiny_3ch, 32, 4, tiny_weights)
        timings["edge_blending_jit"] = (time.perf_counter() - t0) * 1000.0
        logger.info("precompile edge_blending_jit: %.1f ms", timings["edge_blending_jit"])
    except Exception as exc:
        logger.warning("precompile edge_blending_jit failed: %s", exc)

    # -- Splat Synthesis JIT ------------------------------------------------
    try:
        from .materialize_methods_jit import splat_accumulate_jit, splat_resolve_jit

        patches = np.stack([tiny_3ch.copy()])
        masks = np.ones((1, 64, 64), dtype=np.float32)
        coords = np.array([[0, 0]], dtype=np.int32)
        indices = np.array([0], dtype=np.int32)
        accum = np.zeros((64, 64, 3), dtype=np.float32)
        weight = np.zeros((64, 64), dtype=np.float32)

        t0 = time.perf_counter()
        splat_accumulate_jit(accum, weight, patches, masks, coords, indices)
        splat_resolve_jit(accum, weight, tiny_3ch, np.empty_like(accum))
        timings["splat_jit"] = (time.perf_counter() - t0) * 1000.0
        logger.info("precompile splat_jit: %.1f ms", timings["splat_jit"])
    except Exception as exc:
        logger.warning("precompile splat_jit failed: %s", exc)

    # -- Mirror Tiling (2x2) and Offset + Cross-Fade -----------------------
    try:
        tiny_img = np.zeros((64, 64, 3), dtype=np.uint8)
        proc = SeamlessProcessor()
        proc.load_image(tiny_img)

        for method in ("mirror", "offset_crossfade"):
            t0 = time.perf_counter()
            proc.set_parameters(method=method)
            proc.process(preview=True, use_cache=False)
            timings[method] = (time.perf_counter() - t0) * 1000.0
            logger.info("precompile %s: %.1f ms", method, timings[method])

    except Exception as exc:
        logger.warning("precompile seamless methods failed: %s", exc)

    total_ms = sum(timings.values())
    logger.info("precompile complete: %.1f ms total (%d components)", total_ms, len(timings))
    return timings
