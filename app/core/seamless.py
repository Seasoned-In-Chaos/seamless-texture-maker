"""
Main seamless texture generation algorithm.
Orchestrates offset mapping, edge blending, and inpainting.
Optimized with JIT compilation and caching for real-time performance.
"""
import numpy as np
import cv2
from .offset_mapping import offset_image, reverse_offset, create_cross_mask
from .edge_blending import blend_seams, create_blend_mask
from .edge_blending_jit import blend_seams_fast
from .materialize_methods import synthesis_overlap, synthesis_splat
from .inpainting import smart_seam_inpaint
from .delighting import delight_image
from .gpu_utils import GPUAccelerator, is_cuda_available
from .cache import ResultCache, hash_image

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
        
        # Default parameters
        self.method = 'overlap'  # overlap, splat
        
        # Standard params
        # Standard params
        self.blend_strength = 0.5
        self.seam_smoothness = 1.0 # Fixed to 1.0 (Linear Feather)
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
        self.delight_strength = 0.0
        self.flatness = 0.0
    
    def set_parameters(self, **kwargs):
        """Update processing parameters."""
        # Handle flattened params (direct set)
        for key, value in kwargs.items():
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
            pre = kwargs['preprocessing']
            if 'delight' in pre: self.delight_strength = pre['delight']
            if 'flatness' in pre: self.flatness = pre['flatness']

        # Handle old saved 'standard' method: fall back to 'overlap'
        if self.method == 'standard':
            self.method = 'overlap'
    
    def load_image(self, image):
        """
        Set the input image for processing.
        
        Args:
            image: numpy array (BGR format) or path string
        """
        if isinstance(image, str):
            self._original_image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        else:
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
    
    def process(self, image=None, preview=False, params=None):
        """
        Process the image to create a seamless texture with caching.
        
        Args:
            image: Optional input image. If None, uses previously loaded image.
            preview (bool): If True, process at lower resolution for speed.
            params (dict): Optional parameter overrides.
        
        Returns:
            Processed seamless texture
        """
        if params:
            self.set_parameters(**params)
            
        if image is not None:
            self.load_image(image)
        
        if self._original_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Check cache first (for preview mode)
        if preview and self._image_hash:
            cache_params = self._get_cache_params()
            cached_result = self._cache.get(cache_params, self._image_hash)
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
        
        # Apply delighting/flattening
        if self.delight_strength > 0 or self.flatness > 0:
            img = delight_image(img, strength=self.delight_strength, flatness=self.flatness)
        
        # Store for UI display
        self._delighted_image = img.copy()
        
        # Choose method
        if self.method == 'splat':
            result = self._process_splat(img)
        else:  # overlap (default)
            result = self._process_overlap(img)
        
        # Cache preview results
        if preview and self._image_hash:
            cache_params = self._get_cache_params()
            self._cache.set(cache_params, result, self._image_hash)
        
        return result
    
    def _get_cache_params(self):
        """Get current parameters for cache key."""
        return {
            'method': self.method,
            'overlap_x': round(self.overlap_x, 3),
            'overlap_y': round(self.overlap_y, 3),
            'edge_falloff': round(self.edge_falloff, 3),
            'splat_scale': round(self.splat_scale, 2),
            'splat_rotation': int(self.splat_rotation),
            'splat_random_rotation': round(self.splat_random_rotation, 3),
            'splat_wobble': round(self.splat_wobble, 3),
            'splat_randomize': int(self.splat_randomize),
            'delight_strength': round(self.delight_strength, 3),
            'flatness': round(self.flatness, 3)
        }
            
    def _process_overlap(self, img):
        """Process using Overlap method."""
        result = synthesis_overlap(
            img,
            overlap_x=self.overlap_x,
            overlap_y=self.overlap_y,
            falloff=self.edge_falloff
        )
        self._processed_image = result
        return result
        
    def _process_splat(self, img):
        """Process using Splat method with patch caching."""
        h, w = img.shape[:2]

        # KEY OPTIMIZATION: Cache rotated patches.
        # Only re-generate patches if appearance-affecting params change.
        # Coordinate layout is re-computed each call (fast, vectorized).
        cache_key = (
            self._image_hash,
            img.shape,
            round(self.splat_scale, 2),
            int(self.splat_rotation),
            round(self.splat_random_rotation, 3),
            round(self.edge_falloff, 3)
        )

        cached_batches = self._splat_cache.get(cache_key)

        result, batches = synthesis_splat(
            img,
            new_size=(h, w),
            scale=self.splat_scale,
            rotation=self.splat_rotation,
            rand_rot=self.splat_random_rotation,
            wobble=self.splat_wobble,
            falloff=self.edge_falloff,
            cached_batches=cached_batches
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

        self._processed_image = result
        return result

    def _process_standard(self, img):
        """Process using Standard Offset+Inpaint method."""
        h, w = img.shape[:2]
        
        # Step 1: Offset the image to bring seams to center
        offset = offset_image(img, 0.5, 0.5)
        
        # Calculate blend width - LOCAL falloff only (max 10% of image)
        # fixed_width is not used here, assuming blend_strength is the primary control
        # Local falloff constraint: max 10% of image dimension
        max_blend_width = min(h, w) // 10
        blend_width = int(max_blend_width * self.blend_strength)
        
        # Seam width for inpainting (e.g., 30% of blend_width, or a fixed minimum)
        # This ensures the inpainting region is smaller than the blend region
        seam_width = max(1, int(blend_width * 0.3)) # 3% of image if blend_strength is 1.0
        
        # Step 3: Apply smart inpainting to remove seams
        # Optimization: Simplified inpaint for preview
        is_preview = (img.shape[0] < 512) # Heuristic for preview mode
        
        inpainted = smart_seam_inpaint(
            offset,
            seam_width=seam_width,
            detail_preservation=self.detail_preservation if not is_preview else 0.0,
            method='telea'
        )
        
        
        # Color matching removed - narrow local falloff should be sufficient
        
        # Step 4: Apply edge blending (Always Non-Symmetric for best quality)
        # Non-Symmetric blending with automatic width calculation
        blended = blend_seams(
            inpainted,
            blend_strength=self.blend_strength,
            smoothness=self.seam_smoothness,
            symmetric=False,
            original_image=offset
        )
        
        # Step 5: Reverse the offset to restore original positioning
        result = reverse_offset(blended, 0.5, 0.5)
        
        # Step 6: Final color/contrast preservation
        result = self._preserve_color_balance(result, self._original_image)
        
        self._processed_image = result
        return result
    
    def _preserve_color_balance(self, processed, original):
        """
        Ensure processed image maintains original color balance and contrast.
        """
        # Convert to float for processing
        proc_f = processed.astype(np.float32)
        orig_f = original.astype(np.float32)
        
        # Match mean and std for each channel
        for c in range(proc_f.shape[2] if len(proc_f.shape) == 3 else 1):
            if len(proc_f.shape) == 3:
                chan_proc = proc_f[:, :, c]
                chan_orig = orig_f[:, :, c]
            else:
                chan_proc = proc_f
                chan_orig = orig_f
            
            # Calculate statistics
            proc_mean = np.mean(chan_proc)
            proc_std = np.std(chan_proc)
            orig_mean = np.mean(chan_orig)
            orig_std = np.std(chan_orig)
            
            # Normalize and rescale
            if proc_std > 0:
                normalized = (chan_proc - proc_mean) / proc_std
                rescaled = normalized * orig_std + orig_mean
                
                if len(proc_f.shape) == 3:
                    proc_f[:, :, c] = rescaled
                else:
                    proc_f = rescaled
        
        # Clip and convert back to original dtype
        result = np.clip(proc_f, 0, 255).astype(original.dtype)
        return result
    
    def _match_inpaint_colors(self, inpainted, original, seam_width):
        """
        Match the color/brightness of inpainted seam areas to the surrounding original texture.
        This eliminates visible bands caused by inpainting color mismatch.
        """
        h, w = inpainted.shape[:2]
        result = inpainted.copy().astype(np.float32)
        orig = original.astype(np.float32)
        
        # Create mask for the inpainted cross-shaped region
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        
        # Horizontal seam
        y_start = max(0, cy - seam_width)
        y_end = min(h, cy + seam_width)
        mask[y_start:y_end, :] = 255
        
        # Vertical seam
        x_start = max(0, cx - seam_width)
        x_end = min(w, cx + seam_width)
        mask[:, x_start:x_end] = 255
        
        # For each color channel, match histogram of inpainted area to original
        if len(result.shape) == 3:
            for c in range(result.shape[2]):
                # Get pixels in seam area
                seam_pixels = result[:, :, c][mask > 0]
                orig_pixels = orig[:, :, c][mask == 0]  # Sample from NON-seam area
                
                if len(seam_pixels) > 0 and len(orig_pixels) > 100:
                    # Calculate statistics
                    seam_mean = np.mean(seam_pixels)
                    seam_std = np.std(seam_pixels)
                    orig_mean = np.mean(orig_pixels)
                    orig_std = np.std(orig_pixels)
                    
                    # Color correct the seam area to match original
                    if seam_std > 0:
                        normalized = (result[:, :, c] - seam_mean) / seam_std
                        result[:, :, c] = normalized * orig_std + orig_mean
        else:
            # Grayscale
            seam_pixels = result[mask > 0]
            orig_pixels = orig[mask == 0]
            
            if len(seam_pixels) > 0 and len(orig_pixels) > 100:
                seam_mean = np.mean(seam_pixels)
                seam_std = np.std(seam_pixels)
                orig_mean = np.mean(orig_pixels)
                orig_std = np.std(orig_pixels)
                
                if seam_std > 0:
                    normalized = (result - seam_mean) / seam_std
                    result = normalized * orig_std + orig_mean
        
        return np.clip(result, 0, 255).astype(inpainted.dtype)
    
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
