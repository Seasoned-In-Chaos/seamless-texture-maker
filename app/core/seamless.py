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
        self._image_hash = None
        
        # Performance optimizations
        self._cache = ResultCache(max_size=50)
        self.use_jit = True  # Use JIT-compiled functions
        
        # Default parameters
        self.method = 'standard' # standard, overlap, splat
        
        # Standard params
        self.blend_strength = 0.5
        self.seam_smoothness = 0.5
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
    
    def set_parameters(self, **kwargs):
        """Update processing parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
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

        # Cache preview image for live updates (smaller for maximum speed)
        if self._original_image is not None:
             # Hash image for cache key
             self._image_hash = hash_image(self._original_image)
             
             h, w = self._original_image.shape[:2]
             max_dim = 256  # Reduced to 256px for ultra-fast preview
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
        
        # Choose method
        if self.method == 'overlap':
            result = self._process_overlap(img)
        elif self.method == 'splat':
            result = self._process_splat(img)
        else:
            result = self._process_standard(img)
        
        # Cache preview results
        if preview and self._image_hash:
            cache_params = self._get_cache_params()
            self._cache.set(cache_params, result, self._image_hash)
        
        return result
    
    def _get_cache_params(self):
        """Get current parameters for cache key."""
        return {
            'method': self.method,
            'blend_strength': round(self.blend_strength, 3),
            'seam_smoothness': round(self.seam_smoothness, 3),
            'detail_preservation': round(self.detail_preservation, 3),
            'overlap_x': round(self.overlap_x, 3),
            'overlap_y': round(self.overlap_y, 3),
            'edge_falloff': round(self.edge_falloff, 3),
            'splat_scale': round(self.splat_scale, 2),
            'splat_rotation': int(self.splat_rotation),
            'splat_random_rotation': round(self.splat_random_rotation, 3),
            'splat_wobble': round(self.splat_wobble, 3),
            'splat_randomize': int(self.splat_randomize)
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
        """Process using Splat method."""
        # Use random seed based on 'randomize' param
        # New size logic: use same size as input for now
        h, w = img.shape[:2]
        result = synthesis_splat(
            img,
            new_size=(h, w),
            grid_size=int(8 * self.splat_scale), # Rough heuristic
            scale=1.0, # Relative scale is tricky, keeping 1.0
            rotation=self.splat_rotation,
            rand_rot=self.splat_random_rotation,
            wobble=self.splat_wobble,
            falloff=self.edge_falloff
        )
        self._processed_image = result
        return result

    def _process_standard(self, img):
        """Process using Standard Offset+Inpaint method."""
        h, w = img.shape[:2]
        
        # Step 1: Offset the image to bring seams to center
        offset = offset_image(img, 0.5, 0.5)
        
        # Step 2: Calculate seam width based on image size and parameters
        min_dim = min(h, w)
        base_seam_width = max(10, min_dim // 20)
        seam_width = int(base_seam_width * (0.5 + self.blend_strength * 0.5))
        
        # Step 3: Apply smart inpainting to remove seams
        inpainted = smart_seam_inpaint(
            offset,
            seam_width=seam_width,
            detail_preservation=self.detail_preservation,
            method='telea'
        )
        
        # Step 4: Apply edge blending for smooth transitions (JIT-optimized)
        if self.use_jit:
            blended = blend_seams_fast(
                inpainted,
                blend_strength=self.blend_strength,
                smoothness=self.seam_smoothness
            )
        else:
            blended = blend_seams(
                inpainted,
                blend_strength=self.blend_strength,
                smoothness=self.seam_smoothness,
                symmetric=self.symmetric_blending
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
