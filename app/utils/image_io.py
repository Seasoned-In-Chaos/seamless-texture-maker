"""
Image I/O utilities with DPI/resolution preservation.
"""
import os
import re
import tempfile
import time
import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
from PIL import Image

from .app_logging import get_logger, log_exception


logger = get_logger(__name__)
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
SUPPORTED_SAVE_FORMATS = {
    '.png': 'png',
    '.jpg': 'jpg',
    '.jpeg': 'jpg',
    '.tif': 'tiff',
    '.tiff': 'tiff',
    '.tga': 'tga',
    '.exr': 'exr',
}


class ImageMetadata:
    """Container for image metadata."""
    
    def __init__(self):
        self.width = 0
        self.height = 0
        self.dpi = (72, 72)  # Default DPI
        self.format = None
        self.bit_depth = 8
        self.channels = 3
        self.filepath = None


def load_image(filepath):
    """
    Load an image with metadata preservation.
    
    Args:
        filepath: Path to image file
    
    Returns:
        Tuple of (image as numpy array BGR, ImageMetadata)
    """
    filepath = os.path.abspath(os.path.expanduser(str(filepath)))
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")
    if not os.path.isfile(filepath):
        raise IOError(f"Path is not a file: {filepath}")
    
    # Get metadata using PIL
    metadata = ImageMetadata()
    metadata.filepath = filepath
    
    try:
        with Image.open(filepath) as pil_img:
            # Get DPI if available
            if 'dpi' in pil_img.info:
                metadata.dpi = pil_img.info['dpi']
            elif hasattr(pil_img, 'info') and 'resolution' in pil_img.info:
                metadata.dpi = pil_img.info['resolution']
            
            metadata.width, metadata.height = pil_img.size
            metadata.format = pil_img.format
            
            # Determine bit depth and channels
            mode = pil_img.mode
            if mode == 'L':
                metadata.channels = 1
                metadata.bit_depth = 8
            elif mode == 'LA':
                metadata.channels = 2
                metadata.bit_depth = 8
            elif mode == 'RGB':
                metadata.channels = 3
                metadata.bit_depth = 8
            elif mode == 'RGBA':
                metadata.channels = 4
                metadata.bit_depth = 8
            elif mode == 'I;16':
                metadata.channels = 1
                metadata.bit_depth = 16
            elif mode == 'I':
                metadata.channels = 1
                metadata.bit_depth = 32
    except Exception as exc:
        logger.debug("PIL metadata read skipped for %s: %s", filepath, exc)
    
    # Load image with OpenCV for processing
    # Use imdecode to support unicode paths on Windows
    try:
        # Read file as byte stream
        with open(filepath, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
    except Exception as e:
        log_exception(logger, f"Failed to read image file {filepath}", e)
        raise IOError(f"Failed to read image file: {e}")
    
    if image is None:
        raise IOError(f"Failed to decode image: {filepath}")

    if metadata.width == 0 or metadata.height == 0:
        metadata.height, metadata.width = image.shape[:2]
        metadata.channels = 1 if image.ndim == 2 else image.shape[2]
        metadata.bit_depth = 16 if image.dtype == np.uint16 else 32 if image.dtype == np.float32 else 8
        metadata.format = os.path.splitext(filepath)[1].lstrip(".").upper()
    
    return image, metadata


def save_image(image, filepath, metadata=None, format=None, quality=95):
    """
    Save an image with metadata preservation.
    
    Args:
        image: Image as numpy array
        filepath: Output path
        metadata: Optional ImageMetadata for DPI preservation
        format: Output format ('png', 'jpg', 'tiff'), inferred from extension if None
        quality: JPEG quality (1-100)
    
    Returns:
        True if successful
    """
    filepath = os.path.abspath(os.path.expanduser(str(filepath)))
    if image is None:
        raise ValueError("No image data is available to save.")
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image data is invalid or empty.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unsupported channel count: {image.shape[2]}")

    # Determine format from extension if not specified
    ext = os.path.splitext(filepath)[1].lower()
    if format is None:
        format = SUPPORTED_SAVE_FORMATS.get(ext)
    else:
        format = format.lower()

    if not format:
        raise ValueError(f"Unsupported output format: {ext or '(none)'}")
    if format not in set(SUPPORTED_SAVE_FORMATS.values()):
        raise ValueError(f"Unsupported output format: {format}")
    
    # Ensure directory exists
    directory = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'
    ensure_writable_directory(directory)

    image = _coerce_image_for_save(image)
    
    if format == 'exr':
        exr_image = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
        try:
            if cv2.imwrite(filepath, exr_image):
                return True
        except Exception as exc:
            log_exception(logger, f"Failed to write EXR {filepath}", exc)
            raise IOError("EXR export failed. OpenCV may not have OpenEXR support enabled; choose PNG or TIFF.") from exc
        raise IOError("EXR export failed. OpenCV did not write the output file.")

    # Convert BGR to RGB for PIL
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(image)
    
    # Prepare save options
    save_kwargs = {}
    
    # Set DPI if metadata available
    if metadata and metadata.dpi:
        save_kwargs['dpi'] = metadata.dpi
    
    # Format-specific options
    if format == 'jpg':
        save_kwargs['quality'] = quality
        save_kwargs['subsampling'] = 0  # Best quality
        # JPEG doesn't support alpha, convert if needed
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
    elif format == 'png':
        save_kwargs['compress_level'] = 6
    elif format == 'tiff':
        save_kwargs['compression'] = 'tiff_lzw'
    elif format == 'tga':
        format = 'tga'
    
    # Save atomically so interrupted writes do not leave corrupted final files.
    fd, tmp_path = tempfile.mkstemp(prefix=".seams-", suffix=os.path.splitext(filepath)[1] or f".{format}", dir=directory)
    os.close(fd)
    try:
        pil_image.save(tmp_path, format=format.upper() if format == 'tga' else None, **save_kwargs)
        os.replace(tmp_path, filepath)
    except Exception as exc:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        log_exception(logger, f"Failed to save image {filepath}", exc)
        raise
    
    return True


def get_output_path(input_path, suffix='_seamless', output_format=None):
    """
    Generate output path with suffix.
    
    Args:
        input_path: Original file path
        suffix: Suffix to add before extension
        output_format: New format (None to keep original)
    
    Returns:
        New file path
    """
    # Handle None input_path gracefully
    if input_path is None:
        # Return a sensible default
        ext = '.png'
        if output_format:
            ext_map = {
                'png': '.png',
                'jpg': '.jpg',
                'jpeg': '.jpg',
                'tiff': '.tiff',
                'tga': '.tga',
                'exr': '.exr'
            }
            ext = ext_map.get(output_format.lower(), '.png')
        return f"output{suffix}{ext}"
    
    directory = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)
    
    if output_format:
        ext_map = {
            'png': '.png',
            'jpg': '.jpg',
            'jpeg': '.jpg',
            'tiff': '.tiff',
            'tga': '.tga',
            'exr': '.exr'
        }
        ext = ext_map.get(output_format.lower(), ext)
    
    new_name = f"{name}{suffix}{ext}"
    return os.path.join(directory, new_name)


def get_file_info(filepath):
    """
    Get basic file information.
    
    Args:
        filepath: Path to file
    
    Returns:
        Dict with file info
    """
    if not os.path.exists(filepath):
        return None
    
    stat = os.stat(filepath)
    size_bytes = stat.st_size
    
    # Format size
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
    
    return {
        'path': filepath,
        'name': os.path.basename(filepath),
        'size_bytes': size_bytes,
        'size_str': size_str,
        'extension': os.path.splitext(filepath)[1].lower()
    }


def supported_formats():
    """Return list of supported image formats."""
    return ['PNG', 'JPG', 'JPEG', 'TIFF', 'TIF', 'TGA', 'EXR']


def get_format_filter():
    """Get file dialog filter string."""
    return "Images (*.png *.jpg *.jpeg *.tiff *.tif *.tga *.exr);;PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tiff *.tif);;TGA (*.tga);;EXR (*.exr)"


def sanitize_filename_component(value, fallback="Material"):
    """Return a filesystem-safe filename component."""
    cleaned = INVALID_FILENAME_CHARS.sub("_", str(value or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip(" ._")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or fallback


def ensure_writable_directory(directory):
    """Create and verify a directory can receive exported files."""
    directory = os.path.abspath(os.path.expanduser(str(directory or ".")))
    os.makedirs(directory, exist_ok=True)
    if not os.path.isdir(directory):
        raise IOError(f"Export location is not a folder: {directory}")
    _cleanup_stale_temp_files(directory)

    fd, test_path = tempfile.mkstemp(prefix=".seams-write-test-", dir=directory)
    os.close(fd)
    try:
        os.remove(test_path)
    except OSError:
        pass
    return directory


def _cleanup_stale_temp_files(directory):
    try:
        now = time.time()
        for name in os.listdir(directory):
            if not name.startswith(".seams-"):
                continue
            path = os.path.join(directory, name)
            try:
                if os.path.isfile(path) and now - os.path.getmtime(path) > 24 * 60 * 60:
                    os.remove(path)
            except OSError:
                continue
    except OSError:
        return


def _coerce_image_for_save(image):
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        finite = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        if finite.max(initial=0.0) <= 1.0:
            return np.clip(finite * 255.0, 0, 255).astype(np.uint8)
        return np.clip(finite, 0, 255).astype(np.uint8)
    if image.dtype == np.uint16:
        return (image / 257).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def load_as_float32(path: str) -> np.ndarray:
    """Load image and convert to float32 RGB in [0, 1] range.

    This is the **canonical entry point** for the processing pipeline.
    All core/ functions operate on float32 arrays; uint8 conversion
    happens only at save time via ``save_from_float32``.

    Args:
        path: Filesystem path to the image.

    Returns:
        float32 ndarray of shape (H, W, 3) with values in [0.0, 1.0].
    """
    image_bgr, _meta = load_image(path)
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0


def save_from_float32(
    arr: np.ndarray,
    path: str,
    dpi: int = 72,
) -> None:
    """Save a float32 [0, 1] RGB array to disk.

    This is the **canonical exit point** for the processing pipeline.
    Converts float32 -> uint8 and RGB -> BGR for PIL/OpenCV.

    Args:
        arr: float32 ndarray, shape (H, W, 3) or (H, W), values [0.0, 1.0].
        path: Output file path (extension determines format).
        dpi: DPI metadata to embed (default 72).
    """
    if arr.dtype == np.float32 or np.issubdtype(arr.dtype, np.floating):
        uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        uint8 = arr.astype(np.uint8)

    meta = ImageMetadata()
    meta.dpi = (dpi, dpi)

    if uint8.ndim == 3 and uint8.shape[2] == 3:
        uint8_bgr = cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)
    elif uint8.ndim == 2:
        uint8_bgr = uint8
    else:
        uint8_bgr = uint8

    save_image(uint8_bgr, path, metadata=meta)
