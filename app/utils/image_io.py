"""
Image I/O utilities with DPI/resolution preservation.
"""
import os
import numpy as np
import cv2
from PIL import Image


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
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")
    
    # Get metadata using PIL
    metadata = ImageMetadata()
    metadata.filepath = filepath
    
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
    
    # Load image with OpenCV for processing
    # Use IMREAD_UNCHANGED to preserve alpha and bit depth
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise IOError(f"Failed to load image: {filepath}")
    
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
    # Determine format from extension if not specified
    if format is None:
        ext = os.path.splitext(filepath)[1].lower()
        format_map = {
            '.png': 'png',
            '.jpg': 'jpg',
            '.jpeg': 'jpg',
            '.tif': 'tiff',
            '.tiff': 'tiff'
        }
        format = format_map.get(ext, 'png')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
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
    
    # Save the image
    pil_image.save(filepath, **save_kwargs)
    
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
    directory = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)
    
    if output_format:
        ext_map = {
            'png': '.png',
            'jpg': '.jpg',
            'tiff': '.tiff'
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
    return ['PNG', 'JPG', 'JPEG', 'TIFF', 'TIF']


def get_format_filter():
    """Get file dialog filter string."""
    return "Images (*.png *.jpg *.jpeg *.tiff *.tif);;PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tiff *.tif)"
