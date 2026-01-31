"""
Script to convert PNG icon to ICO format for Windows.
"""
from PIL import Image
import os

def create_ico(png_path, ico_path):
    """Convert PNG to ICO with multiple sizes."""
    img = Image.open(png_path)
    
    # Create multiple sizes for ICO
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    
    # Resize and save
    icons = []
    for size in sizes:
        resized = img.resize(size, Image.Resampling.LANCZOS)
        icons.append(resized)
    
    # Save as ICO
    img.save(ico_path, format='ICO', sizes=[(s.width, s.height) for s in icons])
    print(f"Created: {ico_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(script_dir, 'resources', 'icon.png')
    ico_path = os.path.join(script_dir, 'resources', 'icon.ico')
    
    if os.path.exists(png_path):
        create_ico(png_path, ico_path)
    else:
        print(f"PNG not found: {png_path}")
