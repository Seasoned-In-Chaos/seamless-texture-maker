import os
from PIL import Image, ImageDraw, ImageFont

def create_splash():
    width, height = 600, 350
    # Create an image with a dark gradient background
    img = Image.new('RGB', (width, height), color=(20, 20, 25))
    draw = ImageDraw.Draw(img)
    
    # Draw a subtle gradient
    for y in range(height):
        r = int(20 + (y / height) * 20)
        g = int(20 + (y / height) * 25)
        b = int(25 + (y / height) * 40)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
        
    # Draw some "tech" or "design" accents
    draw.rectangle([0, height-6, width, height], fill=(60, 120, 255))
    
    try:
        # Try to use a system font if available
        font_large = ImageFont.truetype("segoeui.ttf", 42)
        font_small = ImageFont.truetype("segoeui.ttf", 16)
        font_version = ImageFont.truetype("segoeui.ttf", 14)
    except IOError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_version = ImageFont.load_default()

    # Text wrapping/positioning
    main_text = "Seamless Texture Maker"
    sub_text = "Professional 3D Material Generator"
    loading_text = "Loading application resources..."
    version_text = "v1.0.0"

    # We can use textbbox to center text
    def get_text_pos(text, font, y_offset, center=True, align_left=False, x_offset=0):
        try:
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(text, font=font)
        
        if center:
            x = (width - tw) / 2
        elif align_left:
            x = x_offset
        else:
            x = width - tw - x_offset
            
        return x, y_offset

    # Draw texts
    x, y = get_text_pos(main_text, font_large, 120)
    draw.text((x, y), main_text, fill=(255, 255, 255), font=font_large)

    x, y = get_text_pos(sub_text, font_small, 180)
    draw.text((x, y), sub_text, fill=(180, 180, 200), font=font_small)

    x, y = get_text_pos(loading_text, font_small, height - 40, align_left=True, x_offset=20)
    draw.text((x, y), loading_text, fill=(150, 150, 150), font=font_small)

    x, y = get_text_pos(version_text, font_version, height - 40, center=False, x_offset=20)
    draw.text((x, y), version_text, fill=(100, 100, 100), font=font_version)

    # Save to resources folder
    out_dir = os.path.join(os.path.dirname(__file__), 'resources')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'splash.png')
    img.save(out_path)
    print(f"Splash screen saved to {out_path}")

if __name__ == '__main__':
    create_splash()
