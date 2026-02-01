"""
Application configuration and settings.
"""
import os
import json


# Application info
APP_NAME = "Seamless Texture Maker"
APP_VERSION = "1.0"
APP_AUTHOR = "Studio Tools"

# Default settings
DEFAULT_SETTINGS = {
    'blend_strength': 0.5,
    'seam_smoothness': 0.5,
    'detail_preservation': 0.75,
    'symmetric_blending': True,
    'export_format': 'png',
    'save_mode': 'new_file',  # 'new_file' or 'overwrite'
    'last_directory': '',
    'window_width': 1200,
    'window_height': 800,
}


def get_config_path():
    """Get path to config file."""
    app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
    config_dir = os.path.join(app_data, 'SeamlessTextureMaker')
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, 'settings.json')


def load_settings():
    """Load settings from config file."""
    config_path = get_config_path()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                saved = json.load(f)
                # Merge with defaults
                settings = DEFAULT_SETTINGS.copy()
                settings.update(saved)
                return settings
        except Exception:
            pass
    
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    """Save settings to config file."""
    config_path = get_config_path()
    
    try:
        with open(config_path, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception:
        return False
