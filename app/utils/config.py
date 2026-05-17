"""
Application configuration and settings.
"""
import os
import json

from .app_logging import get_logger, log_exception, user_data_dir


logger = get_logger(__name__)


# Application info
APP_NAME = "seams"
APP_VERSION = "2.0.0"
APP_AUTHOR = "Shubham Panchasara"

# Default settings
DEFAULT_SETTINGS = {
    'blend_strength': 0.5,
    'seam_smoothness': 0.5,
    'detail_preservation': 0.75,
    'symmetric_blending': True,
    'export_format': 'png',
    'save_mode': 'new_file',  # 'new_file' or 'overwrite'
    'last_directory': '',
    'last_file': '',
    'recent_files': [],
    'active_tool': 'home',
    'preview_tab': 0,
    'window_width': 1200,
    'window_height': 800,
}


def get_config_path():
    """Get path to config file."""
    return os.path.join(user_data_dir(), 'settings.json')


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
        except Exception as exc:
            log_exception(logger, f"Failed to load settings from {config_path}", exc)
    
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    """Save settings to config file."""
    config_path = get_config_path()
    
    try:
        tmp_path = f"{config_path}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp_path, config_path)
        return True
    except Exception as exc:
        log_exception(logger, f"Failed to save settings to {config_path}", exc)
        return False
