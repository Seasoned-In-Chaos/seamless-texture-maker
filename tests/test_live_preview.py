import numpy as np

from app.gui.main_window import MainWindow


class _Viewer:
    def __init__(self):
        self.maps = {}

    def set_map(self, name, image):
        self.maps[name] = image


class _Window:
    def __init__(self, source):
        self.material_maps = {"Normal": source.copy()}
        self.image_viewer = _Viewer()
        self.processed_normal_map = None


def test_generated_channel_preview_does_not_replace_full_resolution_source():
    source = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    preview = source[:2].copy()
    window = _Window(source)

    MainWindow._apply_edit_result(window, "Normal", preview, preview=True)

    np.testing.assert_array_equal(window.material_maps["Normal"], source)
    np.testing.assert_array_equal(window.image_viewer.maps["Normal"], preview)


def test_generated_channel_full_resolution_result_is_committed():
    source = np.zeros((4, 2, 3), dtype=np.float32)
    result = np.ones_like(source)
    window = _Window(source)

    MainWindow._apply_edit_result(window, "Normal", result)

    np.testing.assert_array_equal(window.material_maps["Normal"], result)
    np.testing.assert_array_equal(window.image_viewer.maps["Normal"], result)
    np.testing.assert_array_equal(window.processed_normal_map, result)
