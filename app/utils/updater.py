"""
In-app auto-update system for SEAMS.

Checks GitHub Releases API for a newer version, downloads the
new EXE in the background, then performs a swap-on-restart so
the user never has to manually reinstall.

Flow:
  1. ``UpdateChecker`` QThread calls GitHub API on startup
  2. If update found, ``UpdateBanner`` shows a non-intrusive bar
  3. User clicks "Download & Restart"
  4. ``UpdateDownloader`` QThread downloads the new EXE
  5. On completion, a small batch file is written that waits for
     the current process to exit, copies the new EXE over the old
     one, re-launches, and deletes itself
  6. App calls ``QApplication.quit()`` and the batch file takes over
"""
from __future__ import annotations

import json
import logging
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from typing import Optional, Tuple

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget
from PyQt6.QtCore import QUrl

from ..utils.config import APP_VERSION
from ..utils.app_logging import user_data_dir

logger = logging.getLogger("seams.updater")

GITHUB_OWNER = "Seasoned-In-Chaos"
GITHUB_REPO = "seamless-texture-maker"
API_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"


def _current_exe_path() -> str:
    if getattr(sys, "frozen", False):
        return sys.executable
    return ""


def check_for_update(current_version: str) -> Optional[Tuple[str, str, str]]:
    """Check GitHub Releases for a newer version.

    Args:
        current_version: Semver string, e.g. "2.0.0".

    Returns:
        ``(new_version, download_url, release_notes)`` if a newer
        release exists, otherwise ``None``.  Returns ``None``
        silently on any error.
    """
    try:
        req = urllib.request.Request(API_URL, headers={"User-Agent": "SEAMS-Updater"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        tag = data.get("tag_name", "").lstrip("v")
        body = data.get("body", "") or ""
        assets = data.get("assets", [])

        exe_asset = None
        for a in assets:
            name = a.get("name", "").lower()
            if name.endswith(".exe") and "setup" not in name:
                exe_asset = a
                break
        if exe_asset is None and assets:
            exe_asset = assets[0]

        download_url = exe_asset["browser_download_url"] if exe_asset else ""

        from packaging.version import parse as _parse
        if _parse(tag) > _parse(current_version):
            logger.info("Update available: %s -> %s", current_version, tag)
            return tag, download_url, body[:500]

        logger.debug("No update available (current=%s, latest=%s)", current_version, tag)
    except Exception as exc:
        logger.debug("Update check failed: %s", exc)

    return None


def _sha256_file(path: str) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _fetch_release_sha256(download_url: str) -> Optional[str]:
    """Try to fetch a .sha256 sibling file for the given asset URL.

    GitHub Releases sometimes include a ``<name>.sha256`` sidecar.
    Returns the hex digest string, or None on failure.
    """
    sha_url = download_url + ".sha256"
    try:
        req = urllib.request.Request(sha_url, headers={"User-Agent": "SEAMS-Updater"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            text = resp.read().decode("utf-8").strip()
            return text.split()[0]
    except Exception:
        return None


class UpdateChecker(QThread):
    """Background thread that checks for updates on startup."""

    update_found = pyqtSignal(str, str, str)
    no_update = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._version = APP_VERSION

    def run(self):
        result = check_for_update(self._version)
        if result:
            self.update_found.emit(result[0], result[1], result[2])
        else:
            self.no_update.emit()


class UpdateDownloader(QThread):
    """Background thread that downloads the new EXE."""

    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        self._url = url
        self._dest = os.path.join(user_data_dir(), "update", "SEAMS_new.exe")

    def run(self):
        try:
            os.makedirs(os.path.dirname(self._dest), exist_ok=True)
            req = urllib.request.Request(self._url, headers={"User-Agent": "SEAMS-Updater"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 65536
                with open(self._dest, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            self.progress.emit(int(downloaded * 100 / total))
            logger.info("Update downloaded to %s", self._dest)
            self.finished_ok.emit(self._dest)
        except Exception as exc:
            logger.error("Update download failed: %s", exc)
            self.error.emit(str(exc))


def apply_update_and_restart(new_exe: str, current_exe: str, download_url: str = "") -> None:
    """Write a batch file that swaps the EXE and restarts.

    The batch file:
      1. Waits for the current SEAMS.exe process to exit
      2. Verifies SHA256 integrity (if a sidecar hash is available)
      3. Copies the new EXE over the old one
      4. Re-launches the app
      5. Deletes itself
    """
    if download_url:
        expected = _fetch_release_sha256(download_url)
        if expected:
            actual = _sha256_file(new_exe)
            if actual != expected:
                logger.error("SHA256 mismatch: expected %s, got %s — aborting update", expected[:16], actual[:16])
                raise RuntimeError(f"Integrity check failed: SHA256 mismatch")
            logger.info("SHA256 verified OK")

    update_dir = os.path.join(user_data_dir(), "update")
    os.makedirs(update_dir, exist_ok=True)
    bat_path = os.path.join(update_dir, "apply_update.bat")

    current_dir = os.path.dirname(current_exe)
    bat_content = (
        '@echo off\r\n'
        'echo SEAMS is updating...\r\n'
        'echo Waiting for the application to close...\r\n'
        ':waitloop\r\n'
        'tasklist /fi "PID eq {pid}" 2>nul | find "{pid}" >nul\r\n'
        'if not errorlevel 1 (\r\n'
        '    timeout /t 1 /nobreak >nul\r\n'
        '    goto waitloop\r\n'
        ')\r\n'
        'timeout /t 2 /nobreak >nul\r\n'
        'echo Copying update...\r\n'
        'copy /y "{new_exe}" "{current_exe}"\r\n'
        'if errorlevel 1 (\r\n'
        '    echo Update failed - could not copy file.\r\n'
        '    pause\r\n'
        '    goto cleanup\r\n'
        ')\r\n'
        'echo Restarting SEAMS...\r\n'
        'start "" "{current_exe}"\r\n'
        ':cleanup\r\n'
        'del /q "{new_exe}" 2>nul\r\n'
        'cd /d "{update_dir}"\r\n'
        'del /q "%~f0" 2>nul\r\n'
    ).format(
        pid=os.getpid(),
        new_exe=new_exe.replace("/", "\\"),
        current_exe=current_exe.replace("/", "\\"),
        update_dir=update_dir.replace("/", "\\"),
    )

    with open(bat_path, "w") as f:
        f.write(bat_content)

    subprocess.Popen(
        ["cmd", "/c", bat_path],
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
        close_fds=True,
    )
    logger.info("Update batch launched, quitting for restart...")


def cleanup_stale_update() -> None:
    """Remove leftover update files from a previous session."""
    update_dir = os.path.join(user_data_dir(), "update")
    for name in ("SEAMS_new.exe", "apply_update.bat"):
        p = os.path.join(update_dir, name)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


class UpdateBanner(QWidget):
    """Non-intrusive banner shown at top of main window when an update is available."""

    download_requested = pyqtSignal()
    dismiss_requested = pyqtSignal()

    def __init__(self, version: str, parent=None):
        super().__init__(parent)
        self.setObjectName("UpdateBanner")
        self.setFixedHeight(42)
        self.setStyleSheet("""
            QWidget#UpdateBanner {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a0a30, stop:1 #2d1450);
                border-bottom: 1px solid rgba(160,80,255,0.3);
            }
            QLabel { color: #d0c8e0; font-size: 12px; }
            QPushButton {
                background: rgba(160,80,255,0.25);
                color: #e0d0ff;
                border: 1px solid rgba(160,80,255,0.5);
                border-radius: 4px;
                padding: 4px 14px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background: rgba(160,80,255,0.45); }
            QPushButton#DismissBtn {
                background: transparent;
                border: none;
                color: #807090;
                padding: 4px 8px;
                font-weight: normal;
            }
            QPushButton#DismissBtn:hover { color: #c0b0d0; }
            QProgressBar {
                background: #1a0a30;
                border: 1px solid rgba(160,80,255,0.3);
                border-radius: 3px;
                max-height: 8px;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7b2ff7, stop:1 #c084fc);
                border-radius: 2px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(10)

        self._label = QLabel(f"  Update {version} is available")
        layout.addWidget(self._label)

        self._progress = None

        self._download_btn = QPushButton("Download & Restart")
        self._download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._download_btn.clicked.connect(self._on_download)
        layout.addWidget(self._download_btn)

        self._release_btn = QPushButton("View Release Notes")
        self._release_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._release_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #a080c0; "
            "font-size: 11px; padding: 4px 8px; } QPushButton:hover { color: #d0b0f0; }"
        )
        self._release_btn.clicked.connect(self._on_view_notes)
        layout.addWidget(self._release_btn)

        layout.addStretch()

        dismiss = QPushButton("\u2715")
        dismiss.setObjectName("DismissBtn")
        dismiss.setCursor(Qt.CursorShape.PointingHandCursor)
        dismiss.setFixedSize(28, 28)
        dismiss.clicked.connect(self.dismiss_requested.emit)
        layout.addWidget(dismiss)

        self._version = version
        self._download_url = ""
        self._release_notes = ""

    def set_info(self, download_url: str, release_notes: str):
        self._download_url = download_url
        self._release_notes = release_notes

    def show_progress(self):
        self._download_btn.setVisible(False)
        self._label.setText("  Downloading update...")
        if self._progress is None:
            from PyQt6.QtWidgets import QProgressBar
            self._progress = QProgressBar()
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._progress.setFixedWidth(180)
            layout = self.layout()
            layout.insertWidget(1, self._progress)

    def set_progress(self, pct: int):
        if self._progress:
            self._progress.setValue(pct)

    def show_complete(self):
        if self._progress:
            self._progress.setValue(100)
        self._label.setText("  Update ready — restarting...")

    def show_error(self, msg: str):
        self._label.setText(f"  Update failed: {msg}")
        if self._progress:
            self._progress.setVisible(False)
        self._download_btn.setVisible(True)
        self._download_btn.setText("Retry")

    def _on_download(self):
        self.download_requested.emit()

    def _on_view_notes(self):
        url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
        QDesktopServices.openUrl(QUrl(url))
