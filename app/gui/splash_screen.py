"""
Ultra-premium cinematic loading screen for SEAMS.

Spawns a background QThread that pre-compiles all Numba JIT functions
so the first user action is stall-free.
"""
import math
import random
import os
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, pyqtSignal, QThread, QRect
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient,
    QRadialGradient, QFont, QPainterPath, QPixmap, QRegion
)


class WarmupThread(QThread):
    """Background thread that pre-compiles all Numba JIT functions.

    Emits ``warmup_done(int)`` with the total elapsed milliseconds
    when finished.  The splash screen does NOT wait for this thread
    to close — compilation happens invisibly during the animation.
    """

    warmup_done = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._total_ms: int = 0

    def run(self) -> None:
        try:
            from ..core.warmup import warmup_all_jit_functions
            timings = warmup_all_jit_functions()
            self._total_ms = int(sum(timings.values()))
        except Exception as exc:
            import logging
            logging.getLogger("seams.splash").warning("JIT warmup failed: %s", exc)
            self._total_ms = -1
        self.warmup_done.emit(self._total_ms)


class Particle:
    def __init__(self, w, h):
        self.reset(w, h)

    def reset(self, w, h):
        self.x = random.uniform(0, w)
        self.y = random.uniform(0, h)
        self.vx = random.uniform(-0.3, 0.3)
        self.vy = random.uniform(-0.4, -0.1)
        self.alpha = random.uniform(0.1, 0.6)
        self.size = random.uniform(1.0, 2.5)
        self.life = random.uniform(0, 1.0)
        self.max_life = random.uniform(3.0, 8.0)

    def update(self, dt, w, h):
        self.life += dt
        self.x += self.vx
        self.y += self.vy
        t = self.life / self.max_life
        self.alpha = (1 - t) * 0.5
        if self.life > self.max_life or self.y < 0:
            self.reset(w, h)
            self.y = h


class SplashScreen(QWidget):
    finished = pyqtSignal()

    STAGES = [
        "INITIALIZING ENGINE",
        "LOADING MODULES",
        "PREPARING WORKSPACE",
        "LOADING RESOURCES",
        "FINALIZING",
    ]

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
        else:
            avail = QRect(0, 0, 1920, 1080)
        target_w = min(1200, int(avail.width() * 0.75))
        target_h = int(target_w * 9 / 16)
        if target_h > int(avail.height() * 0.70):
            target_h = int(avail.height() * 0.70)
            target_w = int(target_h * 16 / 9)
        target_w = min(target_w, avail.width() - 40)
        target_h = min(target_h, avail.height() - 40)
        self.setFixedSize(max(720, target_w), max(400, target_h))

        self.move(
            avail.x() + (avail.width() - self.width()) // 2,
            avail.y() + (avail.height() - self.height()) // 2,
        )

        # Load real logo
        logo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'resources', 'logo.png'
        )
        self._logo_pixmap = QPixmap(logo_path) if os.path.exists(logo_path) else None

        # Animation state
        self._tick = 0.0
        self._progress = 0.0
        self._stage_idx = 0
        self._orbit_angle = 0.0
        self._pulse = 0.0
        self._noise = [[random.uniform(0, 1) for _ in range(160)] for _ in range(96)]

        # Particles — fewer for faster paint
        w, h = self.width(), self.height()
        self._particles = [Particle(w, h) for _ in range(50)]

        # Contour wave points (left side)
        self._wave_offset = 0.0

        # Loading timer — 4 s total
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._step)
        self._timer.start()

        # JIT warmup thread — compiles Numba functions in background
        self._warmup_thread = WarmupThread(self)
        self._warmup_thread.warmup_done.connect(self._on_warmup_done)
        self._warmup_thread.start()

    def _on_warmup_done(self, ms: int) -> None:
        import logging
        logger = logging.getLogger("seams.splash")
        if ms >= 0:
            logger.info("JIT warmup completed in %d ms", ms)
        else:
            logger.warning("JIT warmup failed")

    def _step(self):
        dt = 0.016
        self._tick += dt
        # Keep the seams mark itself static; only ambient particles/progress animate.
        self._pulse = (math.sin(self._tick * 1.5) + 1) / 2
        self._wave_offset += 0.8

        # Progress: 0→100 over ~2s (16ms * ~125 frames)
        self._progress = min(100.0, self._progress + 0.82)
        self._stage_idx = min(4, int(self._progress / 21))

        # Update particles
        for p in self._particles:
            p.update(dt, self.width(), self.height())

        self.update()

        if self._progress >= 100:
            self._timer.stop()
            QTimer.singleShot(400, self.finished.emit)

    # ─── PAINT ────────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        W, H = self.width(), self.height()
        cx, cy = W / 2, H * 0.475

        # ── 1. Background ─────────────────────────────────────────────────────
        p.fillRect(0, 0, W, H, QColor(8, 8, 10))

        # No per-frame grain — static background only

        # Vignette
        vg = QRadialGradient(cx, cy, W * 0.75)
        vg.setColorAt(0, QColor(0, 0, 0, 0))
        vg.setColorAt(1, QColor(0, 0, 0, 180))
        p.fillRect(0, 0, W, H, QBrush(vg))

        # ── 2. Wireframe mesh (left-center) ───────────────────────────────────
        self._draw_wireframe(p, W * 0.31, cy + H * 0.045)

        # ── 3. Orbit rings ────────────────────────────────────────────────────
        self._draw_orbits(p, cx, cy)

        # ── 4. Central sphere ─────────────────────────────────────────────────
        self._draw_sphere(p, cx, cy)

        # ── 5. Particles ──────────────────────────────────────────────────────
        for pt in self._particles:
            c = QColor(255, 255, 255, int(pt.alpha * 255))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawEllipse(QPointF(pt.x, pt.y), pt.size, pt.size)

        # ── 6. Sparkles ───────────────────────────────────────────────────────
        self._draw_sparkles(p, W, H)

        # ── 7. Abstract UI elements ───────────────────────────────────────────
        self._draw_ui_elements(p, W, H)

        # ── 8. Left panel ─────────────────────────────────────────────────────
        self._draw_left_panel(p, W, H)

        # ── 9. Right panel ────────────────────────────────────────────────────
        self._draw_right_panel(p, W, H)

        # ── 10. Top bar ───────────────────────────────────────────────────────
        self._draw_top_bar(p, W)

        # ── 11. Bottom progress ───────────────────────────────────────────────
        self._draw_bottom(p, W, H, cx)

    # ─── WIREFRAME ────────────────────────────────────────────────────────────
    def _draw_wireframe(self, p, ox, oy):
        cols, rows = 9, 6
        sp = 32
        pen = QPen(QColor(255, 255, 255, 22), 0.7)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        wo = self._wave_offset
        pts = []
        for r in range(rows + 1):
            row = []
            for c in range(cols + 1):
                x = ox + (c - cols / 2) * sp
                wave = math.sin((c * 0.6 + wo * 0.05)) * 14
                y = oy + (r - rows / 2) * sp + wave
                row.append(QPointF(x, y))
            pts.append(row)
        for r in range(rows):
            for c in range(cols):
                p.drawLine(pts[r][c], pts[r][c + 1])
                p.drawLine(pts[r][c], pts[r + 1][c])
        for r in range(rows + 1):
            p.drawLine(pts[r][cols - 1], pts[r][cols])
        for c in range(cols + 1):
            p.drawLine(pts[rows - 1][c], pts[rows][c])

    # ─── ORBITS ───────────────────────────────────────────────────────────────
    def _draw_orbits(self, p, cx, cy):
        p.save()
        p.translate(cx, cy)

        # Outer ellipse orbit
        pen = QPen(QColor(255, 255, 255, 35), 0.8)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.save()
        p.rotate(self._orbit_angle * 0.3)
        p.scale(1.0, 0.35)
        p.drawEllipse(QPointF(0, 0), 230, 230)
        p.restore()

        # Inner orbit
        pen2 = QPen(QColor(160, 100, 255, 45), 1.0)
        p.setPen(pen2)
        p.save()
        p.rotate(-self._orbit_angle * 0.5)
        p.scale(1.0, 0.28)
        p.drawEllipse(QPointF(0, 0), 175, 175)
        p.restore()

        # Moving dot on outer orbit
        angle_r = math.radians(self._orbit_angle * 0.3)
        dx = math.cos(angle_r) * 230
        dy = math.sin(angle_r) * 230 * 0.35
        glow = QRadialGradient(QPointF(dx, dy), 8)
        glow.setColorAt(0, QColor(180, 120, 255, 200))
        glow.setColorAt(1, QColor(180, 120, 255, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(QPointF(dx, dy), 8, 8)

        p.restore()

    # ─── LOGO / SPHERE ────────────────────────────────────────────────────────
    def _draw_sphere(self, p, cx, cy):
        R = 145
        # Slow breathing scale: 0.94 → 1.0
        scale = 1.0

        # Ambient purple glow behind the logo
        glow_r = R * 1.8 * scale
        glow = QRadialGradient(QPointF(cx, cy), glow_r)
        glow.setColorAt(0, QColor(70, 30, 110, 72))
        glow.setColorAt(0.45, QColor(30, 10, 55, 25))
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

        if self._logo_pixmap and not self._logo_pixmap.isNull():
            logo_size = int(R * 2.2 * scale)
            scaled = self._logo_pixmap.scaled(
                logo_size, logo_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            draw_x = int(cx - scaled.width() / 2)
            draw_y = int(cy - scaled.height() / 2)
            p.setOpacity(0.95)
            p.drawPixmap(draw_x, draw_y, scaled)
            p.setOpacity(1.0)
        else:
            # Fallback simple sphere
            rg = QRadialGradient(QPointF(cx - R * 0.3, cy - R * 0.3), R * 1.4)
            rg.setColorAt(0, QColor(230, 228, 224))
            rg.setColorAt(0.5, QColor(150, 145, 140))
            rg.setColorAt(1, QColor(40, 38, 36))
            p.setBrush(QBrush(rg))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(cx, cy), R * scale, R * scale)


    # ─── SPARKLES ─────────────────────────────────────────────────────────────
    def _draw_sparkles(self, p, W, H):
        spots = [
            (W * 0.82, H * 0.12, 10),
            (W * 0.67, H * 0.22, 7),
            (W * 0.18, H * 0.35, 8),
            (W * 0.75, H * 0.62, 6),
            (W * 0.92, H * 0.55, 9),
        ]
        t = self._tick
        for (sx, sy, sz) in spots:
            pulse = (math.sin(t * 2 + sx) + 1) / 2
            alpha = int(pulse * 180 + 30)
            pen = QPen(QColor(200, 160, 255, alpha), 1.2)
            p.setPen(pen)
            s = sz * (0.7 + pulse * 0.5)
            p.drawLine(QPointF(sx - s, sy), QPointF(sx + s, sy))
            p.drawLine(QPointF(sx, sy - s), QPointF(sx, sy + s))
            s2 = s * 0.5
            p.drawLine(QPointF(sx - s2, sy - s2), QPointF(sx + s2, sy + s2))
            p.drawLine(QPointF(sx + s2, sy - s2), QPointF(sx - s2, sy + s2))

    # ─── UI ELEMENTS ──────────────────────────────────────────────────────────
    def _draw_ui_elements(self, p, W, H):
        s = min(W / 1536.0, H / 864.0)
        # Crosshair target (bottom-left area)
        self._draw_crosshair(p, 208 * s, H * 0.82, 58 * s, QColor(160, 100, 255, 60))
        # Small crosshair right
        self._draw_crosshair(p, W * 0.89, H * 0.66, 18 * s, QColor(255, 255, 255, 35))
        # Corner brackets top-right
        self._draw_corner_bracket(p, W - 74 * s, 46 * s, 22 * s, QColor(255, 255, 255, 40))

    def _draw_crosshair(self, p, cx, cy, r, color):
        pen = QPen(color, 0.8)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.drawEllipse(QPointF(cx, cy), r * 0.55, r * 0.55)
        p.drawEllipse(QPointF(cx, cy), r * 0.15, r * 0.15)
        p.drawLine(QPointF(cx - r * 1.4, cy), QPointF(cx - r * 1.0, cy))
        p.drawLine(QPointF(cx + r * 1.0, cy), QPointF(cx + r * 1.4, cy))
        p.drawLine(QPointF(cx, cy - r * 1.4), QPointF(cx, cy - r * 1.0))
        p.drawLine(QPointF(cx, cy + r * 1.0), QPointF(cx, cy + r * 1.4))

    def _draw_corner_bracket(self, p, x, y, s, color):
        pen = QPen(color, 1.0)
        p.setPen(pen)
        p.drawLine(QPointF(x, y), QPointF(x + s, y))
        p.drawLine(QPointF(x, y), QPointF(x, y + s))

    # ─── LEFT PANEL ───────────────────────────────────────────────────────────
    def _draw_left_panel(self, p, W, H):
        s = min(W / 1536.0, H / 864.0)
        x = 46 * s
        top = 44 * s
        # "seams" logo text
        f = QFont("Impact", max(34, int(66 * s)), QFont.Weight.Black)
        f.setStretch(72)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0)
        p.setFont(f)
        p.setPen(QColor(240, 240, 240, 255))
        p.drawText(QPointF(x, top + 52 * s), "seams")
        logo_w = p.fontMetrics().horizontalAdvance("seams")

        # Vertical divider after logo
        divider_x = x + logo_w + 18 * s
        p.setPen(QPen(QColor(255, 255, 255, 40), 0.8))
        p.drawLine(QPointF(divider_x, top - 2 * s), QPointF(divider_x, top + 64 * s))

        # "SEAMLESS / TEXTURE / STUDIO" stacked
        text_x = divider_x + 18 * s
        f2 = QFont("Segoe UI", max(7, int(10 * s)), QFont.Weight.Bold)
        f2.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        p.setFont(f2)
        p.setPen(QColor(200, 200, 210, 220))
        p.drawText(QPointF(text_x, top + 18 * s), "SEAMLESS")
        p.drawText(QPointF(text_x, top + 38 * s), "TEXTURE")
        p.drawText(QPointF(text_x, top + 58 * s), "STUDIO")

        # Loading indicator — purple dots
        dot_y = H * 0.43
        dot_x = 64 * s
        for i in range(3):
            phase = (self._tick * 2 + i * 0.7) % (2 * math.pi)
            alpha = int((math.sin(phase) + 1) / 2 * 200 + 55)
            size = (4.0 + (math.sin(phase) + 1) * 1.5) * s
            c = QColor(160, 80, 255, alpha) if i == 0 else QColor(120, 60, 200, alpha // 2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawEllipse(QPointF(dot_x, dot_y + i * 14 * s), size / 2, size / 2)

        f3 = QFont("Segoe UI", max(8, int(11 * s)), QFont.Weight.Bold)
        f3.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        p.setFont(f3)
        p.setPen(QColor(160, 80, 255, 220))
        p.drawText(QPointF(88 * s, dot_y + 5 * s), "LOADING")
        f4 = QFont("Segoe UI", max(7, int(9 * s)))
        f4.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        p.setFont(f4)
        p.setPen(QColor(180, 170, 200, 200))
        p.drawText(QPointF(88 * s, dot_y + 25 * s), "TEXTURE ENGINE")

        # Version + copyright
        fv = QFont("Segoe UI", max(7, int(9 * s)))
        fv.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.5)
        p.setFont(fv)

        p.setPen(QColor(150, 148, 165, 210))
        from ..utils.config import APP_VERSION
        p.drawText(QPointF(46 * s, H - 76 * s), f"v {APP_VERSION}")
        p.setPen(QColor(120, 118, 135, 190))
        p.drawText(QPointF(46 * s, H - 44 * s), "\u00a9 2024-2026 SEAMS STUDIO")
        p.drawText(QPointF(46 * s, H - 26 * s), "ALL RIGHTS RESERVED")

    # ─── RIGHT PANEL ──────────────────────────────────────────────────────────
    def _draw_right_panel(self, p, W, H):
        s = min(W / 1536.0, H / 864.0)
        rx = W - 316 * s
        ry = H * 0.45

        for i, stage in enumerate(self.STAGES):
            is_active = (i == self._stage_idx)
            is_done = (i < self._stage_idx)

            num_str = f"0{i+1}"
            fy = ry + i * 43 * s

            # Number
            fn = QFont("Segoe UI", max(8, int(10 * s)), QFont.Weight.Bold)
            p.setFont(fn)
            if is_active:
                p.setPen(QColor(160, 80, 255, 230))
            elif is_done:
                p.setPen(QColor(140, 135, 160, 200))
            else:
                p.setPen(QColor(120, 118, 140, 165))
            p.drawText(int(rx), int(fy), num_str)

            # Stage label
            fs = QFont("Segoe UI", max(8, int(10 * s)))
            fs.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
            if is_active:
                fs.setWeight(QFont.Weight.Bold)
            p.setFont(fs)
            if is_active:
                pulse_a = int(180 + self._pulse * 75)
                p.setPen(QColor(180, 100, 255, pulse_a))
            elif is_done:
                p.setPen(QColor(160, 155, 180, 180))
            else:
                p.setPen(QColor(140, 138, 158, 140))
            p.drawText(QPointF(rx + 34 * s, fy), stage)

            # Active line accent
            if is_active:
                glow_pen = QPen(QColor(160, 80, 255, int(120 + self._pulse * 80)), 0.8)
                p.setPen(glow_pen)
                p.drawLine(QPointF(rx + 34 * s, fy + 4 * s), QPointF(rx + 118 * s, fy + 4 * s))

    # ─── TOP BAR ──────────────────────────────────────────────────────────────
    def _draw_top_bar(self, p, W):
        s = min(W / 1536.0, self.height() / 864.0)
        fb = QFont("Segoe UI", max(7, int(9 * s)), QFont.Weight.Bold)
        fb.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        p.setFont(fb)
        p.setPen(QColor(190, 185, 210, 220))
        txt = "BUILT FOR CREATORS"
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(txt)
        p.drawText(QPointF(W - tw - 74 * s, 75 * s), txt)
        # + sparkle
        p.setPen(QPen(QColor(160, 80, 255, 180), 1.2))
        px = W - 44 * s
        py = 69 * s
        p.drawLine(QPointF(px - 5 * s, py), QPointF(px + 5 * s, py))
        p.drawLine(QPointF(px, py - 5 * s), QPointF(px, py + 5 * s))

    # ─── BOTTOM PROGRESS ──────────────────────────────────────────────────────
    def _draw_bottom(self, p, W, H, cx):
        s = min(W / 1536.0, H / 864.0)
        bar_w = 534 * s
        bar_h = max(3, int(4 * s))
        bar_x = cx - bar_w / 2
        bar_y = H - 104 * s

        # Stage label above bar
        fl = QFont("Segoe UI", max(7, int(9 * s)), QFont.Weight.Bold)
        fl.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        p.setFont(fl)
        p.setPen(QColor(210, 205, 230, 230))
        stage_txt = self.STAGES[self._stage_idx]
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(stage_txt)
        p.drawText(QPointF(cx - tw / 2, bar_y - 24 * s), stage_txt)

        # Background track
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(28, 24, 36, 200))
        p.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2 * s, 2 * s)

        # Purple fill
        fill_w = bar_w * self._progress / 100.0
        if fill_w > 2:
            lg = QLinearGradient(bar_x, 0, bar_x + fill_w, 0)
            lg.setColorAt(0, QColor(90, 40, 160))
            lg.setColorAt(0.6, QColor(150, 80, 255))
            lg.setColorAt(1, QColor(200, 140, 255))
            p.setBrush(QBrush(lg))
            p.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 2 * s, 2 * s)

            # Glow tip
            gx = bar_x + fill_w
            tip = QRadialGradient(QPointF(gx, bar_y + bar_h / 2), 10)
            tip.setColorAt(0, QColor(200, 140, 255, int(180 + self._pulse * 75)))
            tip.setColorAt(1, QColor(200, 140, 255, 0))
            p.setBrush(QBrush(tip))
            p.drawEllipse(QPointF(gx, bar_y + bar_h / 2), 10 * s, 10 * s)

        # Percentage
        fp = QFont("Segoe UI", max(9, int(11 * s)), QFont.Weight.Bold)
        p.setFont(fp)
        p.setPen(QColor(160, 160, 175, 220))
        pct_txt = f"{int(self._progress)}%"
        p.drawText(QPointF(bar_x + bar_w + 24 * s, bar_y + 7 * s), pct_txt)

        # Tagline
        ft = QFont("Segoe UI", max(8, int(10 * s)))
        ft.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
        p.setFont(ft)
        p.setPen(QColor(170, 165, 190, 210))
        tag = "GREAT TEXTURES START WITH GREAT TOOLS."
        fm2 = p.fontMetrics()
        tw2 = fm2.horizontalAdvance(tag)
        p.drawText(QPointF(cx - tw2 / 2, bar_y + 31 * s), tag)

        # Sparkle below tagline
        p.setPen(QPen(QColor(160, 80, 255, int(120 + self._pulse * 135)), 1.2))
        sx, sy = cx, bar_y + 55 * s
        p.drawLine(QPointF(sx - 5 * s, sy), QPointF(sx + 5 * s, sy))
        p.drawLine(QPointF(sx, sy - 5 * s), QPointF(sx, sy + 5 * s))
