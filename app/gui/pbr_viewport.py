"""Realtime GPU material viewport for the SEAMS studio workspace.

The app in this repository is a native PyQt desktop application, so this
module implements the requested hero viewport with Qt's OpenGL surface instead
of a browser or server runtime. It keeps the same production goals: one render
loop, persistent GPU resources, dynamic PBR texture replacement, orbit damping,
mesh switching, tiling, and channel isolation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QImage,
    QMatrix4x4,
    QMouseEvent,
    QVector3D,
)
from PyQt6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVersionFunctionsFactory,
    QOpenGLVersionProfile,
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from ..utils.app_logging import get_logger, log_exception


logger = get_logger(__name__)


GL_FLOAT = 0x1406
GL_TRIANGLES = 0x0004
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_DEPTH_TEST = 0x0B71
GL_CULL_FACE = 0x0B44
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_FRONT_AND_BACK = 0x0408
GL_LINE = 0x1B01
GL_FILL = 0x1B02
GL_TEXTURE0 = 0x84C0


CHANNELS = {
    "Base Color": {
        "uniform": "u_baseMap",
        "flag": "u_hasBaseMap",
        "unit": 0,
        "colorspace": "sRGB",
    },
    "Normal": {
        "uniform": "u_normalMap",
        "flag": "u_hasNormalMap",
        "unit": 1,
        "colorspace": "Linear",
    },
    "Roughness": {
        "uniform": "u_roughnessMap",
        "flag": "u_hasRoughnessMap",
        "unit": 2,
        "colorspace": "Linear",
    },
    "Metallic": {
        "uniform": "u_metalnessMap",
        "flag": "u_hasMetalnessMap",
        "unit": 3,
        "colorspace": "Linear",
    },
    "AO": {
        "uniform": "u_aoMap",
        "flag": "u_hasAoMap",
        "unit": 4,
        "colorspace": "Linear",
    },
    "Height": {
        "uniform": "u_heightMap",
        "flag": "u_hasHeightMap",
        "unit": 5,
        "colorspace": "Linear",
    },
    "Displacement": {
        "uniform": "u_heightMap",
        "flag": "u_hasHeightMap",
        "unit": 5,
        "colorspace": "Linear",
    },
    "Opacity": {
        "uniform": "u_alphaMap",
        "flag": "u_hasAlphaMap",
        "unit": 6,
        "colorspace": "Linear",
    },
    "Emissive": {
        "uniform": "u_emissiveMap",
        "flag": "u_hasEmissiveMap",
        "unit": 7,
        "colorspace": "sRGB",
    },
}


@dataclass
class MeshData:
    vertices: np.ndarray
    vertex_count: int
    triangle_count: int


@dataclass
class TextureRecord:
    texture: QOpenGLTexture
    width: int
    height: int
    bytes_estimate: int


def numpy_to_qimage(image: np.ndarray | None) -> QImage:
    """Create an owned RGB/RGBA QImage from an OpenCV/numpy image."""
    if image is None:
        return QImage()
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if rgb.shape[2] == 4:
        fmt = QImage.Format.Format_RGBA8888
    else:
        fmt = QImage.Format.Format_RGB888
    h, w, c = rgb.shape
    qimg = QImage(rgb.data, w, h, w * c, fmt)
    return qimg.copy()


def _normalize(v: QVector3D) -> QVector3D:
    if v.lengthSquared() <= 1e-8:
        return QVector3D(0.0, 0.0, 0.0)
    return v.normalized()


def _face_basis(axis: str):
    if axis == "+x":
        return QVector3D(1, 0, 0), QVector3D(0, 0, -1), QVector3D(0, 1, 0)
    if axis == "-x":
        return QVector3D(-1, 0, 0), QVector3D(0, 0, 1), QVector3D(0, 1, 0)
    if axis == "+y":
        return QVector3D(0, 1, 0), QVector3D(1, 0, 0), QVector3D(0, 0, -1)
    if axis == "-y":
        return QVector3D(0, -1, 0), QVector3D(1, 0, 0), QVector3D(0, 0, 1)
    if axis == "+z":
        return QVector3D(0, 0, 1), QVector3D(1, 0, 0), QVector3D(0, 1, 0)
    return QVector3D(0, 0, -1), QVector3D(-1, 0, 0), QVector3D(0, 1, 0)


def _append_vertex(out, pos: QVector3D, normal: QVector3D, u: float, v: float):
    out.extend([pos.x(), pos.y(), pos.z(), normal.x(), normal.y(), normal.z(), u, v])


def _build_plane(size=3.0, segments=96, orientation="xz") -> MeshData:
    out = []
    half = size * 0.5
    for y in range(segments):
        v0 = y / segments
        v1 = (y + 1) / segments
        for x in range(segments):
            u0 = x / segments
            u1 = (x + 1) / segments
            coords = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            pts = []
            for u, v in coords:
                px = (u - 0.5) * size
                py = (v - 0.5) * size
                if orientation == "xy":
                    pts.append(QVector3D(px, py, 0.0))
                    normal = QVector3D(0, 0, 1)
                else:
                    pts.append(QVector3D(px, 0.0, py))
                    normal = QVector3D(0, 1, 0)
            for i in (0, 1, 2, 0, 2, 3):
                u, v = coords[i]
                _append_vertex(out, pts[i], normal, u, v)
    data = np.asarray(out, dtype=np.float32)
    return MeshData(data, data.size // 8, data.size // 24)


def _build_cube(size=2.2, segments=40, rounded=False) -> MeshData:
    out = []
    half = size * 0.5
    for axis in ["+x", "-x", "+y", "-y", "+z", "-z"]:
        face_n, tangent, bitangent = _face_basis(axis)
        center = face_n * half
        for y in range(segments):
            v0 = y / segments
            v1 = (y + 1) / segments
            for x in range(segments):
                u0 = x / segments
                u1 = (x + 1) / segments
                coords = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
                pts = []
                norms = []
                for u, v in coords:
                    pos = center + tangent * ((u - 0.5) * size) + bitangent * ((v - 0.5) * size)
                    normal = _normalize(pos) if rounded else face_n
                    if rounded:
                        # A bevel-like preview mesh: cube proportions with softened normals.
                        pos = pos * 0.88 + normal * (half * 0.12)
                    pts.append(pos)
                    norms.append(normal)
                for i in (0, 1, 2, 0, 2, 3):
                    u, v = coords[i]
                    _append_vertex(out, pts[i], norms[i], u, v)
    data = np.asarray(out, dtype=np.float32)
    return MeshData(data, data.size // 8, data.size // 24)


def _build_sphere(radius=1.35, lat=64, lon=96) -> MeshData:
    out = []
    for y in range(lat):
        v0 = y / lat
        v1 = (y + 1) / lat
        theta0 = v0 * math.pi
        theta1 = v1 * math.pi
        for x in range(lon):
            u0 = x / lon
            u1 = (x + 1) / lon
            phi0 = u0 * math.tau
            phi1 = u1 * math.tau
            coords = [(u0, v0, phi0, theta0), (u1, v0, phi1, theta0), (u1, v1, phi1, theta1), (u0, v1, phi0, theta1)]
            pts = []
            for u, v, phi, theta in coords:
                n = QVector3D(
                    math.sin(theta) * math.cos(phi),
                    math.cos(theta),
                    math.sin(theta) * math.sin(phi),
                )
                pts.append((n * radius, n, u, v))
            for i in (0, 1, 2, 0, 2, 3):
                pos, normal, u, v = pts[i]
                _append_vertex(out, pos, normal, u, v)
    data = np.asarray(out, dtype=np.float32)
    return MeshData(data, data.size // 8, data.size // 24)


def build_mesh(name: str) -> MeshData:
    if name == "Cube":
        return _build_cube()
    if name == "Plane":
        return _build_plane(size=3.0, orientation="xz")
    if name == "Wall":
        return _build_plane(size=3.2, orientation="xy")
    if name == "Floor Tile":
        return _build_plane(size=4.0, orientation="xz")
    if name == "Rounded Cube":
        return _build_cube(rounded=True)
    return _build_sphere()


VERTEX_SHADER = """
attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec2 a_uv;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normalMatrix;
uniform vec2 u_tiling;
uniform sampler2D u_heightMap;
uniform bool u_hasHeightMap;
uniform bool u_useDisplacement;
uniform float u_displacementStrength;

varying vec3 v_worldPos;
varying vec3 v_normal;
varying vec2 v_uv;

void main() {
    vec2 tiledUv = a_uv * u_tiling;
    vec3 position = a_position;
    if (u_hasHeightMap && u_useDisplacement) {
        float h = texture2D(u_heightMap, tiledUv).r;
        position += a_normal * ((h - 0.5) * u_displacementStrength);
    }
    vec4 world = u_model * vec4(position, 1.0);
    v_worldPos = world.xyz;
    v_normal = normalize(u_normalMatrix * a_normal);
    v_uv = tiledUv;
    gl_Position = u_mvp * vec4(position, 1.0);
}
"""


FRAGMENT_SHADER = """
#ifdef GL_ES
precision highp float;
#endif

uniform sampler2D u_baseMap;
uniform sampler2D u_normalMap;
uniform sampler2D u_roughnessMap;
uniform sampler2D u_metalnessMap;
uniform sampler2D u_aoMap;
uniform sampler2D u_alphaMap;
uniform sampler2D u_emissiveMap;
uniform sampler2D u_heightMap;

uniform bool u_hasBaseMap;
uniform bool u_hasNormalMap;
uniform bool u_hasRoughnessMap;
uniform bool u_hasMetalnessMap;
uniform bool u_hasAoMap;
uniform bool u_hasAlphaMap;
uniform bool u_hasEmissiveMap;
uniform bool u_hasHeightMap;
uniform bool u_uvChecker;
uniform bool u_triplanar;
uniform int u_isolate;

uniform vec3 u_cameraPos;
uniform vec3 u_lightDir;
uniform vec3 u_lightColor;
uniform vec3 u_envColor;
uniform float u_envIntensity;
uniform float u_exposure;
uniform float u_time;

varying vec3 v_worldPos;
varying vec3 v_normal;
varying vec2 v_uv;

const float PI = 3.14159265359;

float saturate(float v) { return clamp(v, 0.0, 1.0); }
vec3 saturate(vec3 v) { return clamp(v, 0.0, 1.0); }

vec3 srgbToLinear(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

vec3 linearToSrgb(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2));
}

vec3 acesTonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float dGgx(float nDotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (nDotH * nDotH) * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 0.0001);
}

float gSmith(float nDotV, float nDotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float gv = nDotV / (nDotV * (1.0 - k) + k);
    float gl = nDotL / (nDotL * (1.0 - k) + k);
    return gv * gl;
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

vec3 checker(vec2 uv) {
    vec2 grid = floor(fract(uv) * 8.0);
    float c = mod(grid.x + grid.y, 2.0);
    return mix(vec3(0.18), vec3(0.82), c);
}

vec3 triplanarBase(vec3 n) {
    vec3 blend = pow(abs(n), vec3(4.0));
    blend /= max(blend.x + blend.y + blend.z, 0.0001);
    vec3 x = srgbToLinear(texture2D(u_baseMap, v_worldPos.yz).rgb);
    vec3 y = srgbToLinear(texture2D(u_baseMap, v_worldPos.xz).rgb);
    vec3 z = srgbToLinear(texture2D(u_baseMap, v_worldPos.xy).rgb);
    return x * blend.x + y * blend.y + z * blend.z;
}

vec3 mapNormal(vec3 n) {
    if (!u_hasNormalMap) {
        return normalize(n);
    }
    vec3 tangentNormal = texture2D(u_normalMap, v_uv).xyz * 2.0 - 1.0;
    vec3 q1 = dFdx(v_worldPos);
    vec3 q2 = dFdy(v_worldPos);
    vec2 st1 = dFdx(v_uv);
    vec2 st2 = dFdy(v_uv);
    vec3 t = normalize(q1 * st2.t - q2 * st1.t);
    vec3 b = normalize(cross(n, t));
    mat3 tbn = mat3(t, b, n);
    return normalize(tbn * tangentNormal);
}

void main() {
    vec3 geomN = normalize(v_normal);
    vec3 n = mapNormal(geomN);
    vec3 v = normalize(u_cameraPos - v_worldPos);
    vec3 l = normalize(-u_lightDir);
    vec3 h = normalize(v + l);

    vec3 base = vec3(0.72);
    if (u_uvChecker) {
        base = checker(v_uv);
    } else if (u_hasBaseMap) {
        base = u_triplanar ? triplanarBase(geomN) : srgbToLinear(texture2D(u_baseMap, v_uv).rgb);
    }

    float roughness = u_hasRoughnessMap ? texture2D(u_roughnessMap, v_uv).r : 0.48;
    roughness = clamp(roughness, 0.045, 1.0);
    float metalness = u_hasMetalnessMap ? texture2D(u_metalnessMap, v_uv).r : 0.0;
    float ao = u_hasAoMap ? texture2D(u_aoMap, v_uv).r : 1.0;
    float alpha = u_hasAlphaMap ? texture2D(u_alphaMap, v_uv).r : 1.0;
    vec3 emissive = u_hasEmissiveMap ? srgbToLinear(texture2D(u_emissiveMap, v_uv).rgb) : vec3(0.0);
    float height = u_hasHeightMap ? texture2D(u_heightMap, v_uv).r : 0.5;

    if (u_isolate == 1) { gl_FragColor = vec4(linearToSrgb(base), 1.0); return; }
    if (u_isolate == 2) { gl_FragColor = vec4(n * 0.5 + 0.5, 1.0); return; }
    if (u_isolate == 3) { gl_FragColor = vec4(vec3(roughness), 1.0); return; }
    if (u_isolate == 4) { gl_FragColor = vec4(vec3(metalness), 1.0); return; }
    if (u_isolate == 5) { gl_FragColor = vec4(vec3(ao), 1.0); return; }
    if (u_isolate == 6) { gl_FragColor = vec4(vec3(height), 1.0); return; }
    if (u_isolate == 7) { gl_FragColor = vec4(vec3(alpha), 1.0); return; }
    if (u_isolate == 8) { gl_FragColor = vec4(linearToSrgb(emissive), 1.0); return; }

    float nDotL = saturate(dot(n, l));
    float nDotV = saturate(dot(n, v));
    float nDotH = saturate(dot(n, h));
    float hDotV = saturate(dot(h, v));

    vec3 f0 = mix(vec3(0.04), base, metalness);
    vec3 f = fresnelSchlick(hDotV, f0);
    float d = dGgx(nDotH, roughness);
    float g = gSmith(nDotV, nDotL, roughness);
    vec3 spec = (d * g * f) / max(4.0 * nDotV * nDotL, 0.0001);
    vec3 kd = (1.0 - f) * (1.0 - metalness);
    vec3 direct = (kd * base / PI + spec) * u_lightColor * nDotL;

    float horizon = pow(1.0 - saturate(geomN.y * 0.5 + 0.5), 2.0);
    vec3 env = base * u_envColor * u_envIntensity * (0.22 + 0.34 * ao);
    vec3 rim = fresnelSchlick(nDotV, f0) * u_envColor * u_envIntensity * horizon * (1.0 - roughness * 0.65);
    vec3 color = (direct + env + rim) * ao + emissive;
    color = acesTonemap(color * u_exposure);
    gl_FragColor = vec4(linearToSrgb(color), alpha);
}
"""


class PBRViewport(QOpenGLWidget):
    """A persistent realtime PBR material viewport."""

    statsChanged = pyqtSignal(str)
    loadingChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._program: QOpenGLShaderProgram | None = None
        self._vbo: QOpenGLBuffer | None = None
        self._funcs = None
        self._cleaned_up = False
        self._mesh_name = "Sphere"
        self._mesh = build_mesh(self._mesh_name)
        self._textures: dict[str, TextureRecord] = {}
        self._pending_images: dict[str, QImage] = {}
        self._enabled = {name: True for name in CHANNELS}
        self._tiling = (1.0, 1.0)
        self._isolate = 0
        self._wireframe = False
        self._uv_checker = False
        self._triplanar = False
        self._tessellation = False
        self._displacement_strength = 0.08
        self._exposure = 1.0
        self._env_intensity = 1.0
        self._env_rotation = 0.0
        self._hdri = "Studio"

        self._yaw = math.radians(35.0)
        self._pitch = math.radians(18.0)
        self._distance = 4.2
        self._target_yaw = self._yaw
        self._target_pitch = self._pitch
        self._target_distance = self._distance
        self._target = QVector3D(0.0, 0.0, 0.0)
        self._last_pos: QPoint | None = None
        self._drag_mode = ""
        self._camera_pos = QVector3D(0.0, 0.0, 4.2)

        self._last_frame_time = time.perf_counter()
        self._fps_time = self._last_frame_time
        self._fps_frames = 0
        self._fps = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_material_map(self, name: str, image: np.ndarray | QImage | None):
        if name == "Displacement":
            name = "Height"
        if name not in CHANNELS or image is None:
            return
        qimg = image if isinstance(image, QImage) else numpy_to_qimage(image)
        if qimg.isNull():
            return
        self._pending_images[name] = qimg
        self.loadingChanged.emit(True)
        self.update()

    def set_channel_enabled(self, name: str, enabled: bool):
        if name in self._enabled:
            self._enabled[name] = enabled
            self.update()

    def isolate_channel(self, name: str | None):
        order = ["Base Color", "Normal", "Roughness", "Metallic", "AO", "Height", "Opacity", "Emissive"]
        self._isolate = 0 if not name else (order.index(name) + 1 if name in order else 0)
        self.update()

    def set_mesh(self, name: str):
        if name == self._mesh_name:
            return
        self._mesh_name = name
        self._mesh = build_mesh(name)
        if self._vbo is not None:
            self.makeCurrent()
            self._upload_mesh()
            self.doneCurrent()
        self.update()

    def set_tiling(self, value: int | float):
        self._tiling = (float(value), float(value))
        self.update()

    def set_wireframe(self, enabled: bool):
        self._wireframe = enabled
        self.update()

    def set_uv_checker(self, enabled: bool):
        self._uv_checker = enabled
        self.update()

    def set_triplanar(self, enabled: bool):
        self._triplanar = enabled
        self.update()

    def set_tessellation(self, enabled: bool):
        self._tessellation = enabled
        self.update()

    def set_displacement_strength(self, value: int | float):
        self._displacement_strength = float(value) / 100.0
        self.update()

    def set_hdri(self, name: str):
        self._hdri = name
        self.update()

    def set_exposure(self, value: int | float):
        self._exposure = max(0.05, float(value) / 100.0)
        self.update()

    def set_environment_intensity(self, value: int | float):
        self._env_intensity = max(0.0, float(value) / 100.0)
        self.update()

    def set_environment_rotation(self, value: int | float):
        self._env_rotation = math.radians(float(value))
        self.update()

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._timer.stop()
        try:
            if self.context() is not None and self.context().isValid():
                self.makeCurrent()
                for record in self._textures.values():
                    record.texture.destroy()
                self._textures.clear()
                self._pending_images.clear()
                if self._vbo is not None:
                    self._vbo.destroy()
                    self._vbo = None
                if self._program is not None:
                    self._program.removeAllShaders()
                    self._program = None
                self.doneCurrent()
        except Exception as exc:
            log_exception(logger, "OpenGL viewport cleanup failed", exc)

    def initializeGL(self):
        profile = QOpenGLVersionProfile()
        profile.setVersion(2, 1)
        self._funcs = QOpenGLVersionFunctionsFactory.get(profile, self.context())
        if self._funcs is None:
            raise RuntimeError("OpenGL 2.1 functions are unavailable for the material viewport.")
        self._funcs.initializeOpenGLFunctions()
        self._funcs.glEnable(GL_DEPTH_TEST)
        self._funcs.glEnable(GL_CULL_FACE)
        self._funcs.glEnable(GL_BLEND)
        self._funcs.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._program = QOpenGLShaderProgram(self)
        if not self._program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER):
            raise RuntimeError(self._program.log())
        if not self._program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT_SHADER):
            raise RuntimeError(self._program.log())
        if not self._program.link():
            raise RuntimeError(self._program.log())

        self._vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._vbo.create()
        self._upload_mesh()
        self._sync_pending_textures()

    def resizeGL(self, width: int, height: int):
        if self._funcs:
            self._funcs.glViewport(0, 0, max(1, width), max(1, height))

    def paintGL(self):
        if self._program is None or self._funcs is None:
            return
        self._sync_pending_textures()
        bg = self._background_color()
        self._funcs.glClearColor(bg[0], bg[1], bg[2], 1.0)
        self._funcs.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        try:
            self._funcs.glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self._wireframe else GL_FILL)
        except Exception as exc:
            logger.debug("Wireframe mode unavailable: %s", exc)

        self._program.bind()
        self._bind_uniforms()
        self._bind_textures()

        self._bind_vertex_buffer()
        self._funcs.glDrawArrays(GL_TRIANGLES, 0, self._mesh.vertex_count)
        if self._vbo is not None:
            self._vbo.release()
        self._program.release()

        try:
            self._funcs.glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        except Exception as exc:
            logger.debug("Polygon mode reset unavailable: %s", exc)
        self._update_stats()

    def mousePressEvent(self, event: QMouseEvent):
        self._last_pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = "orbit"
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._drag_mode = "pan"

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_pos is None:
            return
        pos = event.position().toPoint()
        delta = pos - self._last_pos
        self._last_pos = pos
        if self._drag_mode == "orbit":
            self._target_yaw -= delta.x() * 0.0065
            self._target_pitch = max(math.radians(-82), min(math.radians(82), self._target_pitch + delta.y() * 0.0048))
        elif self._drag_mode == "pan":
            scale = self._distance * 0.0018
            right = QVector3D(math.cos(self._yaw), 0.0, -math.sin(self._yaw))
            up = QVector3D(0.0, 1.0, 0.0)
            self._target += right * (-delta.x() * scale) + up * (delta.y() * scale)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_mode = ""
        self._last_pos = None

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self._target_distance = max(1.0, min(20.0, self._target_distance * (0.88 ** steps)))

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self._target = QVector3D(0.0, 0.0, 0.0)
        self._target_distance = 4.2
        self._target_yaw = math.radians(35.0)
        self._target_pitch = math.radians(18.0)

    def _tick(self):
        damping = 0.14
        old_yaw = self._yaw
        old_pitch = self._pitch
        old_distance = self._distance
        self._yaw += (self._target_yaw - self._yaw) * damping
        self._pitch += (self._target_pitch - self._pitch) * damping
        self._distance += (self._target_distance - self._distance) * damping
        moving = (
            abs(self._yaw - old_yaw) > 0.0001
            or abs(self._pitch - old_pitch) > 0.0001
            or abs(self._distance - old_distance) > 0.0001
        )
        if moving or self._pending_images:
            self.update()

    def _upload_mesh(self):
        if self._vbo is None:
            return
        self._vbo.bind()
        data = self._mesh.vertices
        self._vbo.allocate(data.tobytes(), data.nbytes)
        self._vbo.release()

    def _bind_vertex_buffer(self):
        if self._program is None or self._vbo is None:
            return
        self._vbo.bind()
        stride = 8 * 4
        for name, offset, size in [
            ("a_position", 0, 3),
            ("a_normal", 12, 3),
            ("a_uv", 24, 2),
        ]:
            self._program.enableAttributeArray(name)
            self._program.setAttributeBuffer(name, GL_FLOAT, offset, size, stride)

    def _sync_pending_textures(self):
        if not self._pending_images:
            return
        for name, image in list(self._pending_images.items()):
            try:
                old = self._textures.pop(name, None)
                if old is not None:
                    old.texture.destroy()
                qimg = image.convertToFormat(QImage.Format.Format_RGBA8888)
                texture = QOpenGLTexture(qimg)
                texture.setWrapMode(QOpenGLTexture.WrapMode.Repeat)
                texture.setMinificationFilter(QOpenGLTexture.Filter.LinearMipMapLinear)
                texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
                try:
                    texture.generateMipMaps()
                except Exception as exc:
                    logger.debug("Mipmap generation skipped for %s: %s", name, exc)
                bytes_estimate = qimg.width() * qimg.height() * 4
                self._textures[name] = TextureRecord(texture, qimg.width(), qimg.height(), bytes_estimate)
                self._pending_images.pop(name, None)
            except Exception as exc:
                log_exception(logger, f"Failed to upload texture map {name}", exc)
                self._pending_images.pop(name, None)
        self.loadingChanged.emit(False)

    def _bind_uniforms(self):
        projection = QMatrix4x4()
        aspect = max(1.0, self.width()) / max(1.0, self.height())
        projection.perspective(38.0, aspect, 0.05, 80.0)
        view = QMatrix4x4()
        cp = math.cos(self._pitch)
        self._camera_pos = self._target + QVector3D(
            self._distance * cp * math.sin(self._yaw),
            self._distance * math.sin(self._pitch),
            self._distance * cp * math.cos(self._yaw),
        )
        view.lookAt(self._camera_pos, self._target, QVector3D(0, 1, 0))
        model = QMatrix4x4()
        if self._mesh_name == "Wall":
            model.translate(0.0, 0.0, 0.0)
        mvp = projection * view * model
        normal_matrix = model.normalMatrix()
        self._program.setUniformValue("u_mvp", mvp)
        self._program.setUniformValue("u_model", model)
        self._program.setUniformValue("u_normalMatrix", normal_matrix)
        self._program.setUniformValue("u_tiling", self._tiling[0], self._tiling[1])
        self._program.setUniformValue("u_cameraPos", self._camera_pos)
        light_dir, light_color, env_color = self._lighting()
        self._program.setUniformValue("u_lightDir", light_dir)
        self._program.setUniformValue("u_lightColor", light_color)
        self._program.setUniformValue("u_envColor", env_color)
        self._program.setUniformValue("u_envIntensity", float(self._env_intensity))
        self._program.setUniformValue("u_exposure", float(self._exposure))
        self._program.setUniformValue("u_time", float(time.perf_counter()))
        self._program.setUniformValue("u_uvChecker", self._uv_checker)
        self._program.setUniformValue("u_triplanar", self._triplanar)
        self._program.setUniformValue("u_useDisplacement", self._tessellation)
        self._program.setUniformValue("u_displacementStrength", self._displacement_strength)
        self._program.setUniformValue("u_isolate", int(self._isolate))

        for channel, meta in CHANNELS.items():
            has_map = channel in self._textures and self._enabled.get(channel, True)
            if channel == "Displacement":
                continue
            self._program.setUniformValue(meta["flag"], bool(has_map))
            self._program.setUniformValue(meta["uniform"], int(meta["unit"]))

    def _bind_textures(self):
        if self._funcs is None:
            return
        bound_units = set()
        for channel, meta in CHANNELS.items():
            canonical = "Height" if channel == "Displacement" else channel
            if canonical in bound_units:
                continue
            record = self._textures.get(canonical)
            if record is None or not self._enabled.get(canonical, True):
                continue
            unit = int(meta["unit"])
            self._funcs.glActiveTexture(GL_TEXTURE0 + unit)
            record.texture.bind(unit)
            bound_units.add(canonical)

    def _lighting(self):
        presets = {
            "Studio": ((-0.45, -0.72, -0.54), (5.0, 4.75, 4.35), (0.92, 0.96, 1.0)),
            "Outdoor": ((-0.30, -0.88, -0.35), (5.9, 5.65, 5.2), (0.78, 0.88, 1.0)),
            "Archviz": ((-0.72, -0.50, -0.48), (4.6, 4.35, 3.9), (1.0, 0.92, 0.82)),
            "Neutral Gray": ((-0.40, -0.68, -0.62), (4.3, 4.3, 4.3), (0.86, 0.86, 0.86)),
        }
        direction, light, env = presets.get(self._hdri, presets["Studio"])
        x, y, z = direction
        ca = math.cos(self._env_rotation)
        sa = math.sin(self._env_rotation)
        rotated = QVector3D(x * ca - z * sa, y, x * sa + z * ca).normalized()
        return rotated, QVector3D(*light), QVector3D(*env)

    def _background_color(self):
        colors = {
            "Studio": (0.025, 0.028, 0.038),
            "Outdoor": (0.035, 0.045, 0.060),
            "Archviz": (0.048, 0.041, 0.034),
            "Neutral Gray": (0.040, 0.040, 0.042),
        }
        return colors.get(self._hdri, colors["Studio"])

    def _update_stats(self):
        now = time.perf_counter()
        self._fps_frames += 1
        if now - self._fps_time >= 0.35:
            self._fps = self._fps_frames / (now - self._fps_time)
            self._fps_frames = 0
            self._fps_time = now
            texture_bytes = sum(record.bytes_estimate for record in self._textures.values())
            res = "No texture"
            if "Base Color" in self._textures:
                record = self._textures["Base Color"]
                res = f"{record.width} x {record.height}"
            mem_mb = texture_bytes / (1024 * 1024)
            mode = "Isolate" if self._isolate else "PBR"
            self.statsChanged.emit(
                f"FPS {self._fps:04.1f}   GPU tex {mem_mb:0.1f} MB   {res}   {mode}   {self._mesh_name}"
            )
