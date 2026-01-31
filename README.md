# Seamless Texture Maker

A Windows desktop application for creating perfectly seamless textures from input images.

## Features

- **Offset Mapping**: Horizontal and vertical texture wrapping
- **Edge Blending**: Smooth gradient transitions
- **Smart Inpainting**: Automatic seam removal
- **Real-time Preview**: Before/after split view
- **GPU Acceleration**: CUDA support with CPU fallback
- **Format Support**: JPG, PNG, TIFF

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Building

```bash
pyinstaller build.spec --clean
```

Output: `dist/SeamlessTextureMaker.exe`

## License

MIT License
