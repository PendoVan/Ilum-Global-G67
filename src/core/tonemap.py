# -*- coding: utf-8 -*-
"""
Mapeo de tono y guardado de imágenes.

Convierte imágenes HDR (radiancia en punto flotante) a LDR (8-bit)
mediante tone mapping de Reinhard y corrección gamma.
"""

import numpy as np
from PIL import Image


def reinhard_tonemap(hdr, exposure=1.0, gamma=2.2):
    """
    Mapeo de tono de Reinhard + corrección gamma.

    hdr : ndarray (H, W, 3) float64 — radiancia HDR
    Devuelve ndarray (H, W, 3) uint8.
    """
    ldr = hdr * exposure
    ldr = ldr / (1.0 + ldr)                            # Reinhard
    ldr = np.clip(ldr, 0.0, 1.0) ** (1.0 / gamma)     # gamma
    return (ldr * 255.0 + 0.5).astype(np.uint8)


def save_image(hdr, filename, exposure=1.0, gamma=2.2):
    """Guarda una imagen HDR como PNG con tone mapping."""
    img_uint8 = reinhard_tonemap(hdr, exposure, gamma)
    Image.fromarray(img_uint8, mode="RGB").save(filename)
    print(f"   -> Imagen guardada: {filename}")


def hdr_to_pil(hdr, exposure=1.0, gamma=2.2):
    """Convierte HDR a PIL Image (útil para composición de grillas)."""
    img_uint8 = reinhard_tonemap(hdr, exposure, gamma)
    return Image.fromarray(img_uint8, mode="RGB")
