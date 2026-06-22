# -*- coding: utf-8 -*-
"""
==========================================================================
 SCRIPT: Grilla comparativa de todos los modelos
 Proyecto: Modelos Globales de Iluminación — Computación Visual (UNMSM)
==========================================================================

Carga las imágenes generadas por run_all_models.py y compone una grilla
2x3 con etiquetas para incluir en el informe o la presentación.

Uso:
  python scripts/comparison_grid.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image, ImageDraw, ImageFont
from src.renderers import RENDERER_MAP
from src.config import RenderConfig


def main():
    cfg = RenderConfig()
    output_dir = cfg.output_dir

    models = list(RENDERER_MAP.keys())
    labels = {
        "local": "Local (Directa)",
        "raytracing": "Ray Tracing (Whitted)",
        "pathtracing": "Path Tracing (Monte Carlo)",
        "radiosity": "Radiosidad",
        "photon": "Mapeo de Fotones",
        "realtime": "Tiempo Real (SSAO+VPL)",
    }

    images = []
    names = []
    for model in models:
        path = os.path.join(output_dir, f"cornell_{model}.png")
        if os.path.exists(path):
            images.append(Image.open(path))
            names.append(labels.get(model, model))
        else:
            print(f"   ⚠ No se encontró: {path}")

    if not images:
        print("No hay imágenes para componer. Ejecuta run_all_models.py primero.")
        return

    # Componer grilla 2x3 (o ajustar según el número de imágenes)
    n = len(images)
    cols = 3
    rows = (n + cols - 1) // cols
    iw, ih = images[0].size
    label_h = 30
    margin = 4
    grid_w = cols * (iw + margin) + margin
    grid_h = rows * (ih + label_h + margin) + margin

    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    # Intentar usar una fuente legible
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, (img, name) in enumerate(zip(images, names)):
        row, col = divmod(idx, cols)
        x = margin + col * (iw + margin)
        y = margin + row * (ih + label_h + margin)
        grid.paste(img, (x, y))
        # Etiqueta centrada debajo
        text_bbox = draw.textbbox((0, 0), name, font=font)
        tw = text_bbox[2] - text_bbox[0]
        tx = x + (iw - tw) // 2
        ty = y + ih + 4
        draw.text((tx, ty), name, fill=(220, 220, 220), font=font)

    grid_path = os.path.join(output_dir, "comparison_grid.png")
    grid.save(grid_path)
    print(f"   -> Grilla guardada: {grid_path}")


if __name__ == "__main__":
    main()
