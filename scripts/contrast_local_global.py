# -*- coding: utf-8 -*-
"""
==========================================================================
 SCRIPT: Contraste Iluminación Local vs. Global
 Proyecto: Modelos Globales de Iluminación — Computación Visual (UNMSM)
==========================================================================

Genera una imagen combinada lado a lado que contrasta la iluminación
local (solo luz directa) con la iluminación global (path tracing),
con etiquetas descriptivas. Sirve para la Figura 1 del informe.

Uso:
  python scripts/contrast_local_global.py
  python scripts/contrast_local_global.py --width 320 --height 320 --spp 64
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image, ImageDraw, ImageFont
from src.config import RenderConfig
from src.core.camera import Camera
from src.core.scene import build_cornell_box
from src.core.tonemap import save_image, reinhard_tonemap
from src.renderers.local import LocalRenderer
from src.renderers.path_tracing import PathTracingRenderer


def main():
    parser = argparse.ArgumentParser(
        description="Contraste iluminación local vs. global."
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--spp", type=int, default=64)
    args = parser.parse_args()

    config = RenderConfig(width=args.width, height=args.height, spp=args.spp)
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    objetos, light_info = build_cornell_box(use_mirror=True, use_glass=False)
    camera = Camera(width=config.width, height=config.height)

    # Renderizar local
    print(">>> Renderizando iluminación LOCAL ...")
    local_renderer = LocalRenderer(config)
    t0 = time.perf_counter()
    hdr_local = local_renderer.render(objetos, light_info, camera)
    t_local = time.perf_counter() - t0
    print(f"    Tiempo: {t_local:.2f} s")
    save_image(hdr_local, os.path.join(output_dir, "contrast_local.png"),
               config.exposure, config.gamma)

    # Renderizar global
    print(">>> Renderizando iluminación GLOBAL (path tracing) ...")
    pt_renderer = PathTracingRenderer(config)
    t0 = time.perf_counter()
    hdr_global = pt_renderer.render(objetos, light_info, camera)
    t_global = time.perf_counter() - t0
    print(f"    Tiempo: {t_global:.2f} s")
    save_image(hdr_global, os.path.join(output_dir, "contrast_global.png"),
               config.exposure, config.gamma)

    # Componer imagen lado a lado
    ldr_local = Image.fromarray(reinhard_tonemap(hdr_local, config.exposure, config.gamma))
    ldr_global = Image.fromarray(reinhard_tonemap(hdr_global, config.exposure, config.gamma))
    iw, ih = ldr_local.size
    margin = 6
    label_h = 28

    combined_w = 2 * iw + 3 * margin
    combined_h = ih + label_h + 2 * margin

    combined = Image.new("RGB", (combined_w, combined_h), (30, 30, 30))
    draw = ImageDraw.Draw(combined)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Pegar imágenes
    combined.paste(ldr_local, (margin, margin))
    combined.paste(ldr_global, (2 * margin + iw, margin))

    # Etiquetas
    for idx, (label, t_render) in enumerate([
        ("Iluminación Local", t_local),
        ("Iluminación Global (Path Tracing)", t_global),
    ]):
        x = margin + idx * (iw + margin)
        y = margin + ih + 4
        text = f"{label}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        draw.text((x + (iw - tw) // 2, y), text,
                  fill=(220, 220, 220), font=font)

    combined_path = os.path.join(output_dir, "contrast_local_vs_global.png")
    combined.save(combined_path)
    print(f"\n   -> Comparación guardada: {combined_path}")


if __name__ == "__main__":
    main()
