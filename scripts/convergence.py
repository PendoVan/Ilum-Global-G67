# -*- coding: utf-8 -*-
"""
==========================================================================
 SCRIPT: Serie de convergencia del ruido (Path Tracing)
 Proyecto: Modelos Globales de Iluminación — Computación Visual (UNMSM)
==========================================================================

Renderiza la Caja de Cornell con path tracing a distintos niveles de
muestras por píxel (SPP) para mostrar cómo el ruido disminuye
progresivamente con más muestras.

Genera la secuencia para la Figura 4 del informe y calcula PSNR de cada
paso vs. la referencia de mayor SPP.

Uso:
  python scripts/convergence.py
  python scripts/convergence.py --width 160 --height 160
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
from src.renderers.path_tracing import PathTracingRenderer
from src.metrics import compute_psnr


def main():
    parser = argparse.ArgumentParser(description="Serie de convergencia.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    args = parser.parse_args()

    spp_series = [1, 4, 16, 64, 256]

    objetos, light_info = build_cornell_box(use_mirror=True, use_glass=False)

    results = {}
    times_list = {}

    print("=" * 70)
    print(" SERIE DE CONVERGENCIA — PATH TRACING")
    print("=" * 70)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    for spp in spp_series:
        config = RenderConfig(width=args.width, height=args.height, spp=spp)
        camera = Camera(width=config.width, height=config.height)
        renderer = PathTracingRenderer(config)

        print(f"\n  SPP = {spp} ...")
        t0 = time.perf_counter()
        hdr = renderer.render(objetos, light_info, camera)
        elapsed = time.perf_counter() - t0

        filename = os.path.join(output_dir, f"convergence_spp{spp:04d}.png")
        save_image(hdr, filename, config.exposure, config.gamma)

        ldr = reinhard_tonemap(hdr, config.exposure, config.gamma)
        results[spp] = ldr
        times_list[spp] = elapsed
        print(f"    Tiempo: {elapsed:.2f} s")

    # PSNR vs. referencia (mayor SPP)
    ref_spp = spp_series[-1]
    ref_ldr = results[ref_spp]

    print(f"\n{'='*70}")
    print(f"  PSNR vs. referencia (SPP={ref_spp})")
    print(f"{'='*70}")
    print(f"  {'SPP':>6} {'Tiempo (s)':>12} {'PSNR (dB)':>12}")
    print(f"  {'-'*6} {'-'*12} {'-'*12}")
    for spp in spp_series:
        t = times_list[spp]
        if spp == ref_spp:
            print(f"  {spp:>6} {t:>12.2f} {'∞ (ref)':>12}")
        else:
            psnr = compute_psnr(results[spp], ref_ldr)
            print(f"  {spp:>6} {t:>12.2f} {psnr:>12.2f}")

    # Componer tira horizontal para la figura del informe
    imgs = [Image.fromarray(results[spp]) for spp in spp_series]
    iw, ih = imgs[0].size
    margin = 4
    label_h = 24
    strip_w = len(imgs) * (iw + margin) + margin
    strip_h = ih + label_h + 2 * margin

    strip = Image.new("RGB", (strip_w, strip_h), (30, 30, 30))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, (spp, img) in enumerate(zip(spp_series, imgs)):
        x = margin + idx * (iw + margin)
        y = margin
        strip.paste(img, (x, y))
        label = f"SPP={spp}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        draw.text((x + (iw - tw) // 2, y + ih + 2), label,
                  fill=(220, 220, 220), font=font)

    strip_path = os.path.join(output_dir, "convergence_strip.png")
    strip.save(strip_path)
    print(f"\n   -> Tira de convergencia: {strip_path}")


if __name__ == "__main__":
    main()
