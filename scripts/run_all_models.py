# -*- coding: utf-8 -*-
"""
==========================================================================
 SCRIPT: Ejecutar todos los modelos y generar tabla de tiempos
 Proyecto: Modelos Globales de Iluminación — Computación Visual (UNMSM)
==========================================================================

Ejecuta los 6 renderers (local, ray tracing, path tracing, radiosidad,
mapeo de fotones, tiempo real) sobre la Caja de Cornell y guarda las
imágenes en outputs/.

Uso:
  python scripts/run_all_models.py
  python scripts/run_all_models.py --width 160 --height 160 --spp 16
"""

import sys
import os
import time
import argparse

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import RenderConfig
from src.core.camera import Camera
from src.core.scene import build_cornell_box
from src.core.tonemap import save_image, reinhard_tonemap
from src.renderers import RENDERER_MAP
from src.metrics import compute_psnr, compute_ssim


def main():
    parser = argparse.ArgumentParser(description="Ejecuta todos los modelos.")
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--spp", type=int, default=128)
    args = parser.parse_args()

    config = RenderConfig(width=args.width, height=args.height, spp=args.spp)
    config.ensure_output_dir()

    objetos, light_info = build_cornell_box(use_mirror=True, use_glass=False)
    camera = Camera(width=config.width, height=config.height)

    results_hdr = {}
    results_ldr = {}
    times = {}

    print("=" * 70)
    print(" EJECUTANDO TODOS LOS MODELOS DE ILUMINACIÓN")
    print("=" * 70)

    for name, cls in RENDERER_MAP.items():
        renderer = cls(config)
        print(f"\n>>> {renderer.name} ...")
        t0 = time.perf_counter()
        hdr = renderer.render(objetos, light_info, camera)
        elapsed = time.perf_counter() - t0

        filename = os.path.join(config.output_dir, f"cornell_{name}.png")
        save_image(hdr, filename, config.exposure, config.gamma)

        ldr = reinhard_tonemap(hdr, config.exposure, config.gamma)
        results_hdr[name] = hdr
        results_ldr[name] = ldr
        times[name] = elapsed
        print(f"    Tiempo: {elapsed:.2f} s")

    # Tabla de resultados
    print(f"\n{'=' * 70}")
    print(" RESULTADOS")
    print(f"{'=' * 70}")
    print(f"  {'Modelo':<30} {'Tiempo (s)':>10} {'PSNR (dB)':>10} {'SSIM':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")

    ref_key = "pathtracing"
    ref_ldr = results_ldr.get(ref_key)

    for name in RENDERER_MAP:
        renderer = RENDERER_MAP[name](config)
        t = times[name]
        if ref_ldr is not None and name != ref_key:
            psnr = compute_psnr(results_ldr[name], ref_ldr)
            ssim = compute_ssim(results_ldr[name], ref_ldr)
            print(f"  {renderer.name:<30} {t:>10.2f} {psnr:>10.2f} {ssim:>8.3f}")
        else:
            label = "(ref)" if name == ref_key else ""
            print(f"  {renderer.name:<30} {t:>10.2f} {'∞':>10} {'1.000':>8}  {label}")

    print(f"\n  Imágenes guardadas en: {config.output_dir}/")
    print()


if __name__ == "__main__":
    main()
