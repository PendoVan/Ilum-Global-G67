# -*- coding: utf-8 -*-
"""
==========================================================================
 PUNTO DE ENTRADA PRINCIPAL
 Proyecto: Modelos Globales de Iluminación — Computación Visual (UNMSM)
==========================================================================

Uso:
  python main.py --model all                    # Ejecuta todos los renderers
  python main.py --model pathtracing --spp 128   # Solo path tracing con 128 spp
  python main.py --model raytracing --width 640 --height 640

Modelos disponibles: local, raytracing, pathtracing, radiosity, photon, realtime, all
"""

import argparse
import time
import os
import sys

# Agregar el directorio del proyecto al path para importar src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RenderConfig
from src.core.camera import Camera
from src.core.scene import build_cornell_box
from src.core.tonemap import save_image
from src.renderers import RENDERER_MAP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Renderiza la Caja de Cornell con distintos modelos de iluminación."
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=list(RENDERER_MAP.keys()) + ["all"],
        help="Modelo de iluminación a ejecutar (default: all)"
    )
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument("--max-bounces", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def run_renderer(name, renderer_cls, config, objetos, light_info, camera):
    """Ejecuta un renderer y devuelve (hdr, tiempo_en_segundos)."""
    renderer = renderer_cls(config)
    print(f"\n{'='*70}")
    print(f"  {renderer.name}")
    print(f"  {config.width}x{config.height}px | spp={config.spp} | "
          f"rebotes={config.max_bounces}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    hdr = renderer.render(objetos, light_info, camera)
    elapsed = time.perf_counter() - t0

    print(f"  Tiempo: {elapsed:.2f} s")
    return hdr, elapsed


def main():
    args = parse_args()

    config = RenderConfig(
        width=args.width,
        height=args.height,
        spp=args.spp,
        max_bounces=args.max_bounces,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    config.ensure_output_dir()

    # Construir escena y cámara
    use_glass = (args.model in ("raytracing", "all"))
    objetos, light_info = build_cornell_box(
        use_mirror=config.use_mirror_sphere,
        use_glass=False  # mantener consistencia entre modelos
    )
    camera = Camera(
        width=config.width,
        height=config.height,
    )

    # Seleccionar renderers a ejecutar
    if args.model == "all":
        renderers_to_run = list(RENDERER_MAP.items())
    else:
        renderers_to_run = [(args.model, RENDERER_MAP[args.model])]

    results = {}
    times = {}

    for name, cls in renderers_to_run:
        hdr, elapsed = run_renderer(name, cls, config, objetos, light_info, camera)
        filename = os.path.join(config.output_dir, f"cornell_{name}.png")
        save_image(hdr, filename, config.exposure, config.gamma)
        results[name] = hdr
        times[name] = elapsed

    # Resumen de tiempos
    print(f"\n{'='*70}")
    print("  RESUMEN DE TIEMPOS")
    print(f"{'='*70}")
    print(f"  {'Modelo':<30} {'Tiempo (s)':>12}")
    print(f"  {'-'*30} {'-'*12}")
    for name, elapsed in times.items():
        renderer_name = RENDERER_MAP[name](config).name
        print(f"  {renderer_name:<30} {elapsed:>12.2f}")
    print()


if __name__ == "__main__":
    main()
