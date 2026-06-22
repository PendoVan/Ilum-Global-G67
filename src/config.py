# -*- coding: utf-8 -*-
"""
Configuración global del proyecto.

Centraliza todos los parámetros de renderizado en un dataclass para que cada
renderer reciba la misma configuración y los scripts puedan modificarla
fácilmente desde la línea de comandos.
"""

from dataclasses import dataclass, field
import os


@dataclass
class RenderConfig:
    """Parámetros compartidos por todos los renderers."""

    # --- Resolución de imagen ---
    width: int = 320
    height: int = 320

    # --- Muestreo ---
    spp: int = 64                   # muestras por píxel (path tracing / photon gather)
    max_bounces: int = 10           # profundidad máxima de rebotes

    # --- Ruleta Rusa ---
    use_russian_roulette: bool = True
    rr_start_bounce: int = 3       # la ruleta rusa actúa desde este rebote

    # --- Escena ---
    use_mirror_sphere: bool = True  # True = esfera espejo, False = difusa
    use_glass_sphere: bool = False  # True = segunda esfera de vidrio (ray tracing)

    # --- Tone mapping ---
    exposure: float = 1.0
    gamma: float = 2.2

    # --- Iluminación local ---
    ambient_local: float = 0.18    # componente ambiente constante (solo modo local)

    # --- Radiosidad ---
    radiosity_patches_per_side: int = 12   # NxN parches por pared
    radiosity_iterations: int = 40         # iteraciones de refinamiento progresivo
    radiosity_form_factor_samples: int = 64  # rayos por parche para form factors

    # --- Photon Mapping ---
    photon_count: int = 50_000      # fotones emitidos en la primera pasada
    photon_k_nearest: int = 80      # vecinos para la estimación de densidad
    photon_caustic_count: int = 100_000   # fotones para el mapa de cáusticas
    photon_caustic_k: int = 60

    # --- Real-time (SSAO + VPLs) ---
    ssao_samples: int = 16          # muestras por píxel para SSAO
    ssao_radius: float = 0.25       # radio de muestreo en espacio de vista
    vpl_count: int = 64             # número de luces puntuales virtuales

    # --- Reproducibilidad ---
    seed: int = 2024

    # --- Salida ---
    output_dir: str = "outputs"

    # --- Constantes físicas ---
    eps: float = 1e-4               # epsilon anti-auto-intersección

    def ensure_output_dir(self):
        """Crea el directorio de salida si no existe."""
        os.makedirs(self.output_dir, exist_ok=True)
