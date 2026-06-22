# -*- coding: utf-8 -*-
"""
Cámara con generación vectorizada de rayos primarios.

Genera todos los rayos de la imagen en una sola llamada con jitter
sub-píxel para anti-aliasing por supermuestreo.
"""

import numpy as np
from .vectors import normalize


class Camera:
    """Cámara pinhole con jitter anti-aliasing."""

    def __init__(self, origin=(0.0, 0.0, -0.85), fov_scale=0.72,
                 width=320, height=320):
        self.origin = np.array(origin, dtype=np.float64)
        self.fov_scale = fov_scale
        self.width = width
        self.height = height

    def generate_primary_rays(self, rng):
        """
        Devuelve (orígenes, direcciones) de todos los rayos primarios.

        El jitter aleatorio dentro de cada píxel proporciona anti-aliasing
        natural cuando se promedian múltiples muestras.
        """
        W, H = self.width, self.height
        aspect = W / H

        jx = rng.random((H, W))
        jy = rng.random((H, W))
        xs = (np.arange(W) + jx) / W
        ys = (np.arange(H)[:, None] + jy) / H

        # Coordenadas de pantalla [-1, 1]; eje y invertido (fila 0 = arriba)
        px = (2.0 * xs - 1.0) * self.fov_scale * aspect
        py = (1.0 - 2.0 * ys) * self.fov_scale

        # La cámara mira hacia +z
        dirs = np.stack([px, py, np.ones_like(px)], axis=-1).reshape(-1, 3)
        dirs = normalize(dirs)
        origins = np.broadcast_to(self.origin, dirs.shape).copy()
        return origins, dirs
