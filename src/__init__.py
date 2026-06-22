# -*- coding: utf-8 -*-
"""
Paquete src — Modelos Globales de Iluminación.
"""

from .config import RenderConfig
from .core import *
from .renderers import (
    LocalRenderer,
    RayTracingRenderer,
    PathTracingRenderer,
    RadiosityRenderer,
    PhotonMappingRenderer,
    RealtimeRenderer,
    RENDERER_MAP,
)
from .metrics import compute_psnr, compute_mse, compute_ssim, metrics_table
