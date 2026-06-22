# -*- coding: utf-8 -*-
"""
Paquete renderers — implementaciones de cada modelo de iluminación.
"""

from .local import LocalRenderer
from .ray_tracing import RayTracingRenderer
from .path_tracing import PathTracingRenderer
from .radiosity import RadiosityRenderer
from .photom_mapping import PhotonMappingRenderer
from .realtime import RealtimeRenderer

__all__ = [
    "LocalRenderer",
    "RayTracingRenderer",
    "PathTracingRenderer",
    "RadiosityRenderer",
    "PhotonMappingRenderer",
    "RealtimeRenderer",
]

# Mapa nombre → clase, útil para selección desde CLI
RENDERER_MAP = {
    "local":       LocalRenderer,
    "raytracing":  RayTracingRenderer,
    "pathtracing": PathTracingRenderer,
    "radiosity":   RadiosityRenderer,
    "photon":      PhotonMappingRenderer,
    "realtime":    RealtimeRenderer,
}
