# -*- coding: utf-8 -*-
"""
Paquete core — utilidades compartidas por todos los renderers.
"""

from .vectors import normalize, dot, cross, length, reflect, refract, fresnel_schlick
from .materials import DIFFUSE, EMISSIVE, MIRROR, GLASS, material, pack_materials
from .geometry import Sphere, Plane
from .camera import Camera
from .sampling import cosine_weighted_hemisphere, uniform_hemisphere, sample_light_rect
from .scene import build_cornell_box, scene_intersect, gather_normals
from .tonemap import reinhard_tonemap, save_image, hdr_to_pil

__all__ = [
    # vectors
    "normalize", "dot", "cross", "length", "reflect", "refract", "fresnel_schlick",
    # materials
    "DIFFUSE", "EMISSIVE", "MIRROR", "GLASS", "material", "pack_materials",
    # geometry
    "Sphere", "Plane",
    # camera
    "Camera",
    # sampling
    "cosine_weighted_hemisphere", "uniform_hemisphere", "sample_light_rect",
    # scene
    "build_cornell_box", "scene_intersect", "gather_normals",
    # tonemap
    "reinhard_tonemap", "save_image", "hdr_to_pil",
]
