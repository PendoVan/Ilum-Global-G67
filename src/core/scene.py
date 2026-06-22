# -*- coding: utf-8 -*-
"""
Construcción de la escena y funciones de intersección.

La escena estándar es la Caja de Cornell: una caja cúbica con pared
izquierda roja, derecha verde, el resto blancas, una fuente de luz
rectangular en el techo y dos esferas interiores.
"""

import numpy as np
from .geometry import Sphere, Plane
from .materials import material, DIFFUSE, EMISSIVE, MIRROR, GLASS, pack_materials
from .vectors import normalize, dot


# Geometría de la fuente de luz rectangular
LIGHT_Y  = 0.998
LIGHT_HX = 0.33
LIGHT_HZ = 0.33


def build_cornell_box(use_mirror=True, use_glass=False):
    """
    Construye la Caja de Cornell.

    Parámetros
    ----------
    use_mirror : bool — la segunda esfera es espejo (True) o difusa (False)
    use_glass  : bool — si True, la segunda esfera es de vidrio (ignora use_mirror)

    Devuelve (objetos, light_info).
    """
    rojo   = material(DIFFUSE,  albedo=(0.75, 0.20, 0.20))
    verde  = material(DIFFUSE,  albedo=(0.20, 0.75, 0.25))
    blanco = material(DIFFUSE,  albedo=(0.78, 0.78, 0.78))
    luz    = material(EMISSIVE, emission=(7.0, 6.5, 5.5))
    espejo = material(MIRROR,   albedo=(0.95, 0.95, 0.95))
    vidrio = material(GLASS,    albedo=(0.98, 0.98, 0.98), ior=1.5)
    mate   = material(DIFFUSE,  albedo=(0.80, 0.80, 0.82))

    # Elegir material de la segunda esfera
    if use_glass:
        mat2 = vidrio
    elif use_mirror:
        mat2 = espejo
    else:
        mat2 = mate

    objetos = [
        # Paredes (planos infinitos)
        Plane(normal=( 1, 0, 0), offset=-1, mat=rojo),     # izquierda  x=-1
        Plane(normal=(-1, 0, 0), offset=-1, mat=verde),    # derecha    x=+1
        Plane(normal=( 0, 1, 0), offset=-1, mat=blanco),   # piso       y=-1
        Plane(normal=( 0,-1, 0), offset=-1, mat=blanco),   # techo      y=+1
        Plane(normal=( 0, 0,-1), offset=-1, mat=blanco),   # fondo      z=+1
        Plane(normal=( 0, 0, 1), offset=-1, mat=blanco),   # frontal    z=-1

        # Fuente de luz rectangular (justo bajo el techo)
        Plane(normal=(0, -1, 0), offset=LIGHT_Y, mat=luz,
              bounds=[(0, -LIGHT_HX, LIGHT_HX), (2, -LIGHT_HZ, LIGHT_HZ)]),

        # Esferas interiores
        Sphere(center=(-0.40, -0.62, 0.35), radius=0.38, mat=mate),
        Sphere(center=( 0.45, -0.65, -0.10), radius=0.35, mat=mat2),
    ]

    light_info = {
        "index": 6,
        "y": LIGHT_Y,
        "hx": LIGHT_HX, "hz": LIGHT_HZ,
        "normal": np.array([0.0, -1.0, 0.0]),
        "emission": luz["emission"].copy(),
        "area": (2 * LIGHT_HX) * (2 * LIGHT_HZ),
    }

    return objetos, light_info


def scene_intersect(objetos, O, D, eps=1e-4):
    """
    Para N rayos, encuentra el objeto más cercano impactado.

    Devuelve (nearest_t, nearest_obj).  nearest_obj = -1 si no hay impacto.
    """
    N = O.shape[0]
    nearest_t = np.full(N, np.inf)
    nearest_obj = np.full(N, -1, dtype=np.int64)
    for idx, obj in enumerate(objetos):
        t = obj.intersect(O, D, eps)
        closer = t < nearest_t
        nearest_t[closer] = t[closer]
        nearest_obj[closer] = idx
    return nearest_t, nearest_obj


def gather_normals(objetos, nearest_obj, P):
    """Calcula la normal de la superficie en cada punto de impacto."""
    Nn = np.zeros_like(P)
    for idx, obj in enumerate(objetos):
        mask = (nearest_obj == idx)
        if not np.any(mask):
            continue
        Nn[mask] = obj.normal_at(P[mask])
    return Nn
