# -*- coding: utf-8 -*-
"""
Definición de materiales para la escena.

Cada material se describe por un tipo (kind) y propiedades ópticas. El diseño
replica las convenciones del path tracer original y añade GLASS para refracciones.
"""

import numpy as np

# Tipos de material
DIFFUSE  = 0    # Lambertiano: refleja luz por igual en todas las direcciones
EMISSIVE = 1    # Fuente de luz: emite radiancia
MIRROR   = 2    # Espejo perfecto: reflexión especular
GLASS    = 3    # Dieléctrico: refracción + reflexión (Fresnel)


def material(kind, albedo=(0, 0, 0), emission=(0, 0, 0), ior=1.5):
    """
    Crea un diccionario de material.

    Parámetros
    ----------
    kind     : int — tipo de material (DIFFUSE, EMISSIVE, MIRROR, GLASS)
    albedo   : tuple — reflectividad RGB [0,1]
    emission : tuple — radiancia emitida RGB (solo fuentes de luz)
    ior      : float — índice de refracción (solo GLASS)
    """
    return {
        "kind": kind,
        "albedo": np.array(albedo, dtype=np.float64),
        "emission": np.array(emission, dtype=np.float64),
        "ior": float(ior),
    }


def pack_materials(objects):
    """
    Empaqueta las propiedades de los materiales de todos los objetos
    en arreglos NumPy para acceso vectorizado.

    Devuelve (kinds, albedos, emissions, iors).
    """
    M = len(objects)
    kinds     = np.zeros(M, dtype=np.int64)
    albedos   = np.zeros((M, 3), dtype=np.float64)
    emissions = np.zeros((M, 3), dtype=np.float64)
    iors      = np.full(M, 1.5, dtype=np.float64)
    for i, obj in enumerate(objects):
        kinds[i]     = obj.mat["kind"]
        albedos[i]   = obj.mat["albedo"]
        emissions[i] = obj.mat["emission"]
        iors[i]      = obj.mat.get("ior", 1.5)
    return kinds, albedos, emissions, iors
