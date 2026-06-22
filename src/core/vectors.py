# -*- coding: utf-8 -*-
"""
Operaciones de álgebra vectorial sobre arreglos (N, 3).

Todas las funciones trabajan de forma vectorizada con NumPy para procesar
miles de rayos en paralelo sin bucles Python.
"""

import numpy as np


def normalize(v):
    """Normaliza un arreglo de vectores (N,3) a longitud unitaria."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return v / n


def dot(a, b):
    """Producto punto fila a fila de dos arreglos (N,3) → (N,)."""
    return np.sum(a * b, axis=-1)


def cross(a, b):
    """Producto cruz fila a fila de dos arreglos (N,3) → (N,3)."""
    return np.cross(a, b)


def length(v):
    """Longitud de cada vector en un arreglo (N,3) → (N,)."""
    return np.linalg.norm(v, axis=-1)


def reflect(D, n):
    """Reflexión especular perfecta: D - 2(D·n)n."""
    return D - 2.0 * dot(D, n)[:, None] * n


def refract(D, n, eta):
    """
    Refracción según la ley de Snell.

    D   : dirección incidente normalizada (N,3)
    n   : normal de la superficie (N,3), apuntando hacia el lado del rayo
    eta : relación de índices n1/n2 (N,)

    Devuelve (refracted_dir, valid_mask).
    Donde valid_mask es False hay reflexión total interna.
    """
    cos_i = -dot(D, n)                                    # (N,)
    sin2_t = eta**2 * (1.0 - cos_i**2)                    # (N,)
    valid = sin2_t <= 1.0
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))       # (N,)
    t = eta[:, None] * D + (eta * cos_i - cos_t)[:, None] * n
    return normalize(t), valid


def fresnel_schlick(cos_theta, ior):
    """
    Aproximación de Schlick para la reflectancia de Fresnel.

    cos_theta : coseno del ángulo de incidencia (N,)
    ior       : índice de refracción del medio interior (float)

    Devuelve la reflectancia R (N,).
    """
    r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
    return r0 + (1.0 - r0) * (1.0 - np.clip(cos_theta, 0, 1)) ** 5
