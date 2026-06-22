# -*- coding: utf-8 -*-
"""
Funciones de muestreo para integración de Monte Carlo.

Incluye muestreo hemisférico ponderado por coseno (para path tracing),
muestreo uniforme (para radiosidad y form factors), y muestreo de la
fuente de luz rectangular de la Caja de Cornell.
"""

import numpy as np
from .vectors import normalize, dot


def cosine_weighted_hemisphere(n, rng):
    """
    Genera direcciones aleatorias (N,3) distribuidas según cos(θ)/π
    alrededor de la normal n.

    Con esta distribución, el estimador de Monte Carlo de la integral
    difusa se simplifica: el coseno y el 1/π de la BRDF lambertiana
    se cancelan, dejando solo  albedo · L_i.
    """
    N = n.shape[0]
    u1 = rng.random(N)
    u2 = rng.random(N)
    r = np.sqrt(u2)
    phi = 2.0 * np.pi * u1

    # Base ortonormal (tangent, bitangent, n) por vector
    w = n
    aux = np.where(np.abs(w[:, 0:1]) > 0.9,
                   np.array([0.0, 1.0, 0.0]),
                   np.array([1.0, 0.0, 0.0]))
    tangent = normalize(np.cross(aux, w))
    bitangent = np.cross(w, tangent)

    d = (tangent * (np.cos(phi) * r)[:, None]
         + bitangent * (np.sin(phi) * r)[:, None]
         + w * np.sqrt(np.maximum(0.0, 1.0 - u2))[:, None])
    return normalize(d)


def uniform_hemisphere(n, rng):
    """
    Genera direcciones aleatorias uniformes en el hemisferio
    definido por la normal n.  PDF = 1/(2π).

    Útil para calcular form factors en radiosidad.
    """
    N = n.shape[0]
    u1 = rng.random(N)
    u2 = rng.random(N)
    cos_theta = u1
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    phi = 2.0 * np.pi * u2

    w = n
    aux = np.where(np.abs(w[:, 0:1]) > 0.9,
                   np.array([0.0, 1.0, 0.0]),
                   np.array([1.0, 0.0, 0.0]))
    tangent = normalize(np.cross(aux, w))
    bitangent = np.cross(w, tangent)

    d = (tangent * (np.cos(phi) * sin_theta)[:, None]
         + bitangent * (np.sin(phi) * sin_theta)[:, None]
         + w * cos_theta[:, None])
    return normalize(d)


def sample_light_rect(n_points, rng, light_info):
    """
    Muestrea n_points puntos uniformes sobre la fuente de luz
    rectangular del techo de la Caja de Cornell.

    Devuelve un arreglo (n_points, 3).
    """
    qx = rng.uniform(-light_info["hx"], light_info["hx"], n_points)
    qz = rng.uniform(-light_info["hz"], light_info["hz"], n_points)
    qy = np.full(n_points, light_info["y"])
    return np.stack([qx, qy, qz], axis=-1)
