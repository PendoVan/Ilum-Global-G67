# -*- coding: utf-8 -*-
"""
Primitivas geométricas con intersección rayo-objeto vectorizada.

Cada primitiva implementa:
  - intersect(O, D)  → t (N,)  distancias de intersección (np.inf si no hay)
  - normal_at(P)     → n (N,3) normales en los puntos de impacto
"""

import numpy as np
from .vectors import normalize, dot


class Sphere:
    """Esfera definida por centro y radio."""

    def __init__(self, center, radius, mat):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.mat = mat
        self.is_sphere = True

    def intersect(self, O, D, eps=1e-4):
        """
        Intersección rayo-esfera vectorizada.
        Resuelve |O + t·D − C|² = r²  →  t² + 2bt + c = 0
        """
        oc = O - self.center                      # (N,3)
        b  = dot(oc, D)                           # (N,)
        c  = dot(oc, oc) - self.radius * self.radius
        disc = b * b - c
        sq = np.sqrt(np.maximum(disc, 0.0))
        t0 = -b - sq                              # raíz cercana
        t1 = -b + sq                              # raíz lejana
        t = np.where(t0 > eps, t0, t1)
        valid = (disc > 0.0) & (t > eps)
        return np.where(valid, t, np.inf)

    def normal_at(self, P):
        """Normal unitaria apuntando hacia afuera."""
        return normalize(P - self.center)


class Plane:
    """
    Plano infinito o cuadrilátero acotado.

    Se define por la ecuación  n · X = offset.
    'bounds' = lista de (eje, mínimo, máximo) para limitar el plano.
    """

    def __init__(self, normal, offset, mat, bounds=None):
        self.n = np.array(normal, dtype=np.float64)
        self.c = float(offset)
        self.mat = mat
        self.bounds = bounds
        self.is_sphere = False

    def intersect(self, O, D, eps=1e-4):
        """Intersección rayo-plano vectorizada."""
        denom = D @ self.n                         # (N,)
        safe = np.where(np.abs(denom) < 1e-9, 1.0, denom)
        t = (self.c - O @ self.n) / safe
        valid = (np.abs(denom) >= 1e-9) & (t > eps)
        if self.bounds is not None:
            P = O + t[:, None] * D
            for axis, lo, hi in self.bounds:
                valid &= (P[:, axis] >= lo) & (P[:, axis] <= hi)
        return np.where(valid, t, np.inf)

    def normal_at(self, P):
        """La normal de un plano es constante."""
        return np.broadcast_to(self.n, P.shape).copy()
