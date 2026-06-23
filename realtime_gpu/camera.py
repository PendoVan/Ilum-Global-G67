# -*- coding: utf-8 -*-
"""
Camara orbital simple para la demo en tiempo real.

Por defecto reproduce exactamente la camara fija de src/core/camera.py
(origen en (0,0,-0.85), mirando hacia +z), pero permite orbitar alrededor
del centro de la Caja de Cornell con el mouse para inspeccionar la escena.
"""

import math
import numpy as np

WORLD_UP = np.array([0.0, 1.0, 0.0])


class OrbitCamera:
    def __init__(self, target=(0.0, 0.0, 0.0), distance=0.85,
                 yaw=0.0, pitch=0.0, fov_scale=0.95):
        self.target = np.array(target, dtype=np.float64)
        self.distance = distance
        self.yaw = yaw      # radianes, alrededor del eje Y
        self.pitch = pitch  # radianes, limitado para no voltear la camara
        self.fov_scale = fov_scale

        self._default = (distance, yaw, pitch)
        self.dirty = True   # fuerza reset de la acumulacion en el primer frame

    def reset(self):
        self.distance, self.yaw, self.pitch = self._default
        self.dirty = True

    def orbit(self, d_yaw, d_pitch):
        self.yaw += d_yaw
        self.pitch = float(np.clip(self.pitch + d_pitch, -1.4, 1.4))
        self.dirty = True

    def zoom(self, amount):
        self.distance = float(np.clip(self.distance - amount, 0.25, 3.0))
        self.dirty = True

    def vectors(self):
        """Devuelve (posicion, right, up, forward) en coordenadas de mundo."""
        cp = math.cos(self.pitch)
        pos_dir = np.array([
            math.sin(self.yaw) * cp,
            math.sin(self.pitch),
            -math.cos(self.yaw) * cp,
        ])
        pos = self.target + self.distance * pos_dir
        forward = self.target - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(WORLD_UP, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        return pos, right, up, forward
