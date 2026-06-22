# -*- coding: utf-8 -*-
"""
==========================================================================
 RENDERER: ILUMINACIÓN LOCAL (Phong simplificado)
 Proyecto: Modelos Globales de Iluminación — Computación Visual (UNMSM)
==========================================================================

Solo calcula luz DIRECTA desde una fuente puntual + término ambiente
constante. NO hay rebotes indirectos → sombras duras, sin sangrado de
color, sin interreflexión difusa.

Este renderer existe como referencia de contraste: evidencia las
limitaciones de la iluminación local frente a los modelos globales.
"""

import numpy as np
from ..core.vectors import normalize, dot
from ..core.materials import DIFFUSE, EMISSIVE, MIRROR, GLASS, pack_materials
from ..core.scene import scene_intersect, gather_normals, LIGHT_Y
from ..config import RenderConfig


# Posición de la luz puntual (centro de la fuente rectangular)
LIGHT_CENTER = np.array([0.0, LIGHT_Y, 0.0])
LIGHT_POWER  = np.array([1.6, 1.5, 1.35])


class LocalRenderer:
    """
    Renderer de iluminación local.

    Modelo: componente difusa lambertiana (n · L) con atenuación 1/r²,
    rayos de sombra para oclusión, y un término ambiente constante que
    aproxima groseramente la luz indirecta.
    """

    name = "Local (Directa)"

    def __init__(self, config: RenderConfig = None):
        self.cfg = config or RenderConfig()

    def render(self, objetos, light_info, camera):
        """
        Renderiza la escena con iluminación local.

        Devuelve hdr (H, W, 3) float64.
        """
        cfg = self.cfg
        kinds, albedos, emissions, iors = pack_materials(objetos)
        H, W = camera.height, camera.width
        Npix = H * W
        accum = np.zeros((Npix, 3))
        eps = cfg.eps

        rng = np.random.default_rng(cfg.seed)

        # Pocas muestras: el modo local no tiene ruido de Monte Carlo,
        # solo necesitamos jitter para anti-aliasing.
        n_aa = max(4, cfg.spp // 8)

        for s in range(n_aa):
            O, D = camera.generate_primary_rays(rng)
            t, obj = scene_intersect(objetos, O, D, eps)
            hit = obj >= 0
            col = np.zeros((Npix, 3))

            if np.any(hit):
                Oi, Di, ti, oi = O[hit], D[hit], t[hit], obj[hit]
                P = Oi + ti[:, None] * Di
                Nn = gather_normals(objetos, oi, P)
                facing = (dot(Di, Nn) > 0)
                Nn[facing] = -Nn[facing]
                kind = kinds[oi]
                alb = albedos[oi]

                c = np.zeros((P.shape[0], 3))

                # Fuente de luz: se muestra como blanca brillante
                emis = (kind == EMISSIVE)
                c[emis] = emissions[oi[emis]] * 0.06

                # Superficies (difuso, espejo tratado como difuso, vidrio como difuso)
                surf = (kind == DIFFUSE) | (kind == MIRROR) | (kind == GLASS)
                if np.any(surf):
                    Ps = P[surf]
                    Ns = Nn[surf]
                    As = alb[surf]

                    # Vector hacia la luz puntual
                    L = LIGHT_CENTER - Ps
                    dist2 = dot(L, L)
                    Ld = normalize(L)
                    cos_s = np.maximum(0.0, dot(Ns, Ld))

                    # Rayo de sombra
                    shadow_O = Ps + Ns * eps
                    st, sobj = scene_intersect(objetos, shadow_O, Ld, eps)
                    dist = np.sqrt(dist2)
                    visible = ((sobj < 0)
                               | (st >= dist - 1e-3)
                               | (kinds[np.where(sobj < 0, 0, sobj)] == EMISSIVE))

                    # Sombreado: difuso + atenuación + sombra + ambiente
                    atten = 1.0 / np.maximum(dist2, 1e-3)
                    direct = As * LIGHT_POWER[None, :] * (atten * cos_s)[:, None]
                    direct[~visible] = 0.0
                    ambient = As * cfg.ambient_local

                    tmp = c[surf]
                    tmp[:] = direct + ambient
                    c[surf] = tmp

                col[hit] = c

            accum += col

        return (accum / n_aa).reshape(H, W, 3)
