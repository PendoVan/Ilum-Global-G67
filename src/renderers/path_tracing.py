# -*- coding: utf-8 -*-
"""
==========================================================================
 RENDERER: TRAZADO DE TRAYECTORIAS (Path Tracing, Kajiya 1986)
 Proyecto: Modelos Globales de Iluminación — Computación Visual (UNMSM)
==========================================================================

Resuelve numéricamente la ECUACIÓN DE RENDERIZADO mediante integración de
Monte Carlo.  Combina:
  (1) Muestreo de BRDF (cosine-weighted) para la iluminación indirecta
  (2) Next Event Estimation (NEE) para la iluminación directa
  (3) Ruleta Rusa para terminar trayectorias sin sesgo

Es el estándar de la producción cinematográfica (Arnold, RenderMan, Cycles).
Refactorizado del path_tracer_cornell.py original usando los módulos core.
"""

import numpy as np
from ..core.vectors import normalize, dot, reflect
from ..core.materials import DIFFUSE, EMISSIVE, MIRROR, pack_materials
from ..core.scene import scene_intersect, gather_normals
from ..core.sampling import cosine_weighted_hemisphere, sample_light_rect
from ..config import RenderConfig


class PathTracingRenderer:
    """
    Trazado de trayectorias con Monte Carlo.

    Captura todos los efectos de transporte de luz: iluminación indirecta,
    sangrado de color, sombras suaves, reflejos, etc.
    """

    name = "Path Tracing (Monte Carlo)"

    def __init__(self, config: RenderConfig = None):
        self.cfg = config or RenderConfig()

    def render(self, objetos, light_info, camera):
        """Renderiza la escena. Devuelve hdr (H, W, 3) float64."""
        cfg = self.cfg
        kinds, albedos, emissions, _ = pack_materials(objetos)
        H, W = camera.height, camera.width
        Npix = H * W
        accum = np.zeros((Npix, 3))
        Le = light_info["emission"]
        n_light = light_info["normal"]
        area = light_info["area"]
        eps = cfg.eps

        rng = np.random.default_rng(cfg.seed)

        for s in range(cfg.spp):
            O, D = camera.generate_primary_rays(rng)
            throughput = np.ones((Npix, 3))
            radiance = np.zeros((Npix, 3))
            active = np.ones(Npix, dtype=bool)
            # Flag: ¿sumar emisión si se golpea la luz?
            # True para rayo primario y tras rebotes especulares;
            # False tras difusos (porque NEE ya contó la luz).
            spec = np.ones(Npix, dtype=bool)

            for bounce in range(cfg.max_bounces):
                if not np.any(active):
                    break

                act_idx = np.where(active)[0]
                Oa, Da = O[act_idx], D[act_idx]
                t, obj = scene_intersect(objetos, Oa, Da, eps)

                hit = obj >= 0
                active[act_idx[~hit]] = False
                if not np.any(hit):
                    continue

                gidx = act_idx[hit]
                Oa, Da, t, obj = Oa[hit], Da[hit], t[hit], obj[hit]
                P = Oa + t[:, None] * Da
                Nn = gather_normals(objetos, obj, P)
                facing = dot(Da, Nn) > 0
                Nn[facing] = -Nn[facing]
                obj_kind = kinds[obj]

                # --- (a) EMISIVO ---
                emis = (obj_kind == EMISSIVE)
                if np.any(emis):
                    ge = gidx[emis]
                    add = spec[ge]
                    radiance[ge[add]] += throughput[ge[add]] * emissions[obj[emis][add]]
                    active[ge] = False

                # --- (b) ESPEJO ---
                mir = (obj_kind == MIRROR)
                if np.any(mir):
                    gm = gidx[mir]
                    D[gm] = reflect(Da[mir], Nn[mir])
                    O[gm] = P[mir] + Nn[mir] * eps
                    throughput[gm] *= albedos[obj[mir]]
                    spec[gm] = True

                # --- (c) DIFUSO: NEE + rebote ~ coseno ---
                dif = (obj_kind == DIFFUSE)
                if np.any(dif):
                    gd = gidx[dif]
                    Pd, Ndn, ad = P[dif], Nn[dif], albedos[obj[dif]]

                    # (c.1) Next Event Estimation
                    Q = sample_light_rect(Pd.shape[0], rng, light_info)
                    to_l = Q - Pd
                    d2 = dot(to_l, to_l)
                    dist = np.sqrt(d2)
                    wi = to_l / dist[:, None]
                    cos_s = np.maximum(0.0, dot(Ndn, wi))
                    cos_l = np.maximum(0.0, dot(-wi, n_light))

                    sh_O = Pd + Ndn * eps
                    st, _ = scene_intersect(objetos, sh_O, wi, eps)
                    visible = st >= (dist - 1e-3)

                    G = (cos_s * cos_l) / np.maximum(d2, 1e-6)
                    contrib = (throughput[gd] * (ad / np.pi)
                               * Le[None, :] * (G * area)[:, None])
                    contrib[~visible] = 0.0
                    radiance[gd] += contrib

                    # (c.2) Rebote con muestreo por coseno
                    D[gd] = cosine_weighted_hemisphere(Ndn, rng)
                    O[gd] = Pd + Ndn * eps
                    throughput[gd] *= ad  # cos y 1/π se cancelan
                    spec[gd] = False

                # --- (d) RULETA RUSA ---
                if cfg.use_russian_roulette and bounce >= cfg.rr_start_bounce:
                    cont = gidx[dif | mir]
                    if cont.size:
                        p = np.clip(np.max(throughput[cont], axis=1), 0.05, 1.0)
                        keep = rng.random(cont.size) < p
                        throughput[cont[keep]] /= p[keep, None]
                        active[cont[~keep]] = False

            accum += radiance

        return (accum / cfg.spp).reshape(H, W, 3)
