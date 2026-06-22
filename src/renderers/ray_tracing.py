# -*- coding: utf-8 -*-
"""
==========================================================================
 RENDERER: TRAZADO DE RAYOS RECURSIVO (Whitted, 1980)
 Proyecto: Modelos Globales de Iluminación — Computación Visual (UNMSM)
==========================================================================

Implementa el trazado de rayos clásico de Whitted:
  - Rayos primarios desde la cámara
  - En cada impacto: sombreado difuso (Blinn-Phong), reflexión especular
    recursiva, refracción (Snell + Fresnel) para materiales GLASS,
    y rayos de sombra hacia la fuente de luz.
  - Profundidad máxima configurable.

A diferencia del path tracing, NO muestrea direcciones aleatorias y por
tanto NO captura la interreflexión difusa ni el sangrado de color, pero
produce reflejos y refracciones limpios sin ruido.
"""

import numpy as np
from ..core.vectors import normalize, dot, reflect, refract, fresnel_schlick
from ..core.materials import DIFFUSE, EMISSIVE, MIRROR, GLASS, pack_materials
from ..core.scene import scene_intersect, gather_normals, LIGHT_Y
from ..core.sampling import sample_light_rect
from ..config import RenderConfig


LIGHT_CENTER = np.array([0.0, LIGHT_Y, 0.0])


class RayTracingRenderer:
    """
    Trazado de rayos recursivo de Whitted (1980).

    Genera reflejos especulares precisos, refracciones con Fresnel,
    y sombras nítidas, pero sin iluminación indirecta difusa.
    """

    name = "Ray Tracing (Whitted)"

    def __init__(self, config: RenderConfig = None):
        self.cfg = config or RenderConfig()

    def render(self, objetos, light_info, camera):
        """Renderiza la escena. Devuelve hdr (H, W, 3) float64."""
        cfg = self.cfg
        H, W = camera.height, camera.width
        Npix = H * W
        eps = cfg.eps
        max_depth = min(cfg.max_bounces, 8)

        kinds, albedos, emissions, iors = pack_materials(objetos)
        Le = light_info["emission"]
        n_light = light_info["normal"]
        area = light_info["area"]

        rng = np.random.default_rng(cfg.seed)

        # Anti-aliasing: varias muestras con jitter
        n_aa = max(4, cfg.spp // 4)
        accum = np.zeros((Npix, 3))

        for s in range(n_aa):
            O, D = camera.generate_primary_rays(rng)
            color = self._trace_batch(
                objetos, O, D, kinds, albedos, emissions, iors,
                light_info, max_depth, eps, rng, Npix
            )
            accum += color

        return (accum / n_aa).reshape(H, W, 3)

    def _trace_batch(self, objetos, O, D, kinds, albedos, emissions, iors,
                     light_info, max_depth, eps, rng, Npix):
        """
        Trazado de rayos iterativo (simula la recursión por lotes).

        Mantiene el color y el factor de atenuación de cada rayo, y en cada
        rebote actualiza según el tipo de material encontrado.
        """
        Le = light_info["emission"]
        n_light = light_info["normal"]
        area = light_info["area"]

        color = np.zeros((Npix, 3))
        throughput = np.ones((Npix, 3))
        active = np.ones(Npix, dtype=bool)

        for depth in range(max_depth):
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

            # --- Emisivo: la fuente de luz ---
            emis = (obj_kind == EMISSIVE)
            if np.any(emis):
                ge = gidx[emis]
                color[ge] += throughput[ge] * emissions[obj[emis]]
                active[ge] = False

            # --- Difuso: Blinn-Phong + sombras ---
            dif = (obj_kind == DIFFUSE)
            if np.any(dif):
                gd = gidx[dif]
                Pd, Ndn = P[dif], Nn[dif]
                ad = albedos[obj[dif]]

                # Componente difusa con múltiples muestras de la fuente de área
                n_shadow = 4
                direct = np.zeros((Pd.shape[0], 3))
                for _ in range(n_shadow):
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
                    contrib = (ad / np.pi) * Le[None, :] * (G * area)[:, None]
                    contrib[~visible] = 0.0
                    direct += contrib

                direct /= n_shadow

                # Componente especular Blinn-Phong (highlight)
                V = -Da[dif]
                Ldir = normalize(LIGHT_CENTER - Pd)
                H_vec = normalize(V + Ldir)
                spec_dot = np.maximum(0.0, dot(Ndn, H_vec))
                shininess = 32.0
                specular = 0.04 * Le[None, :] * (spec_dot ** shininess)[:, None]

                color[gd] += throughput[gd] * (direct + specular)
                active[gd] = False  # difusos no rebotan en Whitted clásico

            # --- Espejo: reflexión perfecta ---
            mir = (obj_kind == MIRROR)
            if np.any(mir):
                gm = gidx[mir]
                D[gm] = reflect(Da[mir], Nn[mir])
                O[gm] = P[mir] + Nn[mir] * eps
                throughput[gm] *= albedos[obj[mir]]

            # --- Vidrio: refracción + Fresnel ---
            glass = (obj_kind == GLASS)
            if np.any(glass):
                gg = gidx[glass]
                Dg = Da[glass]
                Ng = Nn[glass]
                Pg = P[glass]
                ior_mat = iors[obj[glass]]

                # ¿Entramos o salimos del medio?
                entering = dot(Dg, gather_normals(objetos, obj[glass], Pg)) < 0
                # Ajustar normal y eta
                n_refr = Ng.copy()
                eta = np.where(entering, 1.0 / ior_mat, ior_mat)

                cos_i = np.abs(dot(Dg, n_refr))
                R = fresnel_schlick(cos_i, ior_mat[0] if len(ior_mat.shape) == 0 else ior_mat)

                refr_dir, valid_refr = refract(Dg, n_refr, eta)

                # Decisión: reflejar o refractar según Fresnel
                do_reflect = (rng.random(gg.shape[0]) < R) | (~valid_refr)

                # Reflexión
                refl_mask = do_reflect
                if np.any(refl_mask):
                    gr = gg[refl_mask]
                    D[gr] = reflect(Dg[refl_mask], Ng[refl_mask])
                    O[gr] = Pg[refl_mask] + Ng[refl_mask] * eps
                    throughput[gr] *= albedos[obj[glass][refl_mask]]

                # Refracción
                refr_mask = ~do_reflect
                if np.any(refr_mask):
                    gt = gg[refr_mask]
                    D[gt] = refr_dir[refr_mask]
                    O[gt] = Pg[refr_mask] - Ng[refr_mask] * eps
                    throughput[gt] *= albedos[obj[glass][refr_mask]]

        return color
