# -*- coding: utf-8 -*-
"""
==========================================================================
 RENDERER: MAPEO DE FOTONES (Jensen, 1996)
 Proyecto: Modelos Globales de Iluminación — Computación Visual (UNMSM)
==========================================================================

Algoritmo de DOS PASADAS:

PASADA 1 — Emisión de fotones:
  Se emiten fotones desde las fuentes de luz y se trazan por la escena.
  Cada vez que un fotón impacta una superficie difusa, se almacena su
  posición, dirección y energía en un MAPA DE FOTONES (kd-tree).
  Los fotones pueden rebotar (reflexión difusa / especular) según la
  reflectividad del material.

PASADA 2 — Renderizado:
  Se trazan rayos desde la cámara (como en ray tracing). En cada punto
  de impacto difuso, se estima la radiancia buscando los k fotones más
  cercanos en el mapa y promediando su energía sobre el área del disco
  que los contiene.

Separa un MAPA DE CÁUSTICAS (fotones que pasaron por superficies
especulares antes de llegar a una difusa) para representar la
concentración de luz a través de objetos reflectantes.

Fortalezas: resuelve cáusticas de forma eficiente.
Debilidades: sesgado (introduce desenfoque), requiere ajuste de parámetros.
"""

import numpy as np
from scipy.spatial import cKDTree

from ..core.vectors import normalize, dot, reflect
from ..core.materials import DIFFUSE, EMISSIVE, MIRROR, GLASS, pack_materials
from ..core.scene import scene_intersect, gather_normals
from ..core.sampling import cosine_weighted_hemisphere, sample_light_rect
from ..config import RenderConfig


class PhotonMappingRenderer:
    """
    Mapeo de fotones de Jensen (1996).

    Dos pasadas: emisión de fotones → estimación por densidad.
    """

    name = "Mapeo de Fotones (Jensen)"

    def __init__(self, config: RenderConfig = None):
        self.cfg = config or RenderConfig()

    def render(self, objetos, light_info, camera):
        """Renderiza la escena. Devuelve hdr (H, W, 3) float64."""
        cfg = self.cfg
        kinds, albedos, emissions, iors = pack_materials(objetos)
        eps = cfg.eps

        rng = np.random.default_rng(cfg.seed)

        # ==============================================================
        #  PASADA 1: Emitir fotones y construir el mapa
        # ==============================================================
        global_photons, caustic_photons = self._emit_photons(
            objetos, light_info, kinds, albedos, cfg, rng, eps
        )

        # Construir kd-trees para búsqueda eficiente
        global_tree = None
        if len(global_photons["pos"]) > 0:
            global_positions = np.array(global_photons["pos"])
            global_powers = np.array(global_photons["power"])
            global_tree = cKDTree(global_positions)
        else:
            global_positions = np.zeros((0, 3))
            global_powers = np.zeros((0, 3))

        caustic_tree = None
        if len(caustic_photons["pos"]) > 0:
            caustic_positions = np.array(caustic_photons["pos"])
            caustic_powers = np.array(caustic_photons["power"])
            caustic_tree = cKDTree(caustic_positions)
        else:
            caustic_positions = np.zeros((0, 3))
            caustic_powers = np.zeros((0, 3))

        print(f"   Fotones globales:   {len(global_photons['pos']):,}")
        print(f"   Fotones cáusticos:  {len(caustic_photons['pos']):,}")

        # ==============================================================
        #  PASADA 2: Ray tracing + estimación de densidad
        # ==============================================================
        return self._render_pass(
            objetos, light_info, camera, kinds, albedos, emissions,
            global_tree, global_positions, global_powers,
            caustic_tree, caustic_positions, caustic_powers,
            cfg, rng, eps
        )

    def _emit_photons(self, objetos, light_info, kinds, albedos, cfg, rng, eps):
        """
        Emite fotones desde la fuente de luz y los traza por la escena.

        Separa fotones globales (difusos) y cáusticos (que pasaron por
        una superficie especular/vidrio).
        """
        global_photons = {"pos": [], "power": [], "dir": []}
        caustic_photons = {"pos": [], "power": [], "dir": []}

        Le = light_info["emission"]
        area = light_info["area"]
        n_light = light_info["normal"]
        n_global = cfg.photon_count
        n_caustic = cfg.photon_caustic_count

        # Potencia total de cada fotón: (emisión × área) / num_fotones
        total_photons = n_global + n_caustic

        for phase in range(2):
            n_emit = n_global if phase == 0 else n_caustic
            if n_emit == 0:
                continue

            target = global_photons if phase == 0 else caustic_photons
            power_per_photon = Le * area / n_emit

            # Emitir en lotes para eficiencia
            batch_size = min(n_emit, 10000)
            emitted = 0

            while emitted < n_emit:
                bs = min(batch_size, n_emit - emitted)

                # Posición inicial: puntos aleatorios en la fuente de luz
                O = sample_light_rect(bs, rng, light_info)

                # Dirección: hemisferio ponderado por coseno (hacia abajo)
                n_down = np.broadcast_to(n_light, (bs, 3)).copy()
                D = cosine_weighted_hemisphere(n_down, rng)

                power = np.broadcast_to(power_per_photon, (bs, 3)).copy()
                alive = np.ones(bs, dtype=bool)
                came_from_specular = np.zeros(bs, dtype=bool)

                for bounce in range(cfg.max_bounces):
                    if not np.any(alive):
                        break

                    ai = np.where(alive)[0]
                    Oa, Da = O[ai], D[ai]
                    t, obj = scene_intersect(objetos, Oa, Da, eps)

                    hit = obj >= 0
                    alive[ai[~hit]] = False
                    if not np.any(hit):
                        continue

                    hi = ai[hit]
                    Oa, Da, t, obj = Oa[hit], Da[hit], t[hit], obj[hit]
                    P = Oa + t[:, None] * Da
                    Nn = gather_normals(objetos, obj, P)
                    facing = dot(Da, Nn) > 0
                    Nn[facing] = -Nn[facing]
                    obj_kind = kinds[obj]

                    # Emisivos: absorber
                    emis_mask = (obj_kind == EMISSIVE)
                    alive[hi[emis_mask]] = False

                    # Difuso: almacenar fotón y posiblemente rebotar
                    dif_mask = (obj_kind == DIFFUSE)
                    if np.any(dif_mask):
                        di = hi[dif_mask]
                        Pd = P[dif_mask]
                        pw = power[di]

                        # Almacenar el fotón
                        if phase == 0:
                            # Global: solo si el fotón ya ha rebotado al menos una vez
                            # (la iluminación directa se calcula por separado)
                            already_bounced = bounce > 0
                            if already_bounced:
                                for k in range(Pd.shape[0]):
                                    target["pos"].append(Pd[k].copy())
                                    target["power"].append(pw[k].copy())
                                    target["dir"].append(Da[dif_mask][k].copy())
                        else:
                            # Cáusticos: solo si vino de una superficie especular
                            for k in range(Pd.shape[0]):
                                if came_from_specular[di[k]]:
                                    target["pos"].append(Pd[k].copy())
                                    target["power"].append(pw[k].copy())
                                    target["dir"].append(Da[dif_mask][k].copy())

                        # Rebote difuso (ruleta rusa según albedo)
                        alb = albedos[obj[dif_mask]]
                        p_survive = np.max(alb, axis=1)
                        survive = rng.random(di.shape[0]) < p_survive
                        alive[di[~survive]] = False
                        if np.any(survive):
                            ds = di[survive]
                            Ps = Pd[survive]
                            Ns = Nn[dif_mask][survive]
                            D[ds] = cosine_weighted_hemisphere(Ns, rng)
                            O[ds] = Ps + Ns * eps
                            power[ds] *= alb[survive] / np.maximum(p_survive[survive, None], 1e-6)
                            came_from_specular[ds] = False

                    # Espejo: reflexión especular
                    mir_mask = (obj_kind == MIRROR)
                    if np.any(mir_mask):
                        mi = hi[mir_mask]
                        D[mi] = reflect(Da[mir_mask], Nn[mir_mask])
                        O[mi] = P[mir_mask] + Nn[mir_mask] * eps
                        power[mi] *= albedos[obj[mir_mask]]
                        came_from_specular[mi] = True

                    # Vidrio: simplificado como espejo para cáusticas
                    glass_mask = (obj_kind == GLASS)
                    if np.any(glass_mask):
                        gi = hi[glass_mask]
                        D[gi] = reflect(Da[glass_mask], Nn[glass_mask])
                        O[gi] = P[glass_mask] + Nn[glass_mask] * eps
                        power[gi] *= albedos[obj[glass_mask]]
                        came_from_specular[gi] = True

                emitted += bs

        return global_photons, caustic_photons

    def _render_pass(self, objetos, light_info, camera, kinds, albedos,
                     emissions, global_tree, global_pos, global_pow,
                     caustic_tree, caustic_pos, caustic_pow, cfg, rng, eps):
        """
        Segunda pasada: traza rayos desde la cámara y estima la radiancia
        en cada punto de impacto usando el mapa de fotones.
        """
        H, W = camera.height, camera.width
        Npix = H * W
        Le = light_info["emission"]
        n_light = light_info["normal"]
        area = light_info["area"]

        n_aa = max(1, cfg.spp // 16)
        accum = np.zeros((Npix, 3))

        for s in range(n_aa):
            O, D = camera.generate_primary_rays(rng)
            color = np.zeros((Npix, 3))
            throughput = np.ones((Npix, 3))
            active = np.ones(Npix, dtype=bool)

            for depth in range(min(cfg.max_bounces, 4)):
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

                # Emisivo
                emis = (obj_kind == EMISSIVE)
                if np.any(emis):
                    ge = gidx[emis]
                    color[ge] += throughput[ge] * emissions[obj[emis]]
                    active[ge] = False

                # Difuso: iluminación directa + estimación por fotones
                dif = (obj_kind == DIFFUSE)
                if np.any(dif):
                    gd = gidx[dif]
                    Pd, Ndn = P[dif], Nn[dif]
                    ad = albedos[obj[dif]]

                    # Componente directa (NEE simplificado)
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
                    direct = (throughput[gd] * (ad / np.pi)
                              * Le[None, :] * (G * area)[:, None])
                    direct[~visible] = 0.0
                    color[gd] += direct

                    # Componente indirecta: estimación por densidad de fotones
                    if global_tree is not None and len(global_pos) > cfg.photon_k_nearest:
                        indirect = self._estimate_radiance(
                            Pd, Ndn, ad, global_tree, global_pos, global_pow,
                            cfg.photon_k_nearest
                        )
                        color[gd] += throughput[gd] * indirect

                    # Cáusticas
                    if caustic_tree is not None and len(caustic_pos) > cfg.photon_caustic_k:
                        caustic_contrib = self._estimate_radiance(
                            Pd, Ndn, ad, caustic_tree, caustic_pos, caustic_pow,
                            cfg.photon_caustic_k
                        )
                        color[gd] += throughput[gd] * caustic_contrib

                    active[gd] = False  # no propagamos más en esta versión

                # Espejo: reflexión
                mir = (obj_kind == MIRROR)
                if np.any(mir):
                    gm = gidx[mir]
                    D[gm] = reflect(Da[mir], Nn[mir])
                    O[gm] = P[mir] + Nn[mir] * eps
                    throughput[gm] *= albedos[obj[mir]]

            accum += color

        return (accum / n_aa).reshape(H, W, 3)

    def _estimate_radiance(self, points, normals, albedos,
                           tree, photon_pos, photon_pow, k):
        """
        Estima la radiancia en cada punto por densidad de fotones vecinos.

        Para cada punto, busca los k fotones más cercanos en el kd-tree.
        La radiancia se aproxima como:
            L ≈ (1 / (π r²)) Σ (f_r · Φ_j)
        donde r es la distancia al k-ésimo fotón y Φ_j es la potencia.
        """
        N = points.shape[0]
        result = np.zeros((N, 3))

        # Procesar en lotes
        batch_size = 2000
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_P = points[start:end]
            batch_N = normals[start:end]
            batch_A = albedos[start:end]

            dists, indices = tree.query(batch_P, k=k)
            # Radio del disco = distancia al k-ésimo vecino
            r = dists[:, -1]
            r = np.maximum(r, 1e-4)
            area_disk = np.pi * r ** 2

            # Sumar la potencia de todos los fotones en el disco
            # filtrando por orientación (solo los del mismo lado de la normal)
            for i in range(end - start):
                phot_idx = indices[i]
                phot_pw = photon_pow[phot_idx]
                # Filtro: fotones cuya dirección es consistente con la normal
                # (simplificado: aceptar todos)
                total_power = np.sum(phot_pw, axis=0)
                # L = (albedo / π) · Σ Φ / (π r²)
                result[start + i] = (batch_A[i] / np.pi) * total_power / area_disk[i]

        return result
