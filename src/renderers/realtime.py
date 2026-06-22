# -*- coding: utf-8 -*-
"""
==========================================================================
 RENDERER: APROXIMACIÓN EN TIEMPO REAL (SSAO + VPLs)
 Proyecto: Modelos Globales de Iluminación — Computación Visual (UNMSM)
==========================================================================

Implementa dos técnicas de iluminación global aproximada pensadas para
tiempo real:

1. SSAO (Screen-Space Ambient Occlusion, Zhukov et al. 1998 / Mittring 2007):
   Estima cuánta geometría bloquea la luz ambiente en cada píxel muestreando
   puntos en el espacio de pantalla alrededor de cada punto visible.
   Las zonas "encerradas" (esquinas, grietas) se oscurecen.

2. VPLs (Virtual Point Lights / Radiosidad Instantánea, Keller 1997):
   Se emite un número reducido de fotones y en cada impacto difuso se coloca
   una fuente puntual virtual. La escena se renderiza sumando la contribución
   de todas las VPLs.

Ambas técnicas sacrifican exactitud física a cambio de velocidad.
"""

import numpy as np
from ..core.vectors import normalize, dot, reflect
from ..core.materials import DIFFUSE, EMISSIVE, MIRROR, pack_materials
from ..core.scene import scene_intersect, gather_normals
from ..core.sampling import cosine_weighted_hemisphere, sample_light_rect
from ..config import RenderConfig


class RealtimeRenderer:
    """
    Renderer de iluminación global aproximada en tiempo real.

    Combina iluminación directa + SSAO + VPLs para dar una aproximación
    rápida de la iluminación indirecta sin el costo del path tracing.
    """

    name = "Tiempo Real (SSAO + VPLs)"

    def __init__(self, config: RenderConfig = None):
        self.cfg = config or RenderConfig()

    def render(self, objetos, light_info, camera):
        """Renderiza la escena. Devuelve hdr (H, W, 3) float64."""
        cfg = self.cfg
        kinds, albedos, emissions, _ = pack_materials(objetos)
        H, W = camera.height, camera.width
        Npix = H * W
        eps = cfg.eps
        Le = light_info["emission"]
        n_light = light_info["normal"]
        area = light_info["area"]

        rng = np.random.default_rng(cfg.seed)

        # ====================================================================
        #  1. G-BUFFER: trazar rayos primarios y obtener punto, normal, albedo
        # ====================================================================
        O, D = camera.generate_primary_rays(rng)
        t, obj_idx = scene_intersect(objetos, O, D, eps)
        hit = obj_idx >= 0

        P_buf = np.zeros((Npix, 3))
        N_buf = np.zeros((Npix, 3))
        A_buf = np.zeros((Npix, 3))
        K_buf = np.full(Npix, -1, dtype=np.int64)
        T_buf = np.full(Npix, np.inf)

        if np.any(hit):
            P_buf[hit] = O[hit] + t[hit, None] * D[hit]
            Nn = gather_normals(objetos, obj_idx[hit], P_buf[hit])
            facing = dot(D[hit], Nn) > 0
            Nn[facing] = -Nn[facing]
            N_buf[hit] = Nn
            A_buf[hit] = albedos[obj_idx[hit]]
            K_buf[hit] = kinds[obj_idx[hit]]
            T_buf[hit] = t[hit]

        color = np.zeros((Npix, 3))

        # Emisivos
        emis_mask = K_buf == EMISSIVE
        color[emis_mask] = emissions[obj_idx[emis_mask]] * 0.15

        # ====================================================================
        #  2. ILUMINACIÓN DIRECTA (para superficies difusas)
        # ====================================================================
        dif_mask = K_buf == DIFFUSE
        dif_idx = np.where(dif_mask)[0]

        if len(dif_idx) > 0:
            Pd = P_buf[dif_idx]
            Nd = N_buf[dif_idx]
            Ad = A_buf[dif_idx]

            # Muestreo de la fuente de luz
            n_shadow = 4
            direct = np.zeros((len(dif_idx), 3))
            for _ in range(n_shadow):
                Q = sample_light_rect(len(dif_idx), rng, light_info)
                to_l = Q - Pd
                d2 = dot(to_l, to_l)
                dist = np.sqrt(d2)
                wi = to_l / dist[:, None]
                cos_s = np.maximum(0.0, dot(Nd, wi))
                cos_l = np.maximum(0.0, dot(-wi, n_light))
                sh_O = Pd + Nd * eps
                st, _ = scene_intersect(objetos, sh_O, wi, eps)
                visible = st >= (dist - 1e-3)
                G = (cos_s * cos_l) / np.maximum(d2, 1e-6)
                contrib = (Ad / np.pi) * Le[None, :] * (G * area)[:, None]
                contrib[~visible] = 0.0
                direct += contrib
            direct /= n_shadow
            color[dif_idx] += direct

        # ====================================================================
        #  3. SSAO (Screen-Space Ambient Occlusion)
        # ====================================================================
        if len(dif_idx) > 0:
            ao = self._compute_ssao(
                P_buf, N_buf, T_buf, dif_mask, objetos, cfg, rng
            )
            # Aplicar SSAO como factor de oscurecimiento
            color[dif_idx] *= ao[dif_idx, None]

        # ====================================================================
        #  4. VPLs (Virtual Point Lights)
        # ====================================================================
        if len(dif_idx) > 0:
            vpl_contrib = self._compute_vpls(
                Pd, Nd, Ad, objetos, light_info, kinds, albedos, cfg, rng, eps
            )
            color[dif_idx] += vpl_contrib

        # ====================================================================
        #  5. ESPEJO: reflexión simple (1 rebote)
        # ====================================================================
        mir_mask = K_buf == MIRROR
        mir_idx = np.where(mir_mask)[0]
        if len(mir_idx) > 0:
            Pm = P_buf[mir_idx]
            Nm = N_buf[mir_idx]
            Am = A_buf[mir_idx]
            Dm = D[mir_idx]

            refl_D = reflect(Dm, Nm)
            refl_O = Pm + Nm * eps
            t2, obj2 = scene_intersect(objetos, refl_O, refl_D, eps)
            hit2 = obj2 >= 0
            if np.any(hit2):
                P2 = refl_O[hit2] + t2[hit2, None] * refl_D[hit2]
                kind2 = kinds[obj2[hit2]]
                # Emisivo
                e2 = kind2 == EMISSIVE
                reflected_color = np.zeros((len(mir_idx), 3))
                reflected_color[np.where(hit2)[0][e2]] = emissions[obj2[hit2][e2]]
                # Difuso: tomar su albedo como aproximación
                d2_mask = kind2 == DIFFUSE
                if np.any(d2_mask):
                    reflected_color[np.where(hit2)[0][d2_mask]] = \
                        albedos[obj2[hit2][d2_mask]] * 0.3
                color[mir_idx] += Am * reflected_color

        return color.reshape(H, W, 3)

    def _compute_ssao(self, P_buf, N_buf, T_buf, dif_mask, objetos, cfg, rng):
        """
        Calcula la oclusión ambiental en espacio de pantalla.

        Para cada punto difuso, muestrea puntos aleatorios en un hemisferio
        local y comprueba si están ocluidos por geometría cercana.
        """
        Npix = P_buf.shape[0]
        ao = np.ones(Npix)
        dif_idx = np.where(dif_mask)[0]

        if len(dif_idx) == 0:
            return ao

        Pd = P_buf[dif_idx]
        Nd = N_buf[dif_idx]
        n_samples = cfg.ssao_samples
        radius = cfg.ssao_radius
        eps = cfg.eps

        occlusion = np.zeros(len(dif_idx))

        for _ in range(n_samples):
            # Dirección aleatoria en el hemisferio de la normal
            dirs = cosine_weighted_hemisphere(Nd, rng)
            # Punto de muestreo
            sample_O = Pd + Nd * eps
            t, obj = scene_intersect(objetos, sample_O, dirs, eps)
            # Si hay intersección dentro del radio → ocluido
            occluded = (obj >= 0) & (t < radius)
            occlusion += occluded.astype(float)

        # Factor AO: 1 = totalmente abierto, 0 = totalmente ocluido
        ao_factor = 1.0 - (occlusion / n_samples) * 0.7  # atenuar un poco
        ao_factor = np.clip(ao_factor, 0.2, 1.0)
        ao[dif_idx] = ao_factor

        return ao

    def _compute_vpls(self, Pd, Nd, Ad, objetos, light_info, kinds, albedos,
                      cfg, rng, eps):
        """
        Calcula la contribución de luces puntuales virtuales (VPLs).

        Emite fotones desde la fuente de luz, coloca VPLs en los impactos
        difusos, y calcula la iluminación indirecta desde cada VPL.
        """
        n_vpls = cfg.vpl_count
        Le = light_info["emission"]
        n_light = light_info["normal"]
        area = light_info["area"]

        # Emitir fotones y colocar VPLs
        vpl_positions = []
        vpl_normals = []
        vpl_powers = []
        power_per_vpl = Le * area / n_vpls

        O_v = sample_light_rect(n_vpls, rng, light_info)
        n_down = np.broadcast_to(n_light, (n_vpls, 3)).copy()
        D_v = cosine_weighted_hemisphere(n_down, rng)

        t, obj = scene_intersect(objetos, O_v, D_v, eps)
        hit = (obj >= 0)

        if np.any(hit):
            P_hit = O_v[hit] + t[hit, None] * D_v[hit]
            Nn_hit = gather_normals(objetos, obj[hit], P_hit)
            facing = dot(D_v[hit], Nn_hit) > 0
            Nn_hit[facing] = -Nn_hit[facing]
            kind_hit = kinds[obj[hit]]

            dif = kind_hit == DIFFUSE
            if np.any(dif):
                vpl_P = P_hit[dif]
                vpl_N = Nn_hit[dif]
                vpl_alb = albedos[obj[hit][dif]]
                vpl_pw = vpl_alb * power_per_vpl  # modular por albedo

                for k in range(vpl_P.shape[0]):
                    vpl_positions.append(vpl_P[k])
                    vpl_normals.append(vpl_N[k])
                    vpl_powers.append(vpl_pw[k])

        if not vpl_positions:
            return np.zeros((Pd.shape[0], 3))

        vpl_positions = np.array(vpl_positions)
        vpl_normals = np.array(vpl_normals)
        vpl_powers = np.array(vpl_powers)

        # Calcular la contribución de cada VPL a cada punto difuso
        N_pts = Pd.shape[0]
        N_vpl = vpl_positions.shape[0]
        result = np.zeros((N_pts, 3))

        # Procesar en lotes para memoria
        batch = min(N_pts, 1000)
        for start in range(0, N_pts, batch):
            end = min(start + batch, N_pts)
            batch_P = Pd[start:end]         # (B, 3)
            batch_N = Nd[start:end]         # (B, 3)
            batch_A = Ad[start:end]         # (B, 3)

            for v in range(N_vpl):
                to_vpl = vpl_positions[v] - batch_P                    # (B, 3)
                d2 = np.sum(to_vpl ** 2, axis=1)                      # (B,)
                dist = np.sqrt(d2)
                wi = to_vpl / np.maximum(dist[:, None], 1e-6)

                cos_recv = np.maximum(0.0, np.sum(batch_N * wi, axis=1))
                cos_vpl = np.maximum(0.0, np.sum(-wi * vpl_normals[v], axis=1))

                # Contribución = (f_r · Φ_vpl · G) con G = cos_r · cos_v / r²
                G = (cos_recv * cos_vpl) / np.maximum(d2, 1e-4)
                # Limitar G para evitar singularidades
                G = np.minimum(G, 10.0)

                contrib = (batch_A / np.pi) * vpl_powers[v] * G[:, None]
                result[start:end] += contrib

        return result
