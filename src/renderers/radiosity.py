# -*- coding: utf-8 -*-
"""
==========================================================================
 RENDERER: RADIOSIDAD (Goral et al., 1984)
 Proyecto: Modelos Globales de Iluminación — Computación Visual (UNMSM)
==========================================================================

Modela el intercambio de energía radiante entre superficies difusas
(lambertianas).  La escena se discretiza en PARCHES y para cada par se
calcula un FACTOR DE FORMA que cuantifica qué fracción de la energía de
uno alcanza al otro.

Se resuelve iterativamente por REFINAMIENTO PROGRESIVO:
  1. Encontrar el parche con más energía sin distribuir.
  2. Repartir esa energía a todos los demás, ponderada por el factor de
     forma y la reflectividad del receptor.
  3. Repetir hasta convergencia.

RESULTADO independiente del punto de vista: una vez calculada la radiosidad
de cada parche, se puede observar desde cualquier ángulo sin recalcular.

Reproduce: sangrado de color, sombras suaves, interreflexión difusa.
NO reproduce: reflejos especulares ni refracciones.
"""

import numpy as np
from ..core.vectors import dot
from ..core.materials import DIFFUSE, EMISSIVE, pack_materials
from ..core.scene import scene_intersect, gather_normals, LIGHT_Y, LIGHT_HX, LIGHT_HZ
from ..core.sampling import uniform_hemisphere
from ..config import RenderConfig


class RadiosityRenderer:
    """
    Renderer de radiosidad con refinamiento progresivo.

    Discretiza las paredes de la Caja de Cornell en parches NxN y calcula
    la distribución de luz difusa iterativamente.
    """

    name = "Radiosidad (Refinamiento Progresivo)"

    def __init__(self, config: RenderConfig = None):
        self.cfg = config or RenderConfig()

    def render(self, objetos, light_info, camera):
        """Renderiza la escena. Devuelve hdr (H, W, 3) float64."""
        cfg = self.cfg

        # ====================================================================
        #  1. DISCRETIZAR LA ESCENA EN PARCHES
        # ====================================================================
        patches, patch_normals, patch_areas, patch_albedos, patch_emissions = \
            self._create_patches(objetos, cfg.radiosity_patches_per_side)

        N_patches = len(patches)

        # ====================================================================
        #  2. CALCULAR FACTORES DE FORMA (por muestreo hemisférico con rayos)
        # ====================================================================
        form_factors = self._compute_form_factors(
            patches, patch_normals, patch_areas, objetos, cfg
        )

        # ====================================================================
        #  3. REFINAMIENTO PROGRESIVO
        # ====================================================================
        radiosity = patch_emissions.copy()      # B_i = E_i inicialmente
        unshot = patch_emissions.copy()          # energía aún no distribuida

        for iteration in range(cfg.radiosity_iterations):
            # Encontrar el parche con mayor energía sin disparar
            energies = np.sum(unshot * patch_areas[:, None], axis=1)
            i = np.argmax(energies)
            if energies[i] < 1e-6:
                break

            # Distribuir la energía del parche i a todos los demás
            delta = unshot[i]  # energía del emisor
            for j in range(N_patches):
                if j == i:
                    continue
                Fij = form_factors[i, j]
                if Fij < 1e-10:
                    continue
                received = patch_albedos[j] * delta * Fij
                radiosity[j] += received
                unshot[j] += received

            unshot[i] = 0.0  # ya disparado

        # ====================================================================
        #  4. RENDERIZAR DESDE LA CÁMARA
        # ====================================================================
        return self._render_from_patches(
            objetos, patches, patch_normals, radiosity, camera, cfg
        )

    def _create_patches(self, objetos, N):
        """
        Crea los parches dividiendo las caras de la Caja de Cornell.

        Se discretizan las 5 paredes visibles (izq, der, piso, techo, fondo)
        y la fuente de luz, más un parche por cada esfera visible.
        """
        patches = []       # centros (3,)
        normals = []       # normales (3,)
        areas = []         # áreas escalares
        albedos = []       # reflectividades RGB
        emissions = []     # emisiones RGB

        # --- Paredes de la caja: x,y,z ∈ [-1, 1] ---
        walls = [
            # (eje fijo, valor fijo, normal, eje_u, eje_v, albedo)
            (0, -1.0, np.array([ 1, 0, 0], dtype=np.float64), 1, 2,
             np.array([0.75, 0.20, 0.20])),   # izquierda (roja)
            (0,  1.0, np.array([-1, 0, 0], dtype=np.float64), 1, 2,
             np.array([0.20, 0.75, 0.25])),   # derecha (verde)
            (1, -1.0, np.array([ 0, 1, 0], dtype=np.float64), 0, 2,
             np.array([0.78, 0.78, 0.78])),   # piso (blanco)
            (1,  1.0, np.array([ 0,-1, 0], dtype=np.float64), 0, 2,
             np.array([0.78, 0.78, 0.78])),   # techo (blanco)
            (2,  1.0, np.array([ 0, 0,-1], dtype=np.float64), 0, 1,
             np.array([0.78, 0.78, 0.78])),   # fondo (blanco)
        ]

        patch_size = 2.0 / N
        patch_area = patch_size ** 2

        for axis_fixed, val_fixed, normal, axis_u, axis_v, alb in walls:
            for iu in range(N):
                for iv in range(N):
                    center = np.zeros(3)
                    center[axis_fixed] = val_fixed
                    center[axis_u] = -1.0 + (iu + 0.5) * patch_size
                    center[axis_v] = -1.0 + (iv + 0.5) * patch_size
                    patches.append(center)
                    normals.append(normal.copy())
                    areas.append(patch_area)
                    albedos.append(alb.copy())
                    emissions.append(np.zeros(3))

        # --- Fuente de luz (parches emisivos en el techo) ---
        light_patch_n = max(2, N // 3)
        light_patch_size_x = (2 * LIGHT_HX) / light_patch_n
        light_patch_size_z = (2 * LIGHT_HZ) / light_patch_n
        light_patch_area = light_patch_size_x * light_patch_size_z

        for iu in range(light_patch_n):
            for iv in range(light_patch_n):
                center = np.array([
                    -LIGHT_HX + (iu + 0.5) * light_patch_size_x,
                    LIGHT_Y,
                    -LIGHT_HZ + (iv + 0.5) * light_patch_size_z
                ])
                patches.append(center)
                normals.append(np.array([0.0, -1.0, 0.0]))
                areas.append(light_patch_area)
                albedos.append(np.zeros(3))
                emissions.append(np.array([7.0, 6.5, 5.5]))

        # --- Esferas: aproximar con parches en su superficie ---
        sphere_data = [
            (np.array([-0.40, -0.62, 0.35]), 0.38,
             np.array([0.80, 0.80, 0.82])),
            (np.array([ 0.45, -0.65, -0.10]), 0.35,
             np.array([0.80, 0.80, 0.82])),
        ]
        n_phi = max(6, N // 2)
        n_theta = max(4, N // 3)

        for center, radius, alb in sphere_data:
            for ip in range(n_phi):
                for it in range(n_theta):
                    phi = 2.0 * np.pi * (ip + 0.5) / n_phi
                    theta = np.pi * (it + 0.5) / n_theta
                    # Solo el hemisferio visible (y > center_y - radius + eps)
                    normal = np.array([
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)
                    ])
                    # Reorientar: theta=0 → arriba
                    normal_sph = np.array([
                        np.sin(theta) * np.cos(phi),
                        np.cos(theta),
                        np.sin(theta) * np.sin(phi)
                    ])
                    pos = center + radius * normal_sph
                    # Evitar parches dentro de las paredes
                    if np.all(np.abs(pos) < 0.99) and pos[1] > -0.99:
                        sph_area = (4 * np.pi * radius**2) / (n_phi * n_theta)
                        patches.append(pos)
                        n_len = np.linalg.norm(normal_sph)
                        normals.append(normal_sph / max(n_len, 1e-12))
                        areas.append(sph_area)
                        albedos.append(alb.copy())
                        emissions.append(np.zeros(3))

        return (np.array(patches), np.array(normals), np.array(areas),
                np.array(albedos), np.array(emissions))

    def _compute_form_factors(self, patches, normals, areas, objetos, cfg):
        """
        Calcula los factores de forma por muestreo hemisférico con rayos.

        Para cada parche i, dispara rayos en el hemisferio y registra qué
        parche j alcanza cada rayo. F_ij ≈ (hits en j) / total_rays.
        """
        N = len(patches)
        n_samples = cfg.radiosity_form_factor_samples
        rng = np.random.default_rng(cfg.seed + 42)
        eps = cfg.eps

        ff = np.zeros((N, N))

        for i in range(N):
            # Generar direcciones uniformes en el hemisferio de la normal
            ni = np.broadcast_to(normals[i], (n_samples, 3)).copy()
            dirs = uniform_hemisphere(ni, rng)

            # Orígenes: ligeramente desplazados de la superficie
            origins = np.broadcast_to(patches[i] + normals[i] * eps,
                                       (n_samples, 3)).copy()

            # Trazar rayos
            t, obj_idx = scene_intersect(objetos, origins, dirs, eps)
            hit = (obj_idx >= 0) & (t < np.inf)

            if not np.any(hit):
                continue

            # Encontrar los puntos de impacto
            P_hit = origins[hit] + t[hit, None] * dirs[hit]

            # Asignar cada impacto al parche más cercano
            for k in range(P_hit.shape[0]):
                dists = np.linalg.norm(patches - P_hit[k], axis=1)
                j = np.argmin(dists)
                if j != i:
                    ff[i, j] += 1.0

        # Normalizar: F_ij = (hits_ij / total_rays) * pi (corrección hemisférica)
        # La pdf del muestreo uniforme es 1/(2π), y el factor cos se integra a π.
        for i in range(N):
            total = n_samples
            if total > 0:
                ff[i] /= total

        return ff

    def _render_from_patches(self, objetos, patches, patch_normals,
                             radiosity, camera, cfg):
        """
        Renderiza la imagen trazando rayos desde la cámara y asignando
        a cada píxel la radiosidad del parche más cercano al punto de
        impacto.
        """
        H, W = camera.height, camera.width
        Npix = H * W
        eps = cfg.eps

        rng = np.random.default_rng(cfg.seed)
        O, D = camera.generate_primary_rays(rng)

        t, obj_idx = scene_intersect(objetos, O, D, eps)
        hit = obj_idx >= 0

        img = np.zeros((Npix, 3))

        if np.any(hit):
            P_hit = O[hit] + t[hit, None] * D[hit]

            # Para cada punto de impacto, encontrar el parche más cercano
            # Procesamos en lotes para no consumir demasiada memoria
            batch_size = 2000
            hit_indices = np.where(hit)[0]
            for start in range(0, len(hit_indices), batch_size):
                end = min(start + batch_size, len(hit_indices))
                batch_P = P_hit[start:end]
                batch_idx = hit_indices[start:end]

                # Distancia a cada parche
                diffs = batch_P[:, None, :] - patches[None, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                nearest_patch = np.argmin(dists, axis=1)

                img[batch_idx] = radiosity[nearest_patch]

        return img.reshape(H, W, 3)
