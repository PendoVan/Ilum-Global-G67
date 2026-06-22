# -*- coding: utf-8 -*-
"""
====================================================================================
 PATH TRACER OFFLINE (CPU) - CAJA DE CORNELL
 Proyecto: Modelos Globales de Iluminacion - Computacion Visual (UNMSM)
 Integrante 3 - Trazado de Trayectorias (Path Tracing) + Integracion de Monte Carlo
====================================================================================

Este programa resuelve numericamente la ECUACION DE RENDERIZADO de Kajiya (1986):

    L_o(x, w_o) = L_e(x, w_o) + INTEGRAL_sobre_hemisferio[ f_r(x, w_i, w_o) * L_i(x, w_i) * (w_i . n) ] dw_i

donde:
    L_o  = radiancia saliente que llega a la camara desde el punto x
    L_e  = radiancia emitida por la propia superficie (solo la fuente de luz)
    f_r  = BRDF del material (para Lambertiano: albedo / PI)
    L_i  = radiancia incidente (que es, recursivamente, la L_o de otros puntos)
    (w_i . n) = atenuacion por el coseno del angulo respecto a la normal

Como la integral no tiene solucion analitica en escenas reales, se aproxima mediante
INTEGRACION DE MONTE CARLO: se promedian muchas trayectorias aleatorias por pixel.

IMPLEMENTACION:
  - Vectorizada con NumPy: TODOS los rayos de la imagen se procesan en paralelo como
    arreglos (N, 3), evitando los lentos bucles 'for' de Python sobre cada pixel.
  - Muestreo del hemisferio ponderado por coseno (cosine-weighted), lo que cancela
    analiticamente el termino (w_i . n) y el 1/PI de la BRDF, dejando solo el albedo.
  - Ruleta Rusa para terminar trayectorias poco influyentes sin introducir sesgo.
====================================================================================
"""

import time
import numpy as np
from PIL import Image

# ====================================================================================
#  1. PARAMETROS GLOBALES  (lo unico que se necesita tocar para la exposicion)
# ====================================================================================

WIDTH  = 320          # ancho de la imagen en pixeles
HEIGHT = 320          # alto de la imagen en pixeles

SAMPLES_PER_PIXEL = 256    # <-- MUESTRAS POR PIXEL. Subir = menos ruido (1,10,50,100...)
MAX_BOUNCES       = 10     # <-- profundidad maxima de rebotes de cada rayo

# Interruptor para la DEMOSTRACION del contraste:
#   True  -> Iluminacion Global (Path Tracing): luz indirecta, sangrado de color,
#            sombras suaves provenientes de la fuente de area.
#   False -> Iluminacion Local: solo luz directa (luz puntual) + ambiente constante,
#            produce sombras duras/planas y SIN sangrado de color.
ENABLE_GLOBAL_ILLUMINATION = True

# Si True, genera AMBAS imagenes (local y global) en una sola ejecucion, util para
# la diapositiva "antes / despues".
RENDER_BOTH = True

# Si True, ignora SAMPLES_PER_PIXEL y genera la serie de convergencia del ruido.
RENDER_CONVERGENCE_SERIES = False
SPP_SERIES = [1, 10, 50, 100]

USE_RUSSIAN_ROULETTE = True   # terminacion probabilistica de rayos
RR_START_BOUNCE      = 3      # la ruleta rusa solo actua a partir de este rebote

# Segunda esfera: True = espejo (muestra reflexiones, pide la skill del proyecto, pero
# anade ruido de "fireflies"); False = difusa (convergencia mas limpia para la demo).
USE_MIRROR_SPHERE = True

EXPOSURE     = 1.0    # factor de exposicion antes del mapeo de tono
GAMMA        = 2.2    # correccion gamma
AMBIENT_LOCAL = 0.18  # intensidad del termino ambiente constante (solo modo local)

SEED = 2024           # semilla para reproducibilidad del ruido aleatorio
EPS  = 1e-4           # epsilon para evitar auto-interseccion ("acne" de sombra")

# ====================================================================================
#  2. ALGEBRA VECTORIAL  (todas las operaciones trabajan sobre arreglos (N, 3))
# ====================================================================================

def normalize(v):
    """Normaliza un arreglo de vectores (N,3) a longitud unitaria."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)   # evita division por cero
    return v / n

def dot(a, b):
    """Producto punto fila a fila de dos arreglos (N,3) -> (N,)."""
    return np.sum(a * b, axis=-1)

# ====================================================================================
#  3. MATERIALES
#     Cada material se describe por:
#       kind    : 0 = DIFUSO (Lambertiano), 1 = EMISIVO (luz), 2 = ESPEJO (especular)
#       albedo  : reflectividad RGB (cuanta luz refleja en cada canal)
#       emission: radiancia emitida RGB (distinta de 0 solo en las fuentes de luz)
# ====================================================================================

DIFFUSE, EMISSIVE, MIRROR = 0, 1, 2

def material(kind, albedo=(0, 0, 0), emission=(0, 0, 0)):
    return {"kind": kind,
            "albedo": np.array(albedo, dtype=np.float64),
            "emission": np.array(emission, dtype=np.float64)}

# ====================================================================================
#  4. PRIMITIVAS GEOMETRICAS  (con interseccion rayo-objeto vectorizada)
# ====================================================================================

class Sphere:
    """Esfera definida por centro y radio. Interseccion rayo-esfera vectorizada."""
    def __init__(self, center, radius, mat):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.mat = mat
        self.is_sphere = True

    def intersect(self, O, D):
        # Resuelve |O + t*D - C|^2 = r^2  ->  t^2 + 2b t + c = 0  (a=1 pues |D|=1)
        oc = O - self.center                    # (N,3)
        b  = dot(oc, D)                         # (N,)
        c  = dot(oc, oc) - self.radius * self.radius
        disc = b * b - c                        # discriminante
        sq = np.sqrt(np.maximum(disc, 0.0))
        t0 = -b - sq                            # raiz cercana
        t1 = -b + sq                            # raiz lejana
        t = np.where(t0 > EPS, t0, t1)          # elige la primera valida
        valid = (disc > 0.0) & (t > EPS)
        return np.where(valid, t, np.inf)

    def normal_at(self, P):
        return normalize(P - self.center)


class Plane:
    """
    Plano infinito (pared) o cuadrilatero acotado (la fuente de luz del techo).
    Se define por su normal n y el offset c de la ecuacion  n . X = c.
    'bounds' = lista de (eje, minimo, maximo) para acotar el plano (solo la luz).
    """
    def __init__(self, normal, offset, mat, bounds=None):
        self.n = np.array(normal, dtype=np.float64)
        self.c = float(offset)
        self.mat = mat
        self.bounds = bounds
        self.is_sphere = False

    def intersect(self, O, D):
        denom = D @ self.n                       # (N,)
        safe = np.where(np.abs(denom) < 1e-9, 1.0, denom)
        t = (self.c - O @ self.n) / safe         # t de interseccion con el plano
        valid = (np.abs(denom) >= 1e-9) & (t > EPS)
        if self.bounds is not None:
            P = O + t[:, None] * D
            for axis, lo, hi in self.bounds:
                valid &= (P[:, axis] >= lo) & (P[:, axis] <= hi)
        return np.where(valid, t, np.inf)

    def normal_at(self, P):
        # La normal de un plano es constante; se difunde a todos los puntos.
        return np.broadcast_to(self.n, P.shape)


# ====================================================================================
#  5. ESCENA: LA CAJA DE CORNELL
#     Caja cubica en x,y,z in [-1, 1]. La camara mira en +z desde el interior.
#     Pared izquierda ROJA, derecha VERDE, resto BLANCAS. Luz rectangular en el techo.
#     Dos esferas: una difusa (mate) y una especular (espejo).
# ====================================================================================

# Geometria de la fuente de luz rectangular (en el techo, mirando hacia abajo).
LIGHT_Y  = 0.998        # altura de la luz (justo bajo el techo y=1)
LIGHT_HX = 0.33         # semi-ancho en x
LIGHT_HZ = 0.33         # semi-ancho en z

def build_cornell_box():
    rojo    = material(DIFFUSE, albedo=(0.75, 0.20, 0.20))
    verde   = material(DIFFUSE, albedo=(0.20, 0.75, 0.25))
    blanco  = material(DIFFUSE, albedo=(0.78, 0.78, 0.78))
    # La luz: emision intensa para iluminar toda la caja por rebotes.
    luz     = material(EMISSIVE, emission=(7.0, 6.5, 5.5))
    espejo  = material(MIRROR,  albedo=(0.95, 0.95, 0.95))
    mate    = material(DIFFUSE, albedo=(0.80, 0.80, 0.82))

    objetos = [
        # --- Paredes (planos infinitos, validos porque la caja es convexa y cerrada) ---
        Plane(normal=( 1, 0, 0), offset=-1, mat=rojo),    # pared izquierda  x=-1
        Plane(normal=(-1, 0, 0), offset=-1, mat=verde),   # pared derecha    x=+1
        Plane(normal=( 0, 1, 0), offset=-1, mat=blanco),  # piso             y=-1
        Plane(normal=( 0,-1, 0), offset=-1, mat=blanco),  # techo            y=+1
        Plane(normal=( 0, 0,-1), offset=-1, mat=blanco),  # pared del fondo  z=+1
        Plane(normal=( 0, 0, 1), offset=-1, mat=blanco),  # pared frontal    z=-1 (tras la camara)

        # --- Fuente de luz: cuadrilatero emisivo justo bajo el techo ---
        Plane(normal=(0, -1, 0), offset=LIGHT_Y, mat=luz,
              bounds=[(0, -LIGHT_HX, LIGHT_HX), (2, -LIGHT_HZ, LIGHT_HZ)]),

        # --- Objetos dentro de la caja (apoyados sobre el piso y=-1) ---
        Sphere(center=(-0.40, -0.62, 0.35), radius=0.38, mat=mate),    # esfera difusa
        Sphere(center=( 0.45, -0.65, -0.10), radius=0.35,              # esfera 2
               mat=espejo if USE_MIRROR_SPHERE else mate),
    ]

    # Informacion de la fuente de luz para el muestreo directo (Next Event Estimation)
    light_info = {
        "index": 6,                                   # indice del objeto-luz en la lista
        "y": LIGHT_Y,
        "hx": LIGHT_HX, "hz": LIGHT_HZ,
        "normal": np.array([0.0, -1.0, 0.0]),         # la luz emite hacia abajo
        "emission": luz["emission"].copy(),
        "area": (2 * LIGHT_HX) * (2 * LIGHT_HZ),      # area del rectangulo emisor
    }
    return objetos, light_info


# Empaqueta las propiedades de los materiales en arreglos para acceso vectorizado.
def pack_materials(objetos):
    M = len(objetos)
    kinds    = np.zeros(M, dtype=np.int64)
    albedos  = np.zeros((M, 3))
    emissions = np.zeros((M, 3))
    for i, o in enumerate(objetos):
        kinds[i]     = o.mat["kind"]
        albedos[i]   = o.mat["albedo"]
        emissions[i] = o.mat["emission"]
    return kinds, albedos, emissions


# ====================================================================================
#  6. CAMARA: genera los rayos primarios de toda la imagen de una sola vez
# ====================================================================================

CAM_ORIGIN = np.array([0.0, 0.0, -0.85])  # camara dentro de la caja, junto al frente
FOV_SCALE  = 0.72                          # escala del campo de vision (tan del semiangulo)

def generate_primary_rays(rng):
    """
    Devuelve (origenes, direcciones) de todos los rayos primarios con jitter
    aleatorio dentro de cada pixel (anti-aliasing por supermuestreo).
    """
    aspect = WIDTH / HEIGHT
    # Coordenadas de pixel + jitter sub-pixel en [0,1)
    jx = rng.random((HEIGHT, WIDTH))
    jy = rng.random((HEIGHT, WIDTH))
    xs = (np.arange(WIDTH) + jx) / WIDTH          # (H,W) en [0,1]
    ys = (np.arange(HEIGHT)[:, None] + jy) / HEIGHT
    # Mapea a coordenadas de pantalla [-1,1]; el eje y se invierte (fila 0 = arriba)
    px = (2.0 * xs - 1.0) * FOV_SCALE * aspect
    py = (1.0 - 2.0 * ys) * FOV_SCALE
    # La camara mira hacia +z; el plano de imagen esta a distancia focal 1.
    dirs = np.stack([px, py, np.ones_like(px)], axis=-1).reshape(-1, 3)
    dirs = normalize(dirs)
    origins = np.broadcast_to(CAM_ORIGIN, dirs.shape).copy()
    return origins, dirs


# ====================================================================================
#  7. INTERSECCION DE LA ESCENA  (el rayo mas cercano por cada pixel)
# ====================================================================================

def scene_intersect(objetos, O, D):
    """
    Para N rayos, encuentra el objeto mas cercano impactado.
    Devuelve: nearest_t (N,), nearest_obj (N,) [-1 si no hay impacto].
    """
    N = O.shape[0]
    nearest_t = np.full(N, np.inf)
    nearest_obj = np.full(N, -1, dtype=np.int64)
    for idx, obj in enumerate(objetos):
        t = obj.intersect(O, D)
        closer = t < nearest_t
        nearest_t[closer] = t[closer]
        nearest_obj[closer] = idx
    return nearest_t, nearest_obj


def gather_normals(objetos, nearest_obj, P):
    """Calcula la normal de la superficie en el punto de impacto de cada rayo."""
    Nn = np.zeros_like(P)
    for idx, obj in enumerate(objetos):
        mask = (nearest_obj == idx)
        if not np.any(mask):
            continue
        Nn[mask] = obj.normal_at(P[mask])
    return Nn


# ====================================================================================
#  8. MUESTREO DEL HEMISFERIO PONDERADO POR COSENO
#     ------------------------------------------------------------------------------
#     Para una superficie difusa (Lambertiana), la integral de Monte Carlo del
#     termino  f_r * L_i * (w_i . n)  se estima con UNA direccion aleatoria w_i.
#     Si w_i se muestrea UNIFORME, el estimador es:  (albedo/PI) * L_i * cos / pdf,
#     con pdf = 1/(2PI). Eso obliga a multiplicar por cos en cada rebote.
#
#     En cambio, si muestreamos w_i con pdf = cos/PI (ponderada por coseno), entonces:
#         estimador = (albedo/PI) * L_i * cos / (cos/PI) = albedo * L_i
#     El coseno y el PI se CANCELAN analiticamente: basta multiplicar por el albedo.
#     Esto reduce el ruido y simplifica el codigo (ver skills del proyecto).
# ====================================================================================

def cosine_weighted_hemisphere(n, rng):
    """Genera direcciones aleatorias (N,3) alrededor de la normal n, ~ coseno."""
    N = n.shape[0]
    u1 = rng.random(N)
    u2 = rng.random(N)
    r = np.sqrt(u2)                      # radio en el disco
    phi = 2.0 * np.pi * u1              # angulo azimutal
    # Construye una base ortonormal (tangent, bitangent, n) por rayo.
    w = n
    # Vector auxiliar no paralelo a w para el producto cruz.
    aux = np.where(np.abs(w[:, 0:1]) > 0.9,
                   np.array([0.0, 1.0, 0.0]),
                   np.array([1.0, 0.0, 0.0]))
    tangent = normalize(np.cross(aux, w))
    bitangent = np.cross(w, tangent)
    # Direccion final en el sistema (tangent, bitangent, w).
    d = (tangent * (np.cos(phi) * r)[:, None]
         + bitangent * (np.sin(phi) * r)[:, None]
         + w * np.sqrt(np.maximum(0.0, 1.0 - u2))[:, None])
    return normalize(d)


def reflect(D, n):
    """Reflexion especular perfecta: D - 2 (D.n) n."""
    return D - 2.0 * dot(D, n)[:, None] * n


# ====================================================================================
#  9. PATH TRACER  (ILUMINACION GLOBAL) - vectorizado sobre toda la imagen
#     ------------------------------------------------------------------------------
#     Combina dos estrategias para estimar la ecuacion de renderizado con bajo ruido:
#       (1) MUESTREO DE BRDF: en cada rebote difuso se elige una direccion ~ coseno
#           para propagar el camino (capta la iluminacion indirecta de toda la escena).
#       (2) MUESTREO DIRECTO DE LA FUENTE  (Next Event Estimation, NEE): en cada vertice
#           difuso se conecta explicitamente con un punto aleatorio de la luz mediante
#           un rayo de sombra. Esto evita esperar a que el camino "encuentre" la luz
#           por azar y reduce drasticamente la varianza (ver informe, seccion 3.3).
#     Para no contar dos veces la luz, la emision directa solo se suma cuando el rayo
#     proviene de la camara o de un rebote especular (donde NO se aplico NEE).
# ====================================================================================

def sample_light_points(n, rng, light):
    """Muestrea n puntos uniformes sobre el rectangulo emisor (pdf = 1/area)."""
    qx = rng.uniform(-light["hx"], light["hx"], n)
    qz = rng.uniform(-light["hz"], light["hz"], n)
    qy = np.full(n, light["y"])
    return np.stack([qx, qy, qz], axis=-1)


def render_global(objetos, light, rng):
    """
    Trazado de trayectorias progresivo con NEE. Acumula SAMPLES_PER_PIXEL pasadas
    de Monte Carlo y las promedia. Cada pasada procesa TODOS los pixeles en paralelo.
    """
    kinds, albedos, emissions = pack_materials(objetos)
    H, W = HEIGHT, WIDTH
    Npix = H * W
    accum = np.zeros((Npix, 3))
    Le = light["emission"]
    n_light = light["normal"]
    area = light["area"]

    for s in range(SAMPLES_PER_PIXEL):
        O, D = generate_primary_rays(rng)        # rayos primarios (con jitter)
        throughput = np.ones((Npix, 3))          # producto de BRDFs del camino
        radiance   = np.zeros((Npix, 3))         # radiancia recogida
        active     = np.ones(Npix, dtype=bool)   # rayos aun "vivos"
        # Flag: ¿debe sumarse la emision si el rayo golpea la luz?
        # True para el rayo primario y tras rebotes especulares; False tras difusos
        # (porque en los difusos la luz ya se conto con NEE).
        spec = np.ones(Npix, dtype=bool)

        for bounce in range(MAX_BOUNCES):
            if not np.any(active):
                break
            act_idx = np.where(active)[0]
            Oa, Da = O[act_idx], D[act_idx]
            t, obj = scene_intersect(objetos, Oa, Da)

            hit = obj >= 0
            active[act_idx[~hit]] = False         # rayos que escapan: fondo negro
            if not np.any(hit):
                continue

            gidx = act_idx[hit]                   # indices globales con impacto
            Oa, Da, t, obj = Oa[hit], Da[hit], t[hit], obj[hit]
            P  = Oa + t[:, None] * Da
            Nn = gather_normals(objetos, obj, P)
            facing = (dot(Da, Nn) > 0)
            Nn[facing] = -Nn[facing]              # normal siempre contra el rayo
            obj_kind = kinds[obj]

            # ---------- (a) IMPACTO EN LA FUENTE DE LUZ ----------
            emis = (obj_kind == EMISSIVE)
            if np.any(emis):
                ge = gidx[emis]
                add = spec[ge]                    # solo si NEE no lo conto antes
                radiance[ge[add]] += throughput[ge[add]] * emissions[obj[emis][add]]
                active[ge] = False                # el camino termina en la luz

            # ---------- (b) ESPEJO: reflexion especular perfecta ----------
            mir = (obj_kind == MIRROR)
            if np.any(mir):
                gm = gidx[mir]
                D[gm] = reflect(Da[mir], Nn[mir])
                O[gm] = P[mir] + Nn[mir] * EPS
                throughput[gm] *= albedos[obj[mir]]
                spec[gm] = True                   # la siguiente luz vista SI cuenta

            # ---------- (c) DIFUSO: NEE + nuevo rebote ~ coseno ----------
            dif = (obj_kind == DIFFUSE)
            if np.any(dif):
                gd = gidx[dif]
                Pd, Ndn, ad = P[dif], Nn[dif], albedos[obj[dif]]

                # --- (c.1) Next Event Estimation: contribucion directa de la luz ---
                Q = sample_light_points(Pd.shape[0], rng, light)
                to_l = Q - Pd
                d2 = dot(to_l, to_l)
                dist = np.sqrt(d2)
                wi = to_l / dist[:, None]
                cos_s = np.maximum(0.0, dot(Ndn, wi))            # (n . wi) en la superficie
                cos_l = np.maximum(0.0, dot(-wi, n_light))       # coseno en la luz
                # Rayo de sombra: ¿la luz es visible desde Pd?
                sh_O = Pd + Ndn * EPS
                st, _ = scene_intersect(objetos, sh_O, wi)
                visible = st >= (dist - 1e-3)
                # Estimador de area: f_r * Le * (cos_s * cos_l / d2) * area
                # con f_r = albedo/PI  y  pdf = 1/area.
                G = (cos_s * cos_l) / np.maximum(d2, 1e-6)
                contrib = (throughput[gd] * (ad / np.pi)
                           * Le[None, :] * (G * area)[:, None])
                contrib[~visible] = 0.0
                radiance[gd] += contrib

                # --- (c.2) Propagacion del camino con muestreo por coseno ---
                D[gd] = cosine_weighted_hemisphere(Ndn, rng)
                O[gd] = Pd + Ndn * EPS
                throughput[gd] *= ad              # el coseno y el 1/PI se cancelan
                spec[gd] = False                  # la luz ya se conto con NEE

            # ---------- (d) RULETA RUSA: termina trayectorias debiles sin sesgo ----------
            if USE_RUSSIAN_ROULETTE and bounce >= RR_START_BOUNCE:
                cont = gidx[dif | mir]
                if cont.size:
                    p = np.clip(np.max(throughput[cont], axis=1), 0.05, 1.0)
                    keep = rng.random(cont.size) < p
                    throughput[cont[keep]] /= p[keep, None]
                    active[cont[~keep]] = False

        accum += radiance

    return (accum / SAMPLES_PER_PIXEL).reshape(H, W, 3)


# ====================================================================================
#  10. RENDER LOCAL  (ILUMINACION LOCAL para el contraste de la demostracion)
#      Solo luz directa desde una luz PUNTUAL (centro de la fuente) + ambiente
#      constante. Sin rebotes indirectos => sombras duras y SIN sangrado de color.
# ====================================================================================

LIGHT_CENTER = np.array([0.0, LIGHT_Y, 0.0])   # centro de la fuente de luz
LIGHT_POWER  = np.array([1.6, 1.5, 1.35])      # intensidad de la luz puntual equivalente

def render_local(objetos, rng):
    kinds, albedos, emissions = pack_materials(objetos)
    H, W = HEIGHT, WIDTH
    accum = np.zeros((H * W, 3))
    n_aa = max(4, SAMPLES_PER_PIXEL // 8)   # pocas muestras: el modo local no tiene ruido MC

    for s in range(n_aa):
        O, D = generate_primary_rays(rng)
        t, obj = scene_intersect(objetos, O, D)
        hit = obj >= 0
        col = np.zeros((H * W, 3))

        if np.any(hit):
            Oi, Di, ti, oi = O[hit], D[hit], t[hit], obj[hit]
            P = Oi + ti[:, None] * Di
            Nn = gather_normals(objetos, oi, P)
            facing = (dot(Di, Nn) > 0)
            Nn[facing] = -Nn[facing]
            kind = kinds[oi]
            alb = albedos[oi]

            c = np.zeros((P.shape[0], 3))

            # La fuente se ve a si misma como blanca brillante.
            emis = (kind == EMISSIVE)
            c[emis] = emissions[oi[emis]] * 0.06   # atenuada para no saturar

            # Difuso y espejo (en modo local el espejo se trata como gris difuso).
            surf = (kind == DIFFUSE) | (kind == MIRROR)
            if np.any(surf):
                Ps, Ns, As = P[surf], Nn[surf], alb[surf]
                L = LIGHT_CENTER - Ps                      # vector al punto de luz
                dist2 = dot(L, L)
                Ld = normalize(L)
                cos_s = np.maximum(0.0, dot(Ns, Ld))       # termino lambertiano (n . wi)

                # Rayo de sombra: ¿hay geometria entre el punto y la luz?
                shadow_O = Ps + Ns * EPS
                st, sobj = scene_intersect(objetos, shadow_O, Ld)
                dist = np.sqrt(dist2)
                # Visible si el primer impacto es la propia luz o esta mas alla de ella.
                visible = (sobj < 0) | (st >= dist - 1e-3) | (kinds[np.where(sobj < 0, 0, sobj)] == EMISSIVE)

                atten = 1.0 / np.maximum(dist2, 1e-3)      # atenuacion 1/r^2
                direct = (As * LIGHT_POWER[None, :]
                          * (atten * cos_s)[:, None])
                direct[~visible] = 0.0
                ambient = As * AMBIENT_LOCAL               # termino ambiente constante
                shaded = direct + ambient

                tmp = c[surf]
                tmp[:] = shaded
                c[surf] = tmp

            col[hit] = c

        accum += col

    return (accum / n_aa).reshape(H, W, 3)


# ====================================================================================
#  11. MAPEO DE TONO Y GUARDADO
# ====================================================================================

def to_image(hdr):
    """Convierte radiancia HDR (float) a imagen 8-bit con exposicion + gamma."""
    ldr = hdr * EXPOSURE
    ldr = ldr / (1.0 + ldr)                      # mapeo de tono de Reinhard
    ldr = np.clip(ldr, 0.0, 1.0) ** (1.0 / GAMMA)  # correccion gamma
    return (ldr * 255.0 + 0.5).astype(np.uint8)


def save_image(hdr, filename):
    Image.fromarray(to_image(hdr), mode="RGB").save(filename)
    print(f"   -> Imagen guardada: {filename}")


# ====================================================================================
#  12. PROGRAMA PRINCIPAL  (con metricas de tiempo para la tabla del informe)
# ====================================================================================

def render_and_report(objetos, light, modo_global, etiqueta):
    rng = np.random.default_rng(SEED)
    print(f"\n[{etiqueta}]  {WIDTH}x{HEIGHT}px | "
          f"{'GLOBAL (Path Tracing)' if modo_global else 'LOCAL (directa)'} | "
          f"spp={SAMPLES_PER_PIXEL} | rebotes={MAX_BOUNCES}")
    t0 = time.perf_counter()
    if modo_global:
        hdr = render_global(objetos, light, rng)
    else:
        hdr = render_local(objetos, rng)
    elapsed = time.perf_counter() - t0

    # ---- METRICAS para la exposicion ----
    if modo_global:
        per_sample = elapsed / SAMPLES_PER_PIXEL
        print(f"   Tiempo total de render : {elapsed:8.2f} s")
        print(f"   Tiempo por muestra     : {per_sample*1000:8.2f} ms")
        print(f"   Rayos primarios/pasada : {WIDTH*HEIGHT:,}")
        print(f"   Muestras totales       : {WIDTH*HEIGHT*SAMPLES_PER_PIXEL:,}")
    else:
        print(f"   Tiempo total de render : {elapsed:8.2f} s")
    return hdr, elapsed


def main():
    objetos, light = build_cornell_box()
    print("=" * 70)
    print(" PATH TRACER - CAJA DE CORNELL (CPU / NumPy)")
    print("=" * 70)

    if RENDER_CONVERGENCE_SERIES:
        # Serie de imagenes con spp creciente para mostrar la reduccion del ruido.
        global SAMPLES_PER_PIXEL
        print("\nModo: SERIE DE CONVERGENCIA DEL RUIDO")
        for spp in SPP_SERIES:
            SAMPLES_PER_PIXEL = spp
            hdr, _ = render_and_report(objetos, light, True, f"convergencia spp={spp}")
            save_image(hdr, f"cornell_global_spp{spp:04d}.png")
        return

    if RENDER_BOTH:
        hdr_local, _  = render_and_report(objetos, light, False, "LOCAL")
        save_image(hdr_local, "cornell_local.png")
        hdr_global, _ = render_and_report(objetos, light, True, "GLOBAL")
        save_image(hdr_global, "cornell_global.png")
    else:
        hdr, _ = render_and_report(objetos, light, ENABLE_GLOBAL_ILLUMINATION,
                                   "RENDER")
        nombre = "cornell_global.png" if ENABLE_GLOBAL_ILLUMINATION else "cornell_local.png"
        save_image(hdr, nombre)

    print("\nListo. Renderizado completado.\n")


if __name__ == "__main__":
    main()
