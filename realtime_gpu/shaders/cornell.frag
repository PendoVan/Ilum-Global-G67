#version 330
// ==========================================================================
// Trazador GPU de la Caja de Cornell — 6 modelos de iluminacion en un shader
//   0 = LOCAL        : Phong (difusa + especular) + ambiente, sin rebotes
//   1 = RAYTRACING   : Whitted recursivo (espejo/vidrio + sombras duras)
//   2 = PATHTRACING  : Monte Carlo progresivo (NEE + muestreo coseno + RR)
//   3 = REALTIME_AO  : iluminacion local + oclusion ambiental aproximada
//   4 = RADIOSITY    : interreflexion difusa determinista (sin ruido),
//                       solo superficies difusas (Goral et al., 1984)
//   5 = PHOTONMAP    : vidrio refractivo + caustica analitica aproximada
//                       (no es un mapa de fotones real de dos pasadas;
//                       aproxima su efecto distintivo de forma eficiente)
// La escena replica src/core/scene.py (paredes en [-1,1]^3, luz rectangular
// en el techo, dos esferas) para comparar contra los renders offline en CPU.
// ==========================================================================

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_prev;
uniform float u_blend;          // 1.0 = reemplaza, 1/frame = acumula (path tracing)
uniform int u_mode;
uniform vec2 u_resolution;

uniform vec3 u_cam_pos;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform vec3 u_cam_forward;
uniform float u_fov_scale;

uniform float u_ambient;
uniform float u_light_intensity;
uniform int u_max_bounces;
uniform int u_ao_samples;
uniform float u_ao_radius;
uniform int u_sphere2_material;  // 0 = difusa, 1 = espejo, 2 = vidrio
uniform uint u_frame_seed;

// -------------------------------------------------------------------------
// RNG: PCG hash (Jarzynski & Olano), suficiente calidad para Monte Carlo
// -------------------------------------------------------------------------
uint rng_state;

uint pcg_hash(uint v) {
    v = v * 747796405u + 2891336453u;
    uint w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}

float rand() {
    rng_state = pcg_hash(rng_state);
    return float(rng_state) / 4294967296.0;
}

vec2 rand2() { return vec2(rand(), rand()); }

// -------------------------------------------------------------------------
// Escena: caja [-1,1]^3, luz rectangular, dos esferas (igual que scene.py,
// con colores mas saturados para que el sangrado de color se note en vivo).
// -------------------------------------------------------------------------
const float LIGHT_Y = 0.998;
const float LIGHT_HX = 0.33;
const float LIGHT_HZ = 0.33;
// Tono calido (como la foto de referencia de la Cornell Box clasica:
// blanco ligeramente amarillo/anaranjado, no un blanco neutro de estudio).
const vec3 LIGHT_EMISSION = vec3(13.0, 10.3, 6.6);
const float LIGHT_AREA = (2.0 * LIGHT_HX) * (2.0 * LIGHT_HZ);

const vec3 SPHERE1_C = vec3(-0.40, -0.62, 0.35);
const float SPHERE1_R = 0.38;
const vec3 SPHERE2_C = vec3(0.45, -0.65, -0.10);
const float SPHERE2_R = 0.35;

const float PI = 3.14159265359;
const float EPS = 1e-3;
const float GLASS_IOR = 1.5;

// matType: 0 = difuso, 1 = espejo, 2 = vidrio, 3 = emisivo (luz)
struct Hit {
    float t;
    vec3 p;
    vec3 n;
    vec3 albedo;
    vec3 emission;
    int matType;
    float shininess;   // 1.0 en las esferas, 0.0 en paredes planas (sin resalte Phong)
};

bool intersectSphere(vec3 ro, vec3 rd, vec3 c, float r, out float t) {
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    float cc = dot(oc, oc) - r * r;
    float disc = b * b - cc;
    if (disc < 0.0) return false;
    float s = sqrt(disc);
    float t0 = -b - s;
    float t1 = -b + s;
    t = (t0 > EPS) ? t0 : t1;
    return t > EPS;
}

bool intersectLightQuad(vec3 ro, vec3 rd, out float t) {
    if (abs(rd.y) < 1e-6) return false;
    t = (LIGHT_Y - ro.y) / rd.y;
    if (t <= EPS) return false;
    vec3 p = ro + rd * t;
    return abs(p.x) <= LIGHT_HX && abs(p.z) <= LIGHT_HZ;
}

// Paredes = interior de una caja AABB; al estar la camara siempre dentro,
// el punto de impacto es la salida mas cercana (min de los "t lejanos").
bool intersectWalls(vec3 ro, vec3 rd, out float t, out vec3 n, out vec3 albedo) {
    vec3 invD = 1.0 / rd;
    vec3 t1 = (vec3(-1.0) - ro) * invD;
    vec3 t2 = (vec3(1.0) - ro) * invD;
    vec3 tFar = max(t1, t2);
    t = min(min(tFar.x, tFar.y), tFar.z);
    if (t <= EPS) return false;

    // Piso/techo/fondo en un crema calido (igual a la foto de referencia de
    // la Cornell Box clasica) en vez de un gris neutro de laboratorio.
    const vec3 WARM_WALL = vec3(0.86, 0.78, 0.63);
    if (t == tFar.x) {
        n = vec3(rd.x > 0.0 ? -1.0 : 1.0, 0.0, 0.0);
        albedo = (rd.x > 0.0) ? vec3(0.08, 0.62, 0.12) : vec3(0.72, 0.07, 0.06);
    } else if (t == tFar.y) {
        n = vec3(0.0, rd.y > 0.0 ? -1.0 : 1.0, 0.0);
        albedo = WARM_WALL;
    } else {
        n = vec3(0.0, 0.0, rd.z > 0.0 ? -1.0 : 1.0);
        albedo = WARM_WALL;
    }
    return true;
}

bool sceneIntersect(vec3 ro, vec3 rd, out Hit h) {
    float bestT = 1e30;
    bool found = false;
    float t;
    vec3 n, albedo;

    if (intersectWalls(ro, rd, t, n, albedo)) {
        bestT = t; found = true;
        h.n = n; h.albedo = albedo; h.emission = vec3(0.0); h.matType = 0; h.shininess = 0.0;
    }
    if (intersectLightQuad(ro, rd, t) && t < bestT) {
        bestT = t; found = true;
        h.n = vec3(0.0, -1.0, 0.0); h.albedo = vec3(0.0);
        h.emission = LIGHT_EMISSION; h.matType = 3; h.shininess = 0.0;
    }
    if (intersectSphere(ro, rd, SPHERE1_C, SPHERE1_R, t) && t < bestT) {
        bestT = t; found = true;
        vec3 p = ro + rd * t;
        h.n = normalize(p - SPHERE1_C);
        h.albedo = vec3(0.80, 0.76, 0.66); h.emission = vec3(0.0); h.matType = 0; h.shininess = 1.0;
    }
    if (intersectSphere(ro, rd, SPHERE2_C, SPHERE2_R, t) && t < bestT) {
        bestT = t; found = true;
        vec3 p = ro + rd * t;
        h.n = normalize(p - SPHERE2_C);
        h.emission = vec3(0.0);
        h.matType = u_sphere2_material;
        h.shininess = 1.0;
        h.albedo = (u_sphere2_material == 0) ? vec3(0.80, 0.76, 0.66)
                 : (u_sphere2_material == 1) ? vec3(0.97)
                 : vec3(0.98);
    }

    if (found) { h.t = bestT; h.p = ro + rd * bestT; }
    return found;
}

// Sombra dura hacia el centro de la luz (aproximacion puntual, como en
// los modelos locales / Whitted clasico descritos en el informe).
bool occluded(vec3 p, vec3 n, vec3 lightPoint) {
    vec3 dir = lightPoint - p;
    float dist = length(dir);
    dir /= dist;
    vec3 ro = p + n * EPS;
    Hit h;
    if (sceneIntersect(ro, dir, h)) {
        if (h.matType != 3 && h.t < dist - EPS) return true;
    }
    return false;
}

// Luz directa con sombra SUAVE: se muestrea el area de la luz en 4 puntos
// fijos (estratificados, deterministas -> sin ruido) en vez de un unico
// punto central, para que la penumbra de las sombras se vea gradual (como
// en la foto de referencia de la Cornell Box) en todos los modos, no solo
// en Path Tracing/Radiosidad.
vec3 directLight(vec3 p, vec3 n, vec3 albedo) {
    const int LSAMPLES = 4;
    vec2 offs[4] = vec2[4](vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(-0.5, 0.5), vec2(0.5, 0.5));
    vec3 Li = LIGHT_EMISSION * u_light_intensity;
    vec3 result = vec3(0.0);
    for (int i = 0; i < LSAMPLES; i++) {
        vec3 lp = vec3(offs[i].x * LIGHT_HX, LIGHT_Y, offs[i].y * LIGHT_HZ);
        vec3 toL = lp - p;
        float dist2 = max(dot(toL, toL), 1e-4);
        float dist = sqrt(dist2);
        vec3 l = toL / dist;
        float ndotl = max(dot(n, l), 0.0);
        if (ndotl <= 0.0 || occluded(p, n, lp)) continue;
        result += albedo * Li * ndotl * (LIGHT_AREA / float(LSAMPLES)) / (PI * dist2);
    }
    return min(result, vec3(60.0));
}

// Resalte especular de Phong hacia la luz (clasico modelo local de la
// seccion 2.1 del informe: difusa + especular, no solo difusa).
vec3 phongSpecular(vec3 p, vec3 n, vec3 viewDir) {
    vec3 lightCenter = vec3(0.0, LIGHT_Y, 0.0);
    vec3 toL = lightCenter - p;
    vec3 l = normalize(toL);
    if (dot(n, l) <= 0.0) return vec3(0.0);
    if (occluded(p, n, lightCenter)) return vec3(0.0);
    vec3 r = reflect(-l, n);
    float spec = pow(max(dot(r, viewDir), 0.0), 56.0);
    return vec3(1.0, 0.98, 0.92) * spec * 0.55 * u_light_intensity;
}

vec3 cosineSampleHemisphere(vec3 n) {
    vec2 r = rand2();
    float r1 = 2.0 * PI * r.x;
    float r2 = r.y;
    float r2s = sqrt(r2);
    vec3 w = n;
    vec3 axis = (abs(w.x) > 0.1) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 u = normalize(cross(axis, w));
    vec3 v = cross(w, u);
    return normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1.0 - r2));
}

float schlickFresnel(float cosTheta, float ior) {
    float r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);
}

// Luz de relleno minima: evita que las reflexiones de zonas en sombra se
// vean perfectamente negras (un espejo real siempre refleja algo de la
// iluminacion ambiental del entorno, no un vacio absoluto).
const vec3 FILL_LIGHT = vec3(0.045, 0.045, 0.05);

// Sombreado "terminal" (sin mas rebotes) de un impacto: usado para evaluar
// rapidamente que se ve al final de un rayo de reflejo/refraccion.
vec3 shadeSurfaceLocal(vec3 viewDir, Hit h) {
    if (h.matType == 3) return h.emission;
    vec3 albedo = (h.matType == 0) ? h.albedo : vec3(0.85);
    vec3 spec = phongSpecular(h.p, h.n, viewDir) * h.shininess;
    return directLight(h.p, h.n, albedo) + spec + u_ambient * albedo + FILL_LIGHT * albedo;
}

// Espejo con micro-rugosidad leve: promedia unas pocas direcciones fijas
// alrededor de la reflexion perfecta (deterministas, sin parpadeo) para
// evitar el aspecto "espejo perfecto de render barato" y acercarse a un
// modelo de microfacetas simplificado (Cook-Torrance/GGX, seccion 2.2).
vec3 shadeMirrorGlossy(vec3 rd, Hit h) {
    vec3 baseDir = reflect(rd, h.n);
    vec3 axis = (abs(baseDir.x) > 0.1) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tu = normalize(cross(axis, baseDir));
    vec3 tv = cross(baseDir, tu);
    const int GS = 5;
    vec2 offs[5] = vec2[5](vec2(0.0, 0.0), vec2(0.55, 0.0), vec2(-0.55, 0.0), vec2(0.0, 0.55), vec2(0.0, -0.55));
    const float ROUGH = 0.045;
    vec3 acc = vec3(0.0);
    for (int i = 0; i < GS; i++) {
        vec3 dir = normalize(baseDir + (tu * offs[i].x + tv * offs[i].y) * ROUGH);
        Hit rh;
        if (sceneIntersect(h.p + h.n * EPS, dir, rh)) acc += shadeSurfaceLocal(-dir, rh);
        else acc += FILL_LIGHT * 4.0;
    }
    return h.albedo * (acc / float(GS));
}

// Vidrio tipo Whitted: combina un rayo reflejado y uno refractado segun
// Fresnel (Schlick), con un unico nivel adicional de trazado (eficiente).
vec3 shadeGlassWhitted(vec3 rd, Hit h) {
    vec3 n = h.n;
    float cosi = clamp(dot(-rd, n), -1.0, 1.0);
    float eta = 1.0 / GLASS_IOR;
    if (cosi < 0.0) { n = -n; cosi = -cosi; eta = GLASS_IOR; }
    float fresnel = schlickFresnel(cosi, GLASS_IOR);

    vec3 reflDir = reflect(rd, n);
    vec3 refrDir = refract(rd, n, eta);
    bool tir = (refrDir == vec3(0.0));

    Hit rh, fh;
    vec3 colR = vec3(0.0), colF = vec3(0.0);
    if (sceneIntersect(h.p + n * EPS, reflDir, rh)) colR = shadeSurfaceLocal(-reflDir, rh);
    if (!tir) {
        if (sceneIntersect(h.p - n * EPS, refrDir, fh)) colF = shadeSurfaceLocal(-refrDir, fh);
    } else {
        colF = colR;
    }
    return mix(colF, colR, tir ? 1.0 : fresnel) * 0.96;
}

// -------------------------------------------------------------------------
// Modo 0 — Iluminacion LOCAL (Phong difuso + especular + ambiente, sin GI)
// -------------------------------------------------------------------------
vec3 shadeLocal(vec3 ro, vec3 rd) {
    Hit h;
    if (!sceneIntersect(ro, rd, h)) return vec3(0.0);
    if (h.matType == 3) return h.emission;
    // El modelo local no resuelve reflejos/refracciones: toda superficie
    // se trata como difusa, evidenciando la limitacion descrita en 2.1.
    vec3 albedo = (h.matType == 0) ? h.albedo : vec3(0.85);
    vec3 direct = directLight(h.p, h.n, albedo);
    vec3 spec = phongSpecular(h.p, h.n, -rd) * h.shininess;
    vec3 ambient = u_ambient * albedo;
    return direct + spec + ambient;
}

// -------------------------------------------------------------------------
// Modo 1 — RAY TRACING recursivo (Whitted, 1980): espejo/vidrio + sombras duras
// -------------------------------------------------------------------------
vec3 shadeRaytrace(vec3 ro, vec3 rd) {
    vec3 color = vec3(0.0);
    vec3 throughput = vec3(1.0);
    for (int b = 0; b < u_max_bounces; b++) {
        Hit h;
        if (!sceneIntersect(ro, rd, h)) break;
        if (h.matType == 3) { color += throughput * h.emission; break; }
        if (h.matType == 1) {
            throughput *= h.albedo;
            ro = h.p + h.n * EPS;
            rd = reflect(rd, h.n);
            continue;
        }
        if (h.matType == 2) {
            color += throughput * shadeGlassWhitted(rd, h);
            break;
        }
        vec3 direct = directLight(h.p, h.n, h.albedo);
        vec3 spec = phongSpecular(h.p, h.n, -rd) * h.shininess;
        color += throughput * (direct + spec + u_ambient * h.albedo);
        break;
    }
    return color;
}

// -------------------------------------------------------------------------
// Modo 2 — PATH TRACING progresivo (Kajiya, 1986): Monte Carlo + NEE + RR
// -------------------------------------------------------------------------
vec3 shadePathtrace(vec3 ro, vec3 rd) {
    vec3 color = vec3(0.0);
    vec3 throughput = vec3(1.0);

    for (int b = 0; b < u_max_bounces; b++) {
        Hit h;
        if (!sceneIntersect(ro, rd, h)) break;

        if (h.matType == 3) {
            if (b == 0) color += throughput * h.emission;
            break;
        }

        if (h.matType == 1) {
            throughput *= h.albedo;
            ro = h.p + h.n * EPS;
            rd = reflect(rd, h.n);
            continue;
        }

        if (h.matType == 2) {
            vec3 n = h.n;
            float cosi = clamp(dot(-rd, n), -1.0, 1.0);
            float eta = 1.0 / GLASS_IOR;
            if (cosi < 0.0) { n = -n; cosi = -cosi; eta = GLASS_IOR; }
            float fresnel = schlickFresnel(cosi, GLASS_IOR);
            vec3 refrDir = refract(rd, n, eta);
            bool tir = (refrDir == vec3(0.0));
            if (tir || rand() < fresnel) {
                rd = reflect(rd, h.n);
            } else {
                rd = refrDir;
            }
            throughput *= 0.97;
            ro = h.p + rd * (2.0 * EPS);
            continue;
        }

        // Next-event estimation: muestreo directo de un punto en el area de luz
        vec2 r = rand2();
        vec3 lp = vec3((r.x * 2.0 - 1.0) * LIGHT_HX, LIGHT_Y, (r.y * 2.0 - 1.0) * LIGHT_HZ);
        vec3 toL = lp - h.p;
        float dist2 = dot(toL, toL);
        float dist = sqrt(dist2);
        vec3 l = toL / dist;
        float ndotl = max(dot(h.n, l), 0.0);
        float ndotlLight = max(dot(vec3(0.0, -1.0, 0.0), -l), 0.0);

        // ndotlLight requiere un piso minimo: con angulos casi rasantes a la
        // luz, pdfSolidAngle tiende a 0 y la contribucion explota a valores
        // enormes (un solo frame con eso "envenena" para siempre el promedio
        // progresivo, porque mix(prev, +inf, blend) nunca vuelve a bajar).
        if (ndotl > 0.0 && ndotlLight > 0.02 && dist2 > 1e-4) {
            vec3 sro = h.p + h.n * EPS;
            Hit sh;
            bool blocked = false;
            if (sceneIntersect(sro, l, sh)) {
                if (sh.t < dist - EPS) blocked = true;
            }
            if (!blocked) {
                float pdfArea = 1.0 / LIGHT_AREA;
                float pdfSolidAngle = pdfArea * dist2 / ndotlLight;
                vec3 brdf = h.albedo / PI;
                vec3 contrib = throughput * brdf * LIGHT_EMISSION * u_light_intensity * ndotl / pdfSolidAngle;
                // Recorte de "fireflies": acota muestras raras de altisima
                // varianza sin sesgar visiblemente el resultado promedio.
                color += min(contrib, vec3(60.0));
            }
        }

        // Ruleta rusa a partir del cuarto rebote
        if (b > 3) {
            float p = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 1.0);
            if (rand() > p) break;
            throughput /= p;
        }

        // Rebote indirecto: muestreo coseno (pdf = cos/pi se cancela con la BRDF difusa)
        throughput *= h.albedo;
        vec3 newDir = cosineSampleHemisphere(h.n);
        ro = h.p + h.n * EPS;
        rd = newDir;
    }
    return color;
}

// -------------------------------------------------------------------------
// Modo 3 — Tiempo real hibrido: iluminacion local + oclusion ambiental
// -------------------------------------------------------------------------
vec3 shadeAO(vec3 ro, vec3 rd) {
    Hit h;
    if (!sceneIntersect(ro, rd, h)) return vec3(0.0);
    if (h.matType == 3) return h.emission;

    if (h.matType == 1) return shadeMirrorGlossy(rd, h);
    if (h.matType == 2) return shadeGlassWhitted(rd, h);

    vec3 direct = directLight(h.p, h.n, h.albedo);
    vec3 spec = phongSpecular(h.p, h.n, -rd) * h.shininess;

    float occ = 0.0;
    int n = max(u_ao_samples, 1);
    for (int i = 0; i < n; i++) {
        vec3 dir = cosineSampleHemisphere(h.n);
        Hit ah;
        vec3 aro = h.p + h.n * EPS;
        if (sceneIntersect(aro, dir, ah)) {
            if (ah.matType != 3 && ah.t < u_ao_radius) occ += 1.0;
        }
    }
    float ao = 1.0 - occ / float(n);
    return direct + spec * ao + u_ambient * h.albedo * ao;
}

// -------------------------------------------------------------------------
// Modo 4 — RADIOSIDAD (Goral et al., 1984): interreflexion difusa
// determinista (sin Monte Carlo, sin ruido); solo modela superficies
// difusas -- los materiales especulares/vidrio se tratan como difusos
// grises, igual que se discute en la debilidad de la seccion 3.2.
// -------------------------------------------------------------------------
vec3 radiosityDirect(vec3 p, vec3 n, vec3 albedo) {
    vec3 direct = vec3(0.0);
    const int LSAMPLES = 4;
    vec2 offs[4] = vec2[4](vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(-0.5, 0.5), vec2(0.5, 0.5));
    for (int i = 0; i < LSAMPLES; i++) {
        vec3 lp = vec3(offs[i].x * LIGHT_HX, LIGHT_Y, offs[i].y * LIGHT_HZ);
        vec3 toL = lp - p;
        float d2 = max(dot(toL, toL), 1e-4);
        float d = sqrt(d2);
        vec3 l = toL / d;
        float ndotl = max(dot(n, l), 0.0);
        float ndotlLight = max(dot(vec3(0.0, -1.0, 0.0), -l), 0.0);
        if (ndotl > 0.0 && ndotlLight > 0.0) {
            vec3 sro = p + n * EPS;
            Hit sh;
            bool blocked = false;
            if (sceneIntersect(sro, l, sh)) {
                if (sh.t < d - EPS) blocked = true;
            }
            if (!blocked) {
                direct += albedo / PI * LIGHT_EMISSION * u_light_intensity *
                          ndotl * ndotlLight * (LIGHT_AREA / float(LSAMPLES)) / d2;
            }
        }
    }
    return min(direct, vec3(60.0));
}

vec3 shadeRadiosity(vec3 ro, vec3 rd) {
    Hit h;
    if (!sceneIntersect(ro, rd, h)) return vec3(0.0);
    if (h.matType == 3) return h.emission;
    vec3 albedo = (h.matType == 0) ? h.albedo : vec3(0.82);
    vec3 p = h.p;
    vec3 n = h.n;

    vec3 direct = radiosityDirect(p, n, albedo);

    // Un rebote indirecto difuso con direcciones fijas (espiral aurea):
    // determinista por construccion, por lo que no produce ruido de cuadro
    // a cuadro, a diferencia del path tracing (modo 2).
    vec3 indirect = vec3(0.0);
    const int NDIRS = 28;
    vec3 axis = (abs(n.x) > 0.1) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 u = normalize(cross(axis, n));
    vec3 v = cross(n, u);
    for (int i = 0; i < NDIRS; i++) {
        float fi = float(i) + 0.5;
        float r2 = fi / float(NDIRS);
        float r1 = fi * 2.39996323; // angulo aureo: distribucion uniforme sin ruido
        float r2s = sqrt(r2);
        vec3 dir = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + n * sqrt(1.0 - r2));
        Hit ih;
        vec3 iro = p + n * EPS;
        if (sceneIntersect(iro, dir, ih) && ih.matType != 3) {
            // Si la direccion fija golpea la luz directamente se ignora aqui:
            // esa contribucion ya la capta radiosityDirect() sin sesgo ni
            // varianza (era la causa de un "banding" visible con pocas
            // direcciones deterministas golpeando la luz por casualidad).
            vec3 ialbedo = (ih.matType == 0) ? ih.albedo : vec3(0.82);
            indirect += albedo * radiosityDirect(ih.p, ih.n, ialbedo);
        }
    }
    indirect /= float(NDIRS);

    return direct + indirect + u_ambient * albedo * 0.2;
}

// -------------------------------------------------------------------------
// Modo 5 — MAPEO DE FOTONES (aproximacion eficiente): vidrio refractivo
// (Fresnel + Snell) y una caustica analitica proyectada sobre el piso bajo
// la esfera, en lugar de un mapa de fotones de dos pasadas con kd-tree.
// -------------------------------------------------------------------------
vec3 causticApprox(vec3 p, vec3 n) {
    if (u_sphere2_material != 2) return vec3(0.0);
    if (n.y < 0.5) return vec3(0.0); // solo sobre superficies horizontales (piso)
    float dist = length(p.xz - SPHERE2_C.xz);
    float ringR = SPHERE2_R * 0.55;
    float width = 0.09;
    float ring = exp(-pow((dist - ringR) / width, 2.0));
    float core = exp(-pow(dist / (SPHERE2_R * 0.32), 2.0)) * 0.5;
    return LIGHT_EMISSION * u_light_intensity * 0.85 * (ring + core) * vec3(1.0, 0.96, 0.85);
}

vec3 shadePhotonMap(vec3 ro, vec3 rd) {
    Hit h;
    if (!sceneIntersect(ro, rd, h)) return vec3(0.0);
    if (h.matType == 3) return h.emission;

    if (h.matType == 1) return shadeMirrorGlossy(rd, h);
    if (h.matType == 2) return shadeGlassWhitted(rd, h);

    vec3 direct = directLight(h.p, h.n, h.albedo);
    vec3 spec = phongSpecular(h.p, h.n, -rd) * h.shininess;
    vec3 caustic = causticApprox(h.p, h.n);
    return direct + spec + u_ambient * h.albedo + caustic;
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    rng_state = uint(px.x) * 1973u + uint(px.y) * 9277u + u_frame_seed * 26699u + 1u;
    rng_state = pcg_hash(rng_state);

    vec2 jitter = vec2(0.0);
    if (u_mode == 2) jitter = rand2() - 0.5;

    vec2 uv = (gl_FragCoord.xy + jitter) / u_resolution;
    float aspect = u_resolution.x / u_resolution.y;
    vec2 p = uv * 2.0 - 1.0;
    // OJO: gl_FragCoord.y crece hacia ARRIBA (fila 0 = abajo). p.y ya queda
    // correctamente orientado (arriba de pantalla -> +cam_up) sin invertir.

    vec3 rd = normalize(u_cam_right * p.x * u_fov_scale * aspect +
                         u_cam_up * p.y * u_fov_scale +
                         u_cam_forward);
    vec3 ro = u_cam_pos;

    vec3 col;
    if (u_mode == 0) col = shadeLocal(ro, rd);
    else if (u_mode == 1) col = shadeRaytrace(ro, rd);
    else if (u_mode == 2) col = shadePathtrace(ro, rd);
    else if (u_mode == 3) col = shadeAO(ro, rd);
    else if (u_mode == 4) col = shadeRadiosity(ro, rd);
    else col = shadePhotonMap(ro, rd);

    // Ultima red de seguridad: un solo NaN/Inf que llegue aqui contaminaria
    // el promedio progresivo para siempre (mix(prev, NaN, blend) = NaN se
    // perpetua en cada cuadro futuro), asi que se descarta antes de mezclar.
    bool invalid = isnan(col.x) || isnan(col.y) || isnan(col.z) ||
                   isinf(col.x) || isinf(col.y) || isinf(col.z);
    if (invalid) col = vec3(0.0);

    vec3 prev = texture(u_prev, v_uv).rgb;
    vec3 outc = mix(prev, col, u_blend);
    fragColor = vec4(outc, 1.0);
}
