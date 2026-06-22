# Ilum-Global-G67

Proyecto **Modelos Globales de Iluminación** — Computación Visual / Computación Gráfica  
Facultad de Ingeniería de Sistemas e Informática — Universidad Nacional Mayor de San Marcos

---

## ¿Qué es este proyecto?

Implementación en Python (CPU / NumPy) de los principales algoritmos de iluminación global, todos renderizando la misma **Caja de Cornell** para comparación visual directa.

| Modelo | Algoritmo | Referencia |
|---|---|---|
| **Local** | Phong + término ambiente constante | Baseline (sin luz indirecta) |
| **Ray Tracing** | Trazado de rayos recursivo | Whitted (1980) |
| **Path Tracing** | Monte Carlo + NEE + Ruleta Rusa | Kajiya (1986) |
| **Radiosidad** | Form factors + refinamiento progresivo | Goral et al. (1984) |
| **Mapeo de Fotones** | Dos pasadas con kd-tree | Jensen (1996) |
| **Tiempo Real** | SSAO + Luces Puntuales Virtuales | Keller (1997) |

---

## Requisitos previos

- **Python 3.10 o superior** (probado con 3.12)
- Conexión a Internet para la primera instalación de dependencias

---

## Instalación — Configurar el entorno virtual

Ejecutar **una sola vez** desde la carpeta raíz del proyecto:

```powershell
# 1. Crear el entorno virtual
python -m venv venv

# 2. Activarlo (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** Si PowerShell bloquea la ejecución de scripts, usar:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Dependencias** (`requirements.txt`):
- `numpy` — álgebra vectorial y cómputo matricial
- `Pillow` — lectura/escritura de imágenes PNG
- `scipy` — kd-tree para el mapeo de fotones

---

## Activar el entorno (cada sesión)

```powershell
.\venv\Scripts\Activate.ps1
```

Verás `(venv)` al inicio del prompt. Para salir: `deactivate`

---

## Estructura del proyecto

```
Ilum-Global-G67/
├── venv/                        ← entorno virtual (no se sube al repo)
├── src/
│   ├── core/                    ← utilidades compartidas
│   │   ├── vectors.py           ← álgebra vectorial (normalize, dot, reflect…)
│   │   ├── materials.py         ← tipos de material (DIFFUSE, MIRROR, GLASS…)
│   │   ├── geometry.py          ← primitivas Sphere y Plane
│   │   ├── camera.py            ← cámara pinhole con anti-aliasing
│   │   ├── sampling.py          ← muestreo Monte Carlo hemisférico
│   │   ├── scene.py             ← construcción de la Caja de Cornell
│   │   └── tonemap.py           ← tone mapping Reinhard + guardado PNG
│   ├── renderers/               ← implementaciones de los algoritmos
│   │   ├── local.py             ← iluminación local (baseline)
│   │   ├── ray_tracing.py       ← trazado de rayos (Whitted)
│   │   ├── path_tracing.py      ← path tracing (Monte Carlo)
│   │   ├── radiosity.py         ← radiosidad
│   │   ├── photom_mapping.py    ← mapeo de fotones
│   │   └── realtime.py          ← tiempo real (SSAO + VPLs)
│   ├── config.py                ← parámetros globales (RenderConfig)
│   └── metrics.py               ← PSNR, MSE, SSIM
├── scripts/
│   ├── run_all_models.py        ← ejecuta los 6 renderers y tabla comparativa
│   ├── comparison_grid.py       ← grilla 2×3 con todos los renders
│   ├── convergence.py           ← serie de convergencia del ruido (path tracing)
│   └── contrast_local_global.py ← figura local vs. global lado a lado
├── outputs/                     ← imágenes PNG generadas (se crea automáticamente)
├── docs/
│   └── neuronal/
│       └── tabla_metricas.md    ← métricas NeRF y 3D Gaussian Splatting
├── main.py                      ← punto de entrada con CLI
├── path_tracer_cornell.py       ← path tracer original autónomo (referencia)
└── requirements.txt
```

---

## Ejecución — Orden recomendado

### 1. Prueba rápida (un solo modelo)

```powershell
python main.py --model pathtracing --width 160 --height 160 --spp 16
```

La imagen se guarda en `outputs/cornell_pathtracing.png`.

### 2. Ejecutar todos los modelos

```powershell
python main.py --model all --width 320 --height 320 --spp 64
```

Genera 6 imágenes en `outputs/`.

### 3. Tabla comparativa con métricas (PSNR / SSIM)

```powershell
python scripts/run_all_models.py --width 320 --height 320 --spp 64
```

### 4. Figuras para el informe

```powershell
# Figura 1 — Contraste local vs. global
python scripts/contrast_local_global.py

# Figura 4 — Convergencia del ruido (spp = 1, 4, 16, 64, 256)
python scripts/convergence.py

# Grilla comparativa 2×3 (requiere haber ejecutado run_all_models o main --model all)
python scripts/comparison_grid.py
```

---

## Referencia de comandos

```powershell
# Modelos individuales
python main.py --model local
python main.py --model raytracing
python main.py --model pathtracing
python main.py --model radiosity
python main.py --model photon
python main.py --model realtime

# Opciones disponibles
python main.py --help
#   --model     {local, raytracing, pathtracing, radiosity, photon, realtime, all}
#   --width     ancho en píxeles (default: 320)
#   --height    alto en píxeles (default: 320)
#   --spp       muestras por píxel (default: 64)
#   --output-dir directorio de salida (default: outputs)
#   --seed      semilla aleatoria (default: 2024)
```

---

## Tiempos de ejecución aproximados (320×320, spp=64)

| Modelo | Tiempo | Notas |
|---|---|---|
| Local | < 1 seg | Sin Monte Carlo |
| Ray Tracing | ~5 seg | 4 muestras de sombra por hit |
| Realtime | ~5 seg | SSAO + 64 VPLs |
| Path Tracing | ~1–2 min | 64 muestras por píxel |
| Radiosidad | ~3–5 min | 12×12 parches por cara |
| Mapeo de Fotones | ~15–30 min | 50k fotones globales + 100k cáusticos |

> **Consejo para demos rápidas:** usar `--width 80 --height 80 --spp 4`

---

## Path tracer autónomo (original)

El archivo [`path_tracer_cornell.py`](path_tracer_cornell.py) es el path tracer original del proyecto, independiente del paquete `src/`. Se puede ejecutar directamente:

```powershell
python path_tracer_cornell.py
```

Genera `cornell_global.png` y `cornell_local.png` en la raíz.
