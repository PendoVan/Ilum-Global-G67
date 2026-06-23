> **Actualización:** a partir de una revisión conjunta con el equipo de
> proyectos de referencia en GitHub ([yumcyaWiz/glsl330-cornellbox](https://github.com/yumcyaWiz/glsl330-cornellbox),
> [LucasReSilva/Cornell-Box](https://github.com/LucasReSilva/Cornell-Box),
> [cvryn7/Cornell-Box-Photon-Mapping](https://github.com/cvryn7/Cornell-Box-Photon-Mapping)),
> la demo se reescribió también en **C++ + OpenGL** (`cornell_gpu_cpp/`),
> adoptando esa misma arquitectura (GLAD + GLFW + ImGui + path tracer en un
> fragment shader) y, sobre todo, los **parámetros radiométricos canónicos**
> de esos repositorios (dimensiones de la caja, reflectancias, emisión de
> la luz) en vez de valores ajustados a ojo. Es la versión recomendada; ver
> [`cornell_gpu_cpp/README.md`](../cornell_gpu_cpp/README.md). Este
> documento describe la justificación de la versión en Python
> (`realtime_gpu/`), que se mantiene como alternativa.

# Justificación de herramientas — Demo interactiva en GPU (`realtime_gpu/`)

Este documento complementa el **Anexo: Indicaciones para el Programa de Demostración**
del informe (`Informe_Modelos_Globales_Iluminacion.md`, sección A.3) y explica
por qué se eligió cada herramienta para el módulo `realtime_gpu/`, que añade
una visualización **interactiva y acelerada por GPU** de la Caja de Cornell
junto a los renderers offline en CPU que ya existían en `src/renderers/`.

## 1. Por qué un módulo nuevo y no extender los renderers de `src/`

Los renderers existentes (`src/renderers/*.py`) están escritos en **Python +
NumPy puro**, ejecutándose en la CPU. Esto es adecuado para generar imágenes
de referencia de alta calidad (uno de los renders, el de mapeo de fotones,
tarda entre 15 y 30 minutos), pero es **imposible de usar en tiempo real**:
ni siquiera el modelo "Local" alcanza fotogramas por segundo interactivos a
resoluciones razonables, porque NumPy vectoriza por imagen completa pero
sigue corriendo en un único hilo de CPU por término algebraico.

Para cumplir el requisito de "verlo de manera gráfica en tiempo real,
usando la GPU", la única opción real es mover el cómputo por píxel a un
**shader programable**, que la GPU ejecuta en paralelo masivo (miles de
núcleos, uno o más por píxel). Por eso se creó un módulo independiente en
vez de intentar acelerar el código NumPy existente: son dos paradigmas de
ejecución distintos (CPU secuencial/vectorizada vs. GPU masivamente paralela
por shader), y mezclarlos habría complicado ambos sin necesidad.

## 2. Elección de plataforma: Python + ModernGL + GLSL (no un motor de juego, no web)

El informe (Tabla 4, sección A.3) plantea cuatro alternativas. Se descartaron:

- **Unity / Unreal Engine 5**: producen resultados convincentes con poco
  código, pero ocultan el algoritmo detrás de sistemas de iluminación ya
  resueltos por el motor (Lumen, HDRP). El objetivo de la exposición es
  **mostrar cómo funciona cada algoritmo** (sombra dura vs. suave, rebote
  difuso, ruido de Monte Carlo), no solo el resultado visual final.
- **three.js / WebGPU**: viable, pero el resto del proyecto ya está en
  Python y el equipo no necesitaba aprender un segundo lenguaje/ecosistema
  para una sola demo. Mantener todo en Python permite reutilizar el
  conocimiento de la escena (`src/core/scene.py`) y presentar un único
  stack tecnológico coherente con el resto del repositorio.
- **Blender (Cycles/Eevee)**: demuestra el efecto, no el algoritmo, y no es
  "propio" como proyecto de programación (requisito explícito del informe).

Se eligió **Python + ModernGL + GLSL + Dear ImGui**, que:

1. **Sigue usando Python**, coherente con `src/` y `scripts/`.
2. **Ejecuta el algoritmo de iluminación en GPU real**: el shader GLSL no es
   una simulación ni una llamada a una librería de "raytracing automático";
   es el mismo tipo de código (intersección rayo-escena, BRDF, muestreo de
   Monte Carlo, ruleta rusa, NEE) que los renderers de `src/renderers/`,
   pero compilado a shader y ejecutado por la GPU en paralelo por píxel.
3. Da **control total** sobre cada algoritmo, necesario para poder alternar
   entre los seis modelos descritos en este documento (secciones 3.1-3.5 del informe).

### ¿Por qué ModernGL y no PyOpenGL "puro" o pyglet/Panda3D?

`ModernGL` es una capa delgada sobre OpenGL 3.3+ que expone directamente
shaders, buffers, framebuffers y texturas con una API en Python idiomática
(objetos `Program`, `Buffer`, `Framebuffer`, `Texture`), sin la verbosidad
de llamar funciones `gl*` sueltas como en PyOpenGL clásico, y sin el peso de
un motor de juego completo (escenas, físicas, asset pipeline) que no se
necesita para un único *fullscreen shader*. Es la opción estándar en la
comunidad Python para "GPU compute/shaders sin motor de juego".

## 3. Por qué un *path tracer* analítico en un fragment shader (y no rasterización + post-proceso)

La escena de la Caja de Cornell (paredes + 2 esferas + luz rectangular) es
**analítica**: cada superficie tiene una ecuación de intersección exacta
(plano, esfera). Esto permite escribir, dentro de un único fragment shader
que se ejecuta una vez por píxel de pantalla, una función `sceneIntersect()`
que reemplaza por completo la necesidad de una malla poligonal, un *vertex
buffer* de geometría o una API de *ray tracing* por hardware (DXR/RTX): el
"trazado de rayos" ocurre matemáticamente dentro del shader, contra
ecuaciones de planos y esferas, igual que en `src/core/geometry.py` pero en
GLSL. Esta técnica (un único *fullscreen quad* + toda la lógica de
intersección/sombreado en el fragment shader) es la misma que usan demos
clásicas de Shadertoy y es perfectamente capaz de ejecutar en tiempo real
los seis modelos sobre una escena de esta complejidad.

La alternativa —rasterizar triángulos y simular cada algoritmo con varias
pasadas (shadow maps, SSAO de espacio de pantalla, *light probes*, etc.)—
es la técnica real que usan los motores de videojuegos (sección 3.5 y 5.2
del informe) pero exige mucha más infraestructura (geometría poligonal,
*g-buffer*, múltiples *render targets*) para una ganancia nula en este caso,
dado que la escena es fija y analítica.

## 4. Por qué Dear ImGui (vía `imgui_bundle`) para la mini interfaz

Se necesitaba un panel de parámetros mínimo (sliders, combos, checkboxes)
superpuesto directamente sobre el render de la GPU, sin construir una GUI
de escritorio aparte (Qt, Tkinter) que tendría que sincronizar su propio
bucle de eventos con el bucle de render de OpenGL. **Dear ImGui** es el
estándar de facto para paneles de depuración/parámetros sobre aplicaciones
gráficas en tiempo real precisamente porque se dibuja en el mismo contexto
OpenGL, cuadro a cuadro, sin estado retenido.

Se usó el paquete **`imgui_bundle`** (y no el clásico `pyimgui`/`imgui`)
porque, al probarlo en este entorno (Python 3.14), `pyimgui` **no tiene
ruedas (*wheels*) precompiladas** y su compilación desde código fuente falla
por incompatibilidades de la API interna de CPython 3.14 (`_PyLong_AsByteArray`,
`_PyGen_SetStopIterationValue`); es un proyecto con mantenimiento más lento.
`imgui_bundle` es un *fork* mantenido activamente, con *wheels* binarias para
versiones recientes de Python, que además incluye `hello_imgui`: un
*runner* de aplicación que crea la ventana, el contexto OpenGL y el bucle
principal por nosotros (vía GLFW internamente), exponiendo *callbacks*
(`post_init`, `show_gui`) donde se inserta el render de ModernGL. Esto evitó
además un problema real encontrado durante el desarrollo: el backend
GLFW+PyOpenGL "manual" de `imgui_bundle.python_backends` no detectaba el
contexto OpenGL activo en este entorno (Wayland/EGL); usar `hello_imgui.run()`
en su lugar resolvió el problema sin perder la posibilidad de mezclar
dibujado ModernGL (la escena) con widgets ImGui (el panel) en el mismo cuadro.

`PyOpenGL` sigue siendo una dependencia porque `imgui_bundle` lo usa
internamente para subir los *draw calls* de ImGui a la GPU; no se usa
directamente en el código del proyecto.

## 5. Diseño de la acumulación progresiva (Path Tracing)

El modo *Path Tracing* es el único que necesita acumular muchos cuadros
para converger (igual que en `src/renderers/path_tracing.py`, donde la
convergencia se logra aumentando `spp`). En la versión GPU esto se
implementa con dos texturas HDR (`AccumBuffer`, *ping-pong*) y un promedio
incremental `nuevo = mezcla(anterior, muestra, 1/n)` calculado dentro del
mismo shader: cada cuadro suma una muestra de Monte Carlo distinta (con
semilla basada en píxel + número de cuadro) y el contador de muestras se
expone en la interfaz, igual que el informe sugiere para el control
deslizante de *muestras por píxel* (sección A.4, Integrante 3). Los demás
modos no acumulan (se recalculan completos cada cuadro) porque son
deterministas a parámetros de cámara fijos.

## 6. Radiosidad y Mapeo de Fotones: aproximaciones eficientes, no offline

A diferencia de `src/renderers/radiosity.py` y `src/renderers/photom_mapping.py`
(que resuelven el problema "como en el papel": factores de forma entre
parches con refinamiento progresivo, o dos pasadas con un mapa de fotones
y un kd-tree), la versión GPU en tiempo real **no puede** pagar ese costo
cuadro a cuadro y aproxima el efecto distintivo de cada algoritmo:

- **Radiosidad** (`shadeRadiosity`): en vez de resolver el sistema lineal de
  radiosidad, se evalúa en cada píxel un único rebote de interreflexión
  difusa con un número fijo de direcciones generadas con el ángulo dorado
  (secuencia de baja discrepancia, determinista). Al no usar números
  aleatorios, la imagen no tiene ruido de cuadro a cuadro -la firma visual
  de la radiosidad real-, a diferencia del Path Tracing. Los materiales
  especulares/vidrio se fuerzan a un gris difuso, igual que la debilidad
  de la sección 3.2 del informe ("solo modela la reflexión difusa").
- **Mapeo de fotones** (`shadePhotonMap`): en vez de emitir fotones desde
  la luz y estimarlos por densidad con un kd-tree, se renderiza un material
  de vidrio refractivo real (Fresnel + Snell) y se añade una **cáustica
  analítica** -un resplandor proyectado en el piso bajo la esfera, calculado
  con una función de distancia al eje óptico de la esfera- que aproxima
  visualmente el efecto sin la segunda pasada. Esto se documenta
  explícitamente en el código y la interfaz como una aproximación, no como
  mapeo de fotones real, para no inducir a error sobre el algoritmo.

Esta es la misma filosofía que justifica el resto del módulo (sección 3):
priorizar una aproximación *eficiente y honesta* sobre una implementación
"correcta" pero inviable en el presupuesto de cuadro de una demo en vivo.

## 7. Controles de cámara: mouse + teclado

Durante las pruebas se detectó que la inyección sintética de *clics* de
mouse (usada para pruebas automatizadas) no llega al evento de botón a
nivel de GLFW en este entorno (Xwayland sin la extensión XTest), aunque el
movimiento del cursor sí se recibe correctamente; es una limitación del
entorno de prueba, no necesariamente del equipo del usuario final. Aun así,
para no depender de que el arrastre con el botón del mouse funcione en
todos los sistemas (X11/Wayland/portátiles con gestos de panel táctil), la
cámara orbital admite dos rutas de entrada independientes:

1. Arrastre con el botón izquierdo o derecho (`imgui.is_mouse_dragging`,
   más robusto que leer `mouse_down` crudo) + rueda para zoom.
2. Flechas del teclado para orbitar y `+`/`-` para zoom, como respaldo que
   no depende en absoluto del estado de los botones del mouse.

El panel muestra en vivo `yaw`/`pitch`/`distancia` de la cámara para que
sea inmediato confirmar si alguna de las dos rutas esta respondiendo.

## 8. Resumen de la correspondencia con el informe

| Sección del informe | Dónde se cubre en `realtime_gpu/` |
|---|---|
| 2.1 Local vs. global | Modo `LOCAL` vs. `PATHTRACING`, alternable en vivo |
| 3.1 Ray tracing (Whitted) | Modo `RAYTRACING`: reflejos/refracciones recursivas + sombra dura |
| 3.2 Radiosidad (Goral et al.) | Modo `RADIOSITY`: GI difusa determinista, sin ruido, solo difusa |
| 3.3 Path tracing (Kajiya) | Modo `PATHTRACING`: Monte Carlo + NEE + ruleta rusa, progresivo |
| 3.4 Mapeo de fotones (Jensen) | Modo `PHOTONMAP`: vidrio refractivo + cáustica analítica aproximada |
| 3.5 Tiempo real | Modo `REALTIME_AO`: iluminación local + oclusión ambiental, sin acumulación |
| 2.2 BRDF / microfacetas | Especular de Phong en esferas + leve rugosidad en el espejo (`shadeMirrorGlossy`) |
| A.5 Métricas a registrar | Panel muestra FPS, resolución y número de muestras acumuladas en vivo |
