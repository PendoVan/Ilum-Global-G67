# Cornell Box GPU -- C++ / OpenGL 3.3 / GLFW / ImGui

Demo interactiva en tiempo real de la Caja de Cornell, con 6 modelos de
iluminacion conmutables, escrita en **C++ + OpenGL** (sin Python), siguiendo
la arquitectura y los parametros de escena de proyectos de referencia en
GitHub:

- [yumcyaWiz/glsl330-cornellbox](https://github.com/yumcyaWiz/glsl330-cornellbox) --
  arquitectura (GLAD + GLFW + ImGui + path tracer en un fragment shader sobre
  un quad de pantalla completa).
- [LucasReSilva/Cornell-Box](https://github.com/LucasReSilva/Cornell-Box) --
  archivos de escena (.pbrt/.xml/.lxs) con los valores radiometricos
  canonicos de la Cornell Box de referencia (dimensiones, reflectancias,
  emision de la luz) usados aqui para calibrar los materiales y la luz.
- [cvryn7/Cornell-Box-Photon-Mapping](https://github.com/cvryn7/Cornell-Box-Photon-Mapping) --
  referencia para el modo de mapeo de fotones.

## Modelos incluidos

0. **Local** -- Phong (difusa + especular) + ambiente, sin iluminacion indirecta.
1. **Ray Tracing** -- Whitted (1980): reflejos/refracciones recursivas + sombras duras.
2. **Path Tracing** -- Kajiya (1986): Monte Carlo progresivo + NEE + ruleta rusa.
3. **Tiempo real** -- iluminacion local + oclusion ambiental aproximada.
4. **Radiosidad** -- Goral et al. (1984): interreflexion difusa determinista (sin ruido).
5. **Mapeo de fotones** -- Jensen (1996), aproximacion: vidrio refractivo + caustica analitica.

La esfera 2 admite material difuso, espejo o vidrio (selector en el panel).

## Parametros de escena (canonicos)

Caja 556 x 548.8 x 559.2, reflectancias blanco (0.8,0.8,0.8) / rojo
(0.8,0.05,0.05) / verde (0.05,0.8,0.05), luz rectangular con emision
(34,19,10), camara en (278,273,-900) con FOV 45° -- los mismos valores que
usa `glsl330-cornellbox` y que provienen de los datos radiometricos
publicados de la Cornell Box.

## Compilacion

Requiere `g++` (C++17), `pkg-config` y la libreria de desarrollo de GLFW3 y
GLM instaladas en el sistema (en Arch/CachyOS: `pacman -S glfw glm`; en
Debian/Ubuntu: `apt install libglfw3-dev libglm-dev`). GLAD e ImGui ya
estan vendorizados en `external/` (no requieren instalacion).

```bash
make
./cornell_gpu
# o:
make run
```

No se usa CMake porque no es necesario para un solo binario; un Makefile
simple con `pkg-config` es suficiente y evita una dependencia adicional.

## Controles

- **Boton central del mouse + arrastrar**: orbitar la camara.
- **Shift + boton central + arrastrar**: desplazar la camara.
- **Ctrl + boton central + arrastrar**: zoom.
- **Rueda del mouse**: zoom.
- **Flechas del teclado**: orbitar (respaldo si el arrastre no responde bien).
- **+ / -**: zoom (respaldo por teclado).
- **F2**: guarda una captura de la ventana en un archivo `.ppm` (se puede
  abrir con GIMP/IrfanView o convertir con `convert foo.ppm foo.png`).
- **Esc**: cerrar la aplicacion.

## Licencias de terceros

`external/glad` y `external/imgui` se incluyen directamente (no como
submodulos) para simplificar la compilacion sin necesitar `git submodule`;
ambos son de uso libre (GLAD es codigo generado de dominio publico, Dear
ImGui esta bajo licencia MIT, ver `external/imgui/LICENSE.txt` si se desea
consultar el repositorio original en https://github.com/ocornut/imgui).
