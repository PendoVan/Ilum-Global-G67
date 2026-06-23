# -*- coding: utf-8 -*-
"""
Demo en tiempo real (GPU) de los modelos de iluminacion de la Caja de Cornell.

Renderiza la misma escena que src/core/scene.py pero directamente en un
fragment shader (GLSL ejecutado en GPU vía ModernGL), permitiendo alternar
en vivo entre iluminacion local (Phong), ray tracing (Whitted), path
tracing progresivo (Monte Carlo), una aproximacion hibrida de tiempo real
(local + oclusion ambiental), radiosidad (interreflexion difusa
determinista) y mapeo de fotones (vidrio refractivo + caustica analitica),
con una mini interfaz (Dear ImGui) para ajustar los parametros de cada
algoritmo. Ver docs/realtime_gpu_justificacion.md para la justificacion de
las herramientas elegidas.
"""

import os
import time

import moderngl
import numpy as np
from imgui_bundle import hello_imgui, imgui

from .camera import OrbitCamera

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")

MODE_LOCAL = 0
MODE_RAYTRACING = 1
MODE_PATHTRACING = 2
MODE_REALTIME_AO = 3
MODE_RADIOSITY = 4
MODE_PHOTONMAP = 5

MODE_NAMES = {
    MODE_LOCAL: "Local (Phong: difusa + especular)",
    MODE_RAYTRACING: "Ray Tracing (Whitted, 1980)",
    MODE_PATHTRACING: "Path Tracing (Kajiya, 1986)",
    MODE_REALTIME_AO: "Tiempo real (Local + AO)",
    MODE_RADIOSITY: "Radiosidad (Goral et al., 1984)",
    MODE_PHOTONMAP: "Mapeo de fotones (Jensen, 1996) -- aprox.",
}

MATERIAL_DIFFUSE, MATERIAL_MIRROR, MATERIAL_GLASS = 0, 1, 2
MATERIAL_NAMES = ["Difusa", "Espejo", "Vidrio (refractivo)"]

# Resoluciones suficientemente grandes para apreciar sombras suaves,
# sangrado de color y caustica; el path tracing progresivo converge en
# poco tiempo incluso a 800-1024px gracias al paralelismo de la GPU.
RESOLUTIONS = [480, 640, 800, 1024, 1280]


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class AccumBuffer:
    """Dos texturas HDR (ping-pong) para la acumulacion progresiva."""

    def __init__(self, ctx, width, height):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.textures = [
            ctx.texture((width, height), 4, dtype="f4"),
            ctx.texture((width, height), 4, dtype="f4"),
        ]
        for tex in self.textures:
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbos = [ctx.framebuffer(color_attachments=[tex]) for tex in self.textures]
        self.write_idx = 0

    @property
    def read_tex(self):
        return self.textures[1 - self.write_idx]

    @property
    def write_fbo(self):
        return self.fbos[self.write_idx]

    @property
    def write_tex(self):
        return self.textures[self.write_idx]

    def swap(self):
        self.write_idx = 1 - self.write_idx

    def release(self):
        for fbo in self.fbos:
            fbo.release()
        for tex in self.textures:
            tex.release()


class CornellApp:
    def __init__(self):
        self.ctx = None
        self.trace_program = None
        self.present_program = None
        self.quad_vbo = None
        self.trace_vao = None
        self.present_vao = None
        self.accum = None

        self.camera = OrbitCamera()

        # --- parametros expuestos en la mini interfaz ---
        self.mode = MODE_PATHTRACING
        self.ambient = 0.045
        self.light_intensity = 1.3
        self.max_bounces = 10
        self.ao_samples = 20
        self.ao_radius = 0.55
        self.sphere2_material = MATERIAL_MIRROR
        self.exposure = 1.4
        self.gamma = 2.2
        self.res_index = 2  # 800x800 por defecto

        self.frame_count = 1
        self.reset_accum = True
        self.paused = False

        self._last_time = time.perf_counter()
        self._fps = 0.0

    # ------------------------------------------------------------------
    def post_init(self):
        self.ctx = moderngl.create_context()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        vert_src = _read(os.path.join(SHADER_DIR, "quad.vert"))
        self.trace_program = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_read(os.path.join(SHADER_DIR, "cornell.frag")),
        )
        self.present_program = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_read(os.path.join(SHADER_DIR, "present.frag")),
        )

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.quad_vbo = self.ctx.buffer(quad.tobytes())
        self.trace_vao = self.ctx.vertex_array(
            self.trace_program, [(self.quad_vbo, "2f", "in_pos")]
        )
        self.present_vao = self.ctx.vertex_array(
            self.present_program, [(self.quad_vbo, "2f", "in_pos")]
        )

        self._create_accum_buffer()

    def _create_accum_buffer(self):
        if self.accum is not None:
            self.accum.release()
        size = RESOLUTIONS[self.res_index]
        self.accum = AccumBuffer(self.ctx, size, size)
        self.frame_count = 1
        self.reset_accum = False

    # ------------------------------------------------------------------
    def _handle_camera_input(self, io):
        hovering_gui = io.want_capture_mouse

        # Arrastre con el mouse (boton izquierdo o derecho); se usa
        # is_mouse_dragging (en vez de leer mouse_down crudo) porque delega
        # en el seguimiento de arrastre de Dear ImGui, mas confiable entre
        # backends/plataformas (X11/Wayland) que comparar estados sueltos.
        if not hovering_gui:
            dragging = imgui.is_mouse_dragging(imgui.MouseButton_.left) or \
                imgui.is_mouse_dragging(imgui.MouseButton_.right)
            if dragging and (io.mouse_delta.x != 0.0 or io.mouse_delta.y != 0.0):
                self.camera.orbit(io.mouse_delta.x * 0.008, -io.mouse_delta.y * 0.008)
            if io.mouse_wheel != 0.0:
                self.camera.zoom(io.mouse_wheel * 0.12)

        # Respaldo por teclado (mas confiable en algunos entornos donde el
        # arrastre con mouse no se registra bien): flechas para orbitar,
        # +/- para zoom.
        if not io.want_capture_keyboard:
            step = 0.02
            if imgui.is_key_down(imgui.Key.left_arrow):
                self.camera.orbit(-step, 0.0)
            if imgui.is_key_down(imgui.Key.right_arrow):
                self.camera.orbit(step, 0.0)
            if imgui.is_key_down(imgui.Key.up_arrow):
                self.camera.orbit(0.0, step)
            if imgui.is_key_down(imgui.Key.down_arrow):
                self.camera.orbit(0.0, -step)
            if imgui.is_key_down(imgui.Key.equal):
                self.camera.zoom(0.02)
            if imgui.is_key_down(imgui.Key.minus):
                self.camera.zoom(-0.02)

    def _render_scene(self):
        size = self.accum.width
        cam_pos, cam_right, cam_up, cam_forward = self.camera.vectors()

        need_reset = self.reset_accum or self.camera.dirty
        if need_reset:
            self.frame_count = 1
            self.camera.dirty = False
            self.reset_accum = False

        progressive = self.mode == MODE_PATHTRACING
        blend = (1.0 / self.frame_count) if progressive else 1.0

        self.ctx.viewport = (0, 0, size, size)
        self.accum.write_fbo.use()
        self.trace_program["u_prev"] = 0
        self.accum.read_tex.use(location=0)
        self.trace_program["u_blend"].value = blend
        self.trace_program["u_mode"].value = self.mode
        self.trace_program["u_resolution"].value = (size, size)
        self.trace_program["u_cam_pos"].value = tuple(cam_pos)
        self.trace_program["u_cam_right"].value = tuple(cam_right)
        self.trace_program["u_cam_up"].value = tuple(cam_up)
        self.trace_program["u_cam_forward"].value = tuple(cam_forward)
        self.trace_program["u_fov_scale"].value = self.camera.fov_scale
        self.trace_program["u_ambient"].value = self.ambient
        self.trace_program["u_light_intensity"].value = self.light_intensity
        self.trace_program["u_max_bounces"].value = int(self.max_bounces)
        self.trace_program["u_ao_samples"].value = int(self.ao_samples)
        self.trace_program["u_ao_radius"].value = self.ao_radius
        self.trace_program["u_sphere2_material"].value = int(self.sphere2_material)
        self.trace_program["u_frame_seed"].value = self.frame_count
        self.trace_vao.render(moderngl.TRIANGLE_STRIP)
        self.accum.swap()

        if not (self.paused and progressive):
            self.frame_count += 1

        io = imgui.get_io()
        scale = io.display_framebuffer_scale
        dw = max(int(io.display_size.x * (scale.x or 1.0)), 1)
        dh = max(int(io.display_size.y * (scale.y or 1.0)), 1)
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, dw, dh)
        self.ctx.clear(0.05, 0.05, 0.06)
        self.present_program["u_tex"] = 0
        self.accum.read_tex.use(location=0)
        self.present_program["u_exposure"].value = self.exposure
        self.present_program["u_gamma"].value = self.gamma
        self.present_vao.render(moderngl.TRIANGLE_STRIP)

    # ------------------------------------------------------------------
    def show_gui(self):
        io = imgui.get_io()
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        self._handle_camera_input(io)
        self._render_scene()
        self._draw_panel()

    def _draw_panel(self):
        imgui.set_next_window_pos(imgui.ImVec2(15, 15), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(400, 0), imgui.Cond_.first_use_ever)
        imgui.begin("Caja de Cornell -- Modelos de iluminacion (GPU)")

        changed_mode, new_mode = imgui.combo(
            "Modelo", self.mode, list(MODE_NAMES.values())
        )
        if changed_mode:
            self.mode = new_mode
            self.reset_accum = True
            # Sugerir el material que mejor evidencia cada algoritmo.
            if self.mode == MODE_PHOTONMAP:
                self.sphere2_material = MATERIAL_GLASS
            elif self.mode in (MODE_RAYTRACING, MODE_REALTIME_AO):
                self.sphere2_material = MATERIAL_MIRROR

        imgui.separator()
        imgui.text("Parametros comunes")
        c, self.ambient = imgui.slider_float("Ambiente (local)", self.ambient, 0.0, 0.35)
        self.reset_accum |= c
        c, self.light_intensity = imgui.slider_float(
            "Intensidad de luz", self.light_intensity, 0.1, 3.0
        )
        self.reset_accum |= c
        c, self.exposure = imgui.slider_float("Exposicion", self.exposure, 0.1, 3.0)
        c, self.gamma = imgui.slider_float("Gamma", self.gamma, 1.4, 2.6)
        if self.exposure > 2.2 or self.light_intensity > 2.2:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.7, 0.25, 1.0),
                "Exposicion/intensidad muy altas pueden saturar la imagen a blanco.",
            )
        c, self.sphere2_material = imgui.combo(
            "Material esfera 2", self.sphere2_material, MATERIAL_NAMES
        )
        self.reset_accum |= c

        if self.mode in (MODE_RAYTRACING, MODE_PATHTRACING):
            label = "Rebotes maximos" if self.mode == MODE_PATHTRACING else "Reflejos/refracciones recursivas"
            c, self.max_bounces = imgui.slider_int(label, self.max_bounces, 1, 32)
            self.reset_accum |= c

        if self.mode == MODE_REALTIME_AO:
            c, self.ao_samples = imgui.slider_int("Muestras de AO", self.ao_samples, 1, 48)
            self.reset_accum |= c
            c, self.ao_radius = imgui.slider_float("Radio de AO", self.ao_radius, 0.05, 1.0)
            self.reset_accum |= c

        if self.mode == MODE_PHOTONMAP and self.sphere2_material != MATERIAL_GLASS:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.75, 0.3, 1.0),
                "Selecciona material 'Vidrio' para ver la caustica.",
            )

        imgui.separator()
        changed_res, self.res_index = imgui.combo(
            "Resolucion de render", self.res_index, [f"{r}x{r}" for r in RESOLUTIONS]
        )
        if changed_res:
            self._create_accum_buffer()

        if imgui.button("Reiniciar acumulacion"):
            self.reset_accum = True
        imgui.same_line()
        if imgui.button("Restablecer camara"):
            self.camera.reset()
        imgui.same_line()
        _, self.paused = imgui.checkbox("Pausar", self.paused)

        imgui.separator()
        imgui.text(f"FPS: {self._fps:5.1f}  |  Resolucion: {self.accum.width}x{self.accum.height}")
        if self.mode == MODE_PATHTRACING:
            imgui.text(f"Muestras acumuladas: {self.frame_count - 1}")
            imgui.text_wrapped(
                "El ruido (varianza de Monte Carlo) disminuye al acumular mas "
                "muestras; el error decrece con 1/sqrt(N)."
            )
        elif self.mode == MODE_RADIOSITY:
            imgui.text_wrapped(
                "Interreflexion difusa con direcciones fijas (sin ruido): "
                "compara su estabilidad con el ruido del Path Tracing."
            )
        else:
            imgui.text("Muestras acumuladas: 1 (sin acumulacion progresiva)")

        imgui.separator()
        imgui.text(
            f"Camara -- yaw: {self.camera.yaw:+.2f}  pitch: {self.camera.pitch:+.2f}  "
            f"dist: {self.camera.distance:.2f}"
        )
        imgui.text_wrapped(
            "Click (izq. o der.) + arrastrar fuera de este panel: orbitar. "
            "Rueda del mouse: zoom. Si el arrastre no responde en tu sistema, "
            "usa las flechas del teclado para orbitar y +/- para zoom (los "
            "numeros de arriba deben cambiar al usarlos).\n"
            "Compara Local (sin GI ni reflejos) vs. Ray Tracing (espejo/vidrio, "
            "sombras duras) vs. Path Tracing (color bleeding, sombras suaves, "
            "ruido que converge) vs. Radiosidad (GI difusa sin ruido) vs. "
            "Mapeo de fotones (caustica a traves del vidrio)."
        )
        imgui.end()

    # ------------------------------------------------------------------
    def run(self):
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = (
            "Ilum-Global-G67 -- Cornell Box GPU Real-Time"
        )
        params.app_window_params.window_geometry.size = (1280, 960)
        params.fps_idling.enable_idling = False
        params.callbacks.post_init = self.post_init
        params.callbacks.show_gui = self.show_gui
        hello_imgui.run(params)


def main():
    CornellApp().run()


if __name__ == "__main__":
    main()
