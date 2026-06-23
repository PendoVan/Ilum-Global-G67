// ==========================================================================
// Caja de Cornell -- demo interactiva en GPU (C++ / OpenGL 3.3 / GLFW / ImGui)
//
// Arquitectura inspirada en https://github.com/yumcyaWiz/glsl330-cornellbox
// (GLAD + GLFW + ImGui, path tracer en un fragment shader sobre un quad de
// pantalla completa, parametros de escena radiometricos canonicos de la
// Cornell Box de referencia). Extiende esa base con 6 modelos de
// iluminacion conmutables (Local, Ray Tracing, Path Tracing, Tiempo
// real+AO, Radiosidad, Mapeo de fotones aproximado) y un selector de
// material (difusa/espejo/vidrio) para una de las esferas.
// ==========================================================================
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "glad/glad.h"
//
#include "GLFW/glfw3.h"
//
#include "glm/glm.hpp"
//
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
//
#include "camera.h"
#include "shader.h"

enum Mode {
    MODE_LOCAL = 0,
    MODE_RAYTRACING = 1,
    MODE_PATHTRACING = 2,
    MODE_REALTIME_AO = 3,
    MODE_RADIOSITY = 4,
    MODE_PHOTONMAP = 5,
};

enum Material { MAT_DIFFUSE = 0, MAT_MIRROR = 1, MAT_GLASS = 2 };

static const char* kModeNames =
    "Local (Phong: difusa + especular)\0"
    "Ray Tracing (Whitted, 1980)\0"
    "Path Tracing (Kajiya, 1986)\0"
    "Tiempo real (Local + AO)\0"
    "Radiosidad (Goral et al., 1984)\0"
    "Mapeo de fotones (Jensen, 1996) -- aprox.\0";

static const char* kMaterialNames = "Difusa\0Espejo\0Vidrio (refractivo)\0";

struct AccumBuffer {
    GLuint tex[2] = {0, 0};
    GLuint fbo[2] = {0, 0};
    int writeIdx = 0;
    int width = 0, height = 0;

    void create(int w, int h) {
        destroy();
        width = w;
        height = h;
        glGenTextures(2, tex);
        glGenFramebuffers(2, fbo);
        for (int i = 0; i < 2; i++) {
            glBindTexture(GL_TEXTURE_2D, tex[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            glBindFramebuffer(GL_FRAMEBUFFER, fbo[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex[i], 0);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        writeIdx = 0;
    }

    void destroy() {
        if (tex[0]) glDeleteTextures(2, tex);
        if (fbo[0]) glDeleteFramebuffers(2, fbo);
        tex[0] = tex[1] = fbo[0] = fbo[1] = 0;
    }

    GLuint readTex() const { return tex[1 - writeIdx]; }
    GLuint writeFbo() const { return fbo[writeIdx]; }
    void swap() { writeIdx = 1 - writeIdx; }
};

struct AppState {
    int mode = MODE_PATHTRACING;
    float ambient = 0.08f;
    float lightIntensity = 1.0f;
    int maxBounces = 10;
    int aoSamples = 20;
    float aoRadius = 150.0f;
    int sphere2Material = MAT_MIRROR;
    float exposure = 1.0f;
    float gamma = 2.2f;
    int resolution = 640;
    unsigned int frameCount = 1;
    bool resetAccum = true;
    bool paused = false;
};

static AppState app;
static Camera camera;
static AccumBuffer accum;
static bool draggingOrbit = false, draggingPan = false, draggingZoom = false;
static double lastMouseX = 0, lastMouseY = 0;

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) {
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
            bool ctrl = mods & GLFW_MOD_CONTROL;
            bool shift = mods & GLFW_MOD_SHIFT;
            draggingZoom = ctrl;
            draggingPan = shift && !ctrl;
            draggingOrbit = !ctrl && !shift;
        } else if (action == GLFW_RELEASE) {
            draggingOrbit = draggingPan = draggingZoom = false;
        }
    }
}

static void cursorPosCallback(GLFWwindow* window, double x, double y) {
    double dx = x - lastMouseX, dy = y - lastMouseY;
    lastMouseX = x;
    lastMouseY = y;
    if (draggingOrbit) {
        camera.orbit(0.005f * static_cast<float>(dy), 0.005f * static_cast<float>(dx));
    } else if (draggingPan) {
        camera.move(glm::vec3(-static_cast<float>(dx), static_cast<float>(dy), 0.0f));
    } else if (draggingZoom) {
        camera.zoom(static_cast<float>(dy));
    }
}

static void scrollCallback(GLFWwindow*, double, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    camera.zoom(static_cast<float>(yoffset) * 25.0f);
}

// Guarda el framebuffer visible como PPM (sin dependencias externas); util
// para capturar imagenes de cada algoritmo para el informe.
static void saveScreenshotPPM(GLFWwindow* window, const std::string& path) {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    std::vector<unsigned char> pixels(static_cast<size_t>(w) * h * 3);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = h - 1; y >= 0; y--) {
        std::fwrite(pixels.data() + static_cast<size_t>(y) * w * 3, 1, static_cast<size_t>(w) * 3, f);
    }
    std::fclose(f);
    std::printf("Captura guardada en %s\n", path.c_str());
}

static bool f2WasDown = false;

static void handleKeyboard(GLFWwindow* window) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;
    const float step = 0.02f;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) camera.orbit(0.0f, -step);
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) camera.orbit(0.0f, step);
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) camera.orbit(step, 0.0f);
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) camera.orbit(-step, 0.0f);
    if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) camera.zoom(5.0f);
    if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) camera.zoom(-5.0f);
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);

    bool f2Down = glfwGetKey(window, GLFW_KEY_F2) == GLFW_PRESS;
    if (f2Down && !f2WasDown) {
        char path[64];
        std::snprintf(path, sizeof(path), "cornell_%ld.ppm", static_cast<long>(std::time(nullptr)));
        saveScreenshotPPM(window, path);
    }
    f2WasDown = f2Down;
}

int main() {
    if (!glfwInit()) {
        std::fprintf(stderr, "fallo glfwInit\n");
        return EXIT_FAILURE;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(
        1280, 860, "Ilum-Global-G67 -- Cornell Box GPU (C++/OpenGL)", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "fallo al crear la ventana\n");
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::fprintf(stderr, "fallo al cargar GLAD\n");
        return EXIT_FAILURE;
    }

    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    Shader traceShader("src/shaders/quad.vert", "src/shaders/cornell.frag");
    Shader presentShader("src/shaders/quad.vert", "src/shaders/present.frag");

    float quadVerts[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    accum.create(app.resolution, app.resolution);

    if (const char* m = std::getenv("CORNELL_TEST_MODE")) app.mode = std::atoi(m);
    if (const char* mt = std::getenv("CORNELL_TEST_MATERIAL")) app.sphere2Material = std::atoi(mt);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        handleKeyboard(window);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // -------------------- render --------------------
        bool needReset = app.resetAccum || camera.dirty;
        if (needReset) {
            app.frameCount = 1;
            camera.dirty = false;
            app.resetAccum = false;
        }
        bool progressive = (app.mode == MODE_PATHTRACING);
        float blend = progressive ? (1.0f / static_cast<float>(app.frameCount)) : 1.0f;

        glViewport(0, 0, accum.width, accum.height);
        glBindFramebuffer(GL_FRAMEBUFFER, accum.writeFbo());
        traceShader.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, accum.readTex());
        traceShader.set("u_prev", 0);
        traceShader.set("u_blend", blend);
        traceShader.set("u_mode", app.mode);
        traceShader.set("u_resolution", glm::vec2(accum.width, accum.height));
        traceShader.set("u_cam_pos", camera.pos);
        traceShader.set("u_cam_right", camera.right);
        traceShader.set("u_cam_up", camera.up);
        traceShader.set("u_cam_forward", camera.forward);
        traceShader.set("u_fov_scale", camera.fovScale());
        traceShader.set("u_ambient", app.ambient);
        traceShader.set("u_light_intensity", app.lightIntensity);
        traceShader.set("u_max_bounces", app.maxBounces);
        traceShader.set("u_ao_samples", app.aoSamples);
        traceShader.set("u_ao_radius", app.aoRadius);
        traceShader.set("u_sphere2_material", app.sphere2Material);
        traceShader.set("u_frame_seed", app.frameCount);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        accum.swap();

        if (!(app.paused && progressive)) app.frameCount++;

        int displayW, displayH;
        glfwGetFramebufferSize(window, &displayW, &displayH);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, displayW, displayH);
        glClear(GL_COLOR_BUFFER_BIT);
        presentShader.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, accum.readTex());
        presentShader.set("u_tex", 0);
        presentShader.set("u_exposure", app.exposure);
        presentShader.set("u_gamma", app.gamma);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // -------------------- panel ImGui --------------------
        ImGui::SetNextWindowPos(ImVec2(15, 15), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(460, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Caja de Cornell -- Modelos de iluminacion (GPU)");

        if (ImGui::Combo("Modelo", &app.mode, kModeNames)) {
            app.resetAccum = true;
            if (app.mode == MODE_PHOTONMAP) app.sphere2Material = MAT_GLASS;
            else if (app.mode == MODE_RAYTRACING || app.mode == MODE_REALTIME_AO)
                app.sphere2Material = MAT_MIRROR;
        }

        ImGui::Separator();
        ImGui::Text("Parametros comunes");
        app.resetAccum |= ImGui::SliderFloat("Ambiente (local)", &app.ambient, 0.0f, 0.4f);
        app.resetAccum |= ImGui::SliderFloat("Intensidad de luz", &app.lightIntensity, 0.1f, 3.0f);
        ImGui::SliderFloat("Exposicion", &app.exposure, 0.1f, 3.0f);
        ImGui::SliderFloat("Gamma", &app.gamma, 1.4f, 2.6f);
        app.resetAccum |= ImGui::Combo("Material esfera 2", &app.sphere2Material, kMaterialNames);

        if (app.mode == MODE_RAYTRACING || app.mode == MODE_PATHTRACING) {
            const char* label = (app.mode == MODE_PATHTRACING) ? "Rebotes maximos"
                                                                 : "Reflejos/refracciones recursivas";
            app.resetAccum |= ImGui::SliderInt(label, &app.maxBounces, 1, 32);
        }
        if (app.mode == MODE_REALTIME_AO) {
            app.resetAccum |= ImGui::SliderInt("Muestras de AO", &app.aoSamples, 1, 48);
            app.resetAccum |= ImGui::SliderFloat("Radio de AO", &app.aoRadius, 10.0f, 300.0f);
        }
        if (app.mode == MODE_PHOTONMAP && app.sphere2Material != MAT_GLASS) {
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                                "Selecciona material 'Vidrio' para ver la caustica.");
        }

        ImGui::Separator();
        static int resIdx = 2;
        const char* resOptions[] = {"320x320", "480x480", "640x640", "800x800", "1024x1024"};
        const int resValues[] = {320, 480, 640, 800, 1024};
        if (ImGui::Combo("Resolucion de render", &resIdx, resOptions, 5)) {
            app.resolution = resValues[resIdx];
            accum.create(app.resolution, app.resolution);
            app.frameCount = 1;
        }

        if (ImGui::Button("Reiniciar acumulacion")) app.resetAccum = true;
        ImGui::SameLine();
        if (ImGui::Button("Restablecer camara")) camera.reset();
        ImGui::SameLine();
        ImGui::Checkbox("Pausar", &app.paused);

        ImGui::Separator();
        ImGui::Text("FPS: %.1f  |  Resolucion: %dx%d", ImGui::GetIO().Framerate, accum.width,
                    accum.height);
        if (app.mode == MODE_PATHTRACING) {
            ImGui::Text("Muestras acumuladas: %u", app.frameCount - 1);
            ImGui::TextWrapped(
                "El ruido (varianza de Monte Carlo) disminuye al acumular mas "
                "muestras; el error decrece con 1/sqrt(N).");
        } else if (app.mode == MODE_RADIOSITY) {
            ImGui::TextWrapped(
                "Interreflexion difusa con direcciones fijas (sin ruido): "
                "compara su estabilidad con el ruido del Path Tracing.");
        } else {
            ImGui::Text("Muestras acumuladas: 1 (sin acumulacion progresiva)");
        }

        ImGui::Separator();
        ImGui::Text("Camara -- pos: (%.0f, %.0f, %.0f)", camera.pos.x, camera.pos.y, camera.pos.z);
        ImGui::TextWrapped(
            "Boton central + arrastrar: orbitar.  +Shift: desplazar.  +Ctrl: "
            "zoom.  Rueda: zoom.  Flechas/+-: respaldo por teclado.\n"
            "Compara Local (sin GI ni reflejos) vs. Ray Tracing (espejo/vidrio, "
            "sombras duras) vs. Path Tracing (color bleeding, sombras suaves, "
            "ruido que converge) vs. Radiosidad (GI difusa sin ruido) vs. "
            "Mapeo de fotones (caustica a traves del vidrio).");
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        // Gancho de prueba automatizada (no interactivo): si se define
        // CORNELL_TEST_FRAMES, guarda una captura a esa cuenta de cuadros y
        // cierra la ventana. No afecta el uso normal de la aplicacion.
        static const char* testFramesEnv = std::getenv("CORNELL_TEST_FRAMES");
        if (testFramesEnv) {
            static unsigned long target = std::strtoul(testFramesEnv, nullptr, 10);
            if (app.frameCount >= target) {
                const char* outPath = std::getenv("CORNELL_TEST_OUT");
                saveScreenshotPPM(window, outPath ? outPath : "cornell_test.ppm");
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
        }
    }

    accum.destroy();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
