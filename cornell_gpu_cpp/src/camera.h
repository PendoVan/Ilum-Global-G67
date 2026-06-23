#ifndef _CAMERA_H
#define _CAMERA_H
#include <cmath>

#include "glm/glm.hpp"

constexpr float PI_F = 3.14159265358979323846f;

// Camara orbital con la misma convencion que
// https://github.com/yumcyaWiz/glsl330-cornellbox (posicion, lookat y FOV
// canonicos de la Cornell Box de referencia), extendida con un respaldo de
// teclado (flechas + +/-) por si el arrastre de mouse no se registra bien
// en algunos entornos.
class Camera {
   public:
    glm::vec3 pos;
    glm::vec3 forward;
    glm::vec3 right;
    glm::vec3 up;
    float fov;
    glm::vec3 lookat;
    bool dirty = true;

    Camera()
        : pos(278.0f, 273.0f, -900.0f),
          forward(0.0f, 0.0f, 1.0f),
          right(-1.0f, 0.0f, 0.0f),
          up(0.0f, 1.0f, 0.0f),
          fov(45.0f * PI_F / 180.0f),
          lookat(278.0f, 273.0f, 279.6f) {}

    float fovScale() const { return std::tan(0.5f * fov); }

    void reset() {
        pos = glm::vec3(278.0f, 273.0f, -900.0f);
        forward = glm::vec3(0.0f, 0.0f, 1.0f);
        right = glm::vec3(-1.0f, 0.0f, 0.0f);
        up = glm::vec3(0.0f, 1.0f, 0.0f);
        lookat = glm::vec3(278.0f, 273.0f, 279.6f);
        fov = 45.0f * PI_F / 180.0f;
        dirty = true;
    }

    void move(const glm::vec3& v) {
        glm::vec3 delta = v.x * right + v.y * up + v.z * forward;
        pos += delta;
        lookat += delta;
        dirty = true;
    }

    void orbit(float dTheta, float dPhi) {
        glm::vec3 r = glm::normalize(pos - lookat);
        float phi = std::atan2(r.z, r.x);
        if (phi < 0) phi += 2 * PI_F;
        float theta = std::acos(glm::clamp(r.y, -1.0f, 1.0f));

        phi += dPhi;
        theta = glm::clamp(theta + dTheta, 0.05f, PI_F - 0.05f);

        r = glm::vec3(std::cos(phi) * std::sin(theta), std::cos(theta),
                      std::sin(phi) * std::sin(theta));

        const float dist = glm::distance(lookat, pos);
        pos = lookat + dist * r;
        forward = -r;
        right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
        up = glm::normalize(glm::cross(right, forward));
        dirty = true;
    }

    void zoom(float amount) {
        float dist = glm::distance(lookat, pos);
        dist = glm::clamp(dist - amount, 50.0f, 3000.0f);
        pos = lookat - dist * forward;
        dirty = true;
    }
};

#endif
