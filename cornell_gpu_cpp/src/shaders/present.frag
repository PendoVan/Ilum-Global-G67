#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform float u_exposure;
uniform float u_gamma;

vec3 acesFilm(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = max(texture(u_tex, v_uv).rgb, 0.0) * u_exposure;
    vec3 c = acesFilm(hdr);
    c = pow(c, vec3(1.0 / u_gamma));
    fragColor = vec4(c, 1.0);
}
