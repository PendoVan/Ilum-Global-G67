#ifndef _SHADER_H
#define _SHADER_H
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "glad/glad.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"

inline std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "no se pudo abrir el shader: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline GLuint compileShader(GLenum type, const std::string& source, const std::string& name) {
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLint logSize = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
        std::vector<GLchar> log(logSize);
        glGetShaderInfoLog(shader, logSize, &logSize, log.data());
        std::cerr << "fallo al compilar " << name << ":\n" << log.data() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return shader;
}

class Shader {
   private:
    GLuint program = 0;

   public:
    Shader() = default;
    Shader(const std::string& vertPath, const std::string& fragPath) {
        GLuint vs = compileShader(GL_VERTEX_SHADER, readFile(vertPath), vertPath);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, readFile(fragPath), fragPath);

        program = glCreateProgram();
        glAttachShader(program, vs);
        glAttachShader(program, fs);
        glLinkProgram(program);

        GLint success = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            GLint logSize = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
            std::vector<GLchar> log(logSize);
            glGetProgramInfoLog(program, logSize, &logSize, log.data());
            std::cerr << "fallo al enlazar shader:\n" << log.data() << std::endl;
            std::exit(EXIT_FAILURE);
        }
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    void use() const { glUseProgram(program); }

    void set(const std::string& name, int v) const {
        glUniform1i(glGetUniformLocation(program, name.c_str()), v);
    }
    void set(const std::string& name, unsigned int v) const {
        glUniform1ui(glGetUniformLocation(program, name.c_str()), v);
    }
    void set(const std::string& name, float v) const {
        glUniform1f(glGetUniformLocation(program, name.c_str()), v);
    }
    void set(const std::string& name, const glm::vec2& v) const {
        glUniform2fv(glGetUniformLocation(program, name.c_str()), 1, glm::value_ptr(v));
    }
    void set(const std::string& name, const glm::vec3& v) const {
        glUniform3fv(glGetUniformLocation(program, name.c_str()), 1, glm::value_ptr(v));
    }
};

#endif
