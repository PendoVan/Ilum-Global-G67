CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-parameter \
            -Iexternal/glad/include -Iexternal/imgui -Iexternal/imgui/backends -Isrc \
            $(shell pkg-config --cflags glfw3)
LDFLAGS := $(shell pkg-config --libs glfw3) -ldl -lGL

TARGET := cornell_gpu

SRC := src/main.cpp \
       external/glad/src/glad.c \
       external/imgui/imgui.cpp \
       external/imgui/imgui_draw.cpp \
       external/imgui/imgui_tables.cpp \
       external/imgui/imgui_widgets.cpp \
       external/imgui/imgui_demo.cpp \
       external/imgui/backends/imgui_impl_glfw.cpp \
       external/imgui/backends/imgui_impl_opengl3.cpp

OBJ := $(SRC:.cpp=.o)
OBJ := $(OBJ:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)
