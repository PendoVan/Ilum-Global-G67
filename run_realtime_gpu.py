# -*- coding: utf-8 -*-
"""
Punto de entrada de la demo interactiva en GPU (tiempo real).

Uso:
  python run_realtime_gpu.py

Requiere las dependencias listadas en requirements-realtime.txt
(moderngl, glfw, imgui_bundle, PyOpenGL). Ver README.md y
docs/realtime_gpu_justificacion.md para mas detalle.
"""

from realtime_gpu.app import main

if __name__ == "__main__":
    main()
