# Métricas Comparativas — Renderizado Neuronal

Tabla recopilatoria de métricas reportadas en los papers originales de NeRF y 3D Gaussian Splatting, para referencia en la exposición del Integrante 4.

> **Nota**: Estas métricas NO provienen de nuestras implementaciones (no implementamos NeRF ni Gaussian Splatting en este proyecto). Son valores reportados en la literatura.

## NeRF (Mildenhall et al., 2020)

| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Tiempo de entrenamiento | Tiempo de render |
| --- | ---: | ---: | ---: | --- | --- |
| Synthetic (Blender) | 31.01 | 0.947 | 0.081 | ~1-2 días (V100) | ~30 s/imagen |
| Real (LLFF) | 26.50 | 0.811 | 0.250 | ~1-2 días (V100) | ~30 s/imagen |

**Fuente**: Tabla 1 y Tabla 2 de Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV 2020*.

## 3D Gaussian Splatting (Kerbl et al., 2023)

| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Tiempo de entrenamiento | FPS (render) |
| --- | ---: | ---: | ---: | --- | ---: |
| Mip-NeRF360 (outdoor) | 27.21 | 0.815 | 0.214 | ~6-12 min (A6000) | ~134 |
| Mip-NeRF360 (indoor) | 30.41 | 0.920 | 0.189 | ~6-12 min (A6000) | ~160 |
| Tanks & Temples | 23.14 | 0.841 | 0.183 | ~6-12 min (A6000) | ~154 |
| Deep Blending | 29.41 | 0.903 | 0.243 | ~6-12 min (A6000) | ~137 |

**Fuente**: Tabla 1 de Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM Transactions on Graphics (SIGGRAPH 2023)*.

## Comparación clave para la exposición

| Aspecto | NeRF | 3D Gaussian Splatting |
| --- | --- | --- |
| Representación | Red neuronal (MLP) implícita | Conjunto explícito de gaussianas 3D |
| Entrenamiento | Horas-días | Minutos |
| Renderizado | Lento (~30 s/imagen) | Tiempo real (>100 FPS) |
| Calidad (PSNR) | Comparable | Comparable o superior |
| Editabilidad | Difícil | Moderada (puntos explícitos) |
| Relighting | Limitado (codifica iluminación) | Limitado (codifica iluminación) |
| Memoria | Baja (pesos de la red) | Alta (millones de gaussianas) |

## Métricas explicadas

- **PSNR (Peak Signal-to-Noise Ratio)**: Mide la fidelidad píxel a píxel en dB. Valores más altos = más similares a la referencia. Típicamente >30 dB se considera buena calidad.
- **SSIM (Structural Similarity Index)**: Mide similitud perceptual de estructura, luminancia y contraste. Rango [0, 1], 1 = idénticas.
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Métrica perceptual basada en redes neuronales. Valores más bajos = más similar perceptualmente.
- **FPS**: Frames per second durante el renderizado.
