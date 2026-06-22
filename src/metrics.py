# -*- coding: utf-8 -*-
"""
Métricas de calidad de imagen para comparar renderers.

Implementa PSNR, MSE y una versión simplificada de SSIM sin dependencias
externas. Las métricas comparan cada render contra una referencia
(típicamente path tracing con alto SPP).
"""

import numpy as np


def compute_mse(img, ref):
    """
    Mean Squared Error entre dos imágenes.

    img, ref : ndarray (H, W, 3) float64 o uint8
    Devuelve un float escalar.
    """
    img = img.astype(np.float64)
    ref = ref.astype(np.float64)
    return np.mean((img - ref) ** 2)


def compute_psnr(img, ref, max_val=255.0):
    """
    Peak Signal-to-Noise Ratio en dB.

    img, ref : ndarray (H, W, 3)
    max_val  : valor máximo del rango (255 para uint8, 1.0 para float [0,1])
    """
    mse = compute_mse(img, ref)
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(max_val ** 2 / mse)


def compute_ssim(img, ref, window_size=7):
    """
    Simplified Structural Similarity Index (SSIM).

    Implementación simplificada usando media y varianza locales con un
    filtro uniforme (en lugar del gaussiano completo de Wang et al. 2004).
    Resultado en [0, 1], donde 1 = imágenes idénticas.

    img, ref : ndarray (H, W, 3) float64 o uint8
    """
    img = img.astype(np.float64)
    ref = ref.astype(np.float64)

    # Constantes de estabilización
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Promedio por canales
    ssim_vals = []
    for c in range(3):
        x = img[:, :, c]
        y = ref[:, :, c]

        # Medias y varianzas locales con filtro uniforme
        mu_x = _uniform_filter(x, window_size)
        mu_y = _uniform_filter(y, window_size)
        sigma_xx = _uniform_filter(x * x, window_size) - mu_x ** 2
        sigma_yy = _uniform_filter(y * y, window_size) - mu_y ** 2
        sigma_xy = _uniform_filter(x * y, window_size) - mu_x * mu_y

        # Fórmula SSIM
        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_xx + sigma_yy + C2)
        ssim_map = num / den
        ssim_vals.append(np.mean(ssim_map))

    return float(np.mean(ssim_vals))


def _uniform_filter(x, size):
    """Filtro uniforme 2D usando ventana deslizante con padding."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = size // 2
    padded = np.pad(x, pad, mode="reflect")
    windows = sliding_window_view(padded, (size, size))
    return np.mean(windows, axis=(-2, -1))


def metrics_table(results_dict, ref_key="pathtracing"):
    """
    Genera una tabla Markdown con métricas de cada renderer vs. referencia.

    results_dict : dict {nombre: ndarray_uint8 (H,W,3)}
    ref_key      : clave del renderer usado como referencia

    Devuelve un string con la tabla Markdown.
    """
    ref = results_dict.get(ref_key)
    if ref is None:
        return "⚠ Referencia no encontrada."

    lines = [
        "| Modelo | PSNR (dB) | MSE | SSIM |",
        "| --- | ---: | ---: | ---: |",
    ]

    for name, img in results_dict.items():
        if name == ref_key:
            lines.append(f"| **{name}** (ref) | ∞ | 0.00 | 1.000 |")
        else:
            psnr = compute_psnr(img, ref)
            mse = compute_mse(img, ref)
            ssim = compute_ssim(img, ref)
            lines.append(f"| {name} | {psnr:.2f} | {mse:.2f} | {ssim:.3f} |")

    return "\n".join(lines)
