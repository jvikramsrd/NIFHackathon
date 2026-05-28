"""
utils/window.py
────────────────────
Utility for creating a cosine‑spline window used for smooth tile blending.
Both the Stage‑1 segmentation tiler and the Dense‑CRF post‑processor rely on
the same window, so a single source of truth prevents bugs and makes tuning
easier.
"""

import numpy as np


def gaussian_window(window_size: int, sigma: float | None = None) -> np.ndarray:
    window_size = int(window_size)
    if sigma is None:
        sigma = window_size / 6.0
    ax = np.linspace(-(window_size - 1) / 2.0, (window_size - 1) / 2.0, window_size)
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    return gauss_2d.astype(np.float32)


def cosine_window(window_size: int, overlap: int, power: float = 2.0) -> np.ndarray:
    """
    Return a 2‑D cosine spline window for seamless blending of overlapping tiles.

    Args:
        window_size: Size of the (square) tile, e.g. 512 or 2048.
        overlap:     Number of pixels that overlap between neighboring tiles.
        power:      Exponent applied to the 1‑D cosine ramp (default = 2 → cosine‑squared).

    Returns:
        A ``window_size × window_size`` ``np.ndarray`` with values in the range [0, 1],
        where the centre of the tile is close to 1 and the edges taper smoothly to 0.
    """
    # Ensure integer values
    window_size = int(window_size)
    overlap = int(overlap)

    # Edge case: no overlap → return an all‑ones window
    if overlap <= 0:
        return np.ones((window_size, window_size), dtype=np.float32)

    # 1‑D cosine ramp from 0 → 1 → 0 over the overlapping region
    ramp = (np.cos(np.pi * np.arange(overlap) / overlap) + 1) / 2

    # Build the 1‑D weighting vector
    w = np.ones(window_size, dtype=np.float32)
    w[:overlap] = ramp[::-1]  # left border
    w[-overlap:] = ramp  # right border
    w **= power  # control the sharpness of the blend

    # Outer product gives a symmetric 2‑D window
    return np.outer(w, w)
