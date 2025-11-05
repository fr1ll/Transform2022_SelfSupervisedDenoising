"""Preprocessing functions for blind-trace denoising."""

from __future__ import annotations

import numpy as np


def multi_active_pixels(
    patch: np.ndarray,
    active_number: int,
    noise_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt a patch by replacing `active_number` trace columns with uniform noise.

    Returns the corrupted patch and a mask where 0 marks corrupted columns.
    """
    if patch.ndim != 2:
        raise ValueError("patch must be 2D")
    n_time, n_traces = patch.shape
    if not (0 < active_number <= n_traces):
        raise ValueError("active_number must be in 1..n_traces")

    cols = np.random.choice(n_traces, size=active_number, replace=False)

    cp_patch = patch.copy()
    cp_patch[:, cols] = np.random.uniform(-noise_level, noise_level, size=(n_time, active_number))

    mask = np.ones_like(patch)
    mask[:, cols] = 0

    return cp_patch, mask
