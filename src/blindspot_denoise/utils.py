"""Utility functions for seismic denoising."""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def weights_init(m: nn.Module) -> None:
    """Initialize convolutional layer weights in-place."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def add_trace_wise_noise(
    d: np.ndarray,
    num_noisy_traces: int,
    noisy_trace_value: float,
    num_realisations: int,
) -> np.ndarray:
    """Return noisy copies of input with random trace-wise corruption per shot."""
    if d.ndim != 3:
        raise ValueError("`d` must have shape (shots, time, traces)")

    shots, time_len, n_traces = d.shape
    out_list: list[np.ndarray] = []

    for shot_idx in range(shots):
        clean: np.ndarray = d[shot_idx]
        data = np.repeat(clean[None, ...], num_realisations, axis=0)
        for i in range(num_realisations):
            cols = np.random.randint(0, n_traces, num_noisy_traces)
            data[i, :, cols] = noisy_trace_value
        out_list.append(data)

    noisy = np.concatenate(out_list, axis=0)
    return noisy


def make_data_loader(
    noisy_patches: np.ndarray,
    corrupted_patches: np.ndarray,
    masks: np.ndarray,
    n_training: int,
    n_test: int,
    batch_size: int,
    torch_generator: torch.Generator,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders from numpy arrays."""
    train_X = np.expand_dims(corrupted_patches[:n_training], axis=1)
    train_y = np.expand_dims(noisy_patches[:n_training], axis=1)
    train_m = np.expand_dims(masks[:n_training], axis=1)

    train_dataset = TensorDataset(
        torch.from_numpy(train_X).float(),
        torch.from_numpy(train_y).float(),
        torch.from_numpy(train_m).float(),
    )

    test_X = np.expand_dims(corrupted_patches[n_training:n_training + n_test], axis=1)
    test_y = np.expand_dims(noisy_patches[n_training:n_training + n_test], axis=1)
    test_m = np.expand_dims(masks[n_training:n_training + n_test], axis=1)
    test_dataset = TensorDataset(
        torch.from_numpy(test_X).float(),
        torch.from_numpy(test_y).float(),
        torch.from_numpy(test_m).float(),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch_generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

