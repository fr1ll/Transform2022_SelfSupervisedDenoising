"""Add trace-wise noise to seismic data via Python API or CLI."""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic_settings import CliApp

from blindspot_denoise.config import AddNoiseConfig
from blindspot_denoise.utils import add_trace_wise_noise


def add_noise(
    data: np.ndarray,
    *,
    num_noisy_traces: int = 5,
    noisy_trace_value: float = 0.0,
    num_realisations: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return a copy of ``data`` with random trace-wise noise added.

    Parameters
    ----------
    data:
        Array of shape ``(shots, time, traces)`` containing the clean data.
    num_noisy_traces:
        Number of traces to corrupt in each realisation.
    noisy_trace_value:
        Value to assign to corrupted traces.
    num_realisations:
        Number of noisy realisations to generate per shot.
    seed:
        Optional random seed to make noise generation deterministic.
    """
    if data.ndim != 3:
        raise ValueError(
            "Expected `data` to have shape (shots, time, traces); got ``ndim`` != 3"
        )

    original_state = None
    if seed is not None:
        original_state = np.random.get_state()
        np.random.seed(seed)

    try:
        noisy = add_trace_wise_noise(
            data,
            num_noisy_traces=num_noisy_traces,
            noisy_trace_value=noisy_trace_value,
            num_realisations=num_realisations,
        )
    finally:
        if original_state is not None:
            np.random.set_state(original_state)

    return noisy


def _add_noise_from_config(config: AddNoiseConfig, data: np.ndarray) -> np.ndarray:
    return add_noise(
        data,
        num_noisy_traces=config.num_noisy_traces,
        noisy_trace_value=config.noisy_trace_value,
        num_realisations=config.num_realisations,
        seed=config.seed,
    )


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint for adding trace-wise noise to seismic data."""
    config = CliApp.run(AddNoiseConfig, cli_args=args)

    print(f"Loading data from {config.data}")
    data = np.load(config.data)
    print(f"Input data shape: {data.shape}")

    noisy = _add_noise_from_config(config, data)

    print(f"Saving noisy data to {config.output}")
    np.save(config.output, noisy)
    print("Done!")


__all__ = ["add_noise", "main"]
