# Usage Guide

This guide explains how to use the modernized seismic trace denoising package.

## Setup

### Using uv (recommended)

```bash
# Install uv if you haven't already
pip install uv

# Install the package in development mode
uv pip install -e .

# Or install with dev dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e .
```

## Generating Test Data

Generate a test dataset of random seismic-like events:

```bash
python tests/generate_test_data.py --output tests/test_data.npy --n-samples 100
```

Options:
- `--n-samples`: Number of samples (shots) in the dataset (default: 100)
- `--n-traces`: Number of traces per sample (default: 64)
- `--n-time-samples`: Number of time samples per trace (default: 128)
- `--noise-level`: Level of random noise (default: 0.1)

## Training

Train a model on your data:

```bash
train --data path/to/your/data.npy --output-dir checkpoints
```

Or use the test dataset:

```bash
train --data tests/test_data.npy --output-dir checkpoints --n-epochs 20
```

Key training parameters:
- `--n-epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--hidden-channels`: UNet hidden channels (default: 32)
- `--levels`: UNet levels (default: 2)

## Inference

Run inference on data:

```bash
infer --model checkpoints/denoise_final.net --input path/to/data.npy --output denoised.npy
```

Options:
- `--sample-index`: Process only a specific sample index
- `--add-noise`: Add trace-wise noise before denoising (for testing)

## Module Structure

The code is organized into the following modules:

- `seismic_denoising.models`: UNet architecture
- `seismic_denoising.utils`: Utility functions (seed setting, data loading, noise addition)
- `seismic_denoising.preprocessing`: Preprocessing functions (active pixel corruption)
- `seismic_denoising.training`: Training and evaluation functions
- `seismic_denoising.train`: Training entrypoint
- `seismic_denoising.infer`: Inference entrypoint

