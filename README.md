# Self-Supervised Denoising - Transform 2022

This repository contains the modernized code for the Self-supervised denoising tutorial presented at Transform 2022, specifically Tutorial 3 (Trace-wise Noise Suppression).

## Authors

- Claire Birnie (claire.birnie@kaust.edu.sa)
- Sixiu Liu (sixiu.liu@kaust.edu.sa)

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### Generate Test Data

```bash
python tests/generate_test_data.py --output tests/test_data.npy
```

### Train a Model

```bash
train --data tests/test_data.npy --output-dir checkpoints
```

### Run Inference

```bash
infer --model checkpoints/denoise_final.net --input data.npy --output denoised.npy
```

## Documentation

- **[MODERNIZATION.md](MODERNIZATION.md)** - Complete overview of modernization work
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design
- **[DECISIONS.md](DECISIONS.md)** - Key architectural decisions and rationale
- **[USAGE.md](USAGE.md)** - Usage guide and examples

## Features

- ✅ Modern Python package structure with `pyproject.toml`
- ✅ Pydantic-based configuration with type validation
- ✅ Typer CLI for modern command-line interface
- ✅ Environment variable support
- ✅ Modular, reusable code organization
- ✅ Test dataset generator for synthetic seismic data
- ✅ Compatible with Python 3.8-3.12+
- ✅ GPU support with automatic detection

## Project Structure

```
├── src/seismic_denoising/     # Main package
│   ├── models.py              # UNet architecture
│   ├── utils.py               # Utility functions
│   ├── preprocessing.py       # Data preprocessing
│   ├── training.py            # Training functions
│   ├── config.py              # Pydantic configuration
│   ├── train.py               # Training entrypoint
│   └── infer.py               # Inference entrypoint
├── tests/                     # Test utilities
│   └── generate_test_data.py  # Test dataset generator
├── examples/                  # Example scripts
└── pyproject.toml            # Project configuration
```

## Configuration

Configuration can be provided via:
- **CLI arguments** (highest priority)
- **Environment variables** (with `SEISMIC_TRAIN_` or `SEISMIC_INFER_` prefix)
- **Default values** (lowest priority)

Example with environment variables:
```bash
export SEISMIC_TRAIN_DATA=tests/test_data.npy
export SEISMIC_TRAIN_N_EPOCHS=20
train
```

## Usage Examples

### Training with Custom Parameters

```bash
train \
  --data data.npy \
  --output-dir ./models \
  --n-epochs 100 \
  --learning-rate 5e-5 \
  --batch-size 64 \
  --hidden-channels 64 \
  --levels 3
```

### Inference with Noise Addition

```bash
infer \
  --model checkpoints/denoise_final.net \
  --input clean_data.npy \
  --output denoised.npy \
  --add-noise \
  --num-noisy-traces 5
```

### Using in Python

```python
from seismic_denoising.config import TrainingConfig
from seismic_denoising.train import train_model

config = TrainingConfig(
    data="data.npy",
    n_epochs=20,
    learning_rate=1e-4
)
train_model(config)
```

## Notebook

A companion notebook (`Tutorial 3 - Using Entrypoints.ipynb`) demonstrates:
- Using the entrypoint scripts
- Visualizing training progress
- Running inference
- Comparing results

## Methodology

This implementation uses the Structured Noise2Void (StructN2V) methodology:
- Self-supervised learning (no clean data required)
- Blind-trace corruption for training
- Masked loss computation
- UNet architecture for denoising

## Dependencies

- Python 3.8+
- PyTorch
- NumPy, SciPy
- Pydantic Settings
- Typer
- Matplotlib, tqdm

See `pyproject.toml` for complete dependency list.

## Citation

If you use this code, please cite:

> Birnie, C., M. Ravasi, S. Liu, and T. Alkhalifah, 2021, The potential of self-supervised networks for random noise suppression in seismic data: Artificial Intelligence in Geosciences.

> Liu, S., C. Birnie, and T. Alkhalifah, 2022, Coherent noise suppression via a self-supervised deep learning scheme: 83rd EAGE Conference and Exhibition 2022, European Association of Geoscientists & Engineers, 1–5

## License

See LICENSE file for details.

## Acknowledgments

Original tutorial presented at Transform 2022:
- YouTube: https://www.youtube.com/watch?v=d9yv90-JCZ0
- Original repository maintained the tutorial materials
