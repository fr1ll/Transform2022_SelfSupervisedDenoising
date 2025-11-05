# Modernization Documentation

## Overview

This document describes the modernization of the Transform 2022 Self-Supervised Denoising tutorial repository, specifically focusing on Tutorial 3 (Trace-wise Noise Suppression). The goal was to convert the notebook-based tutorial into a production-ready Python package with modern tooling and best practices.

## Key Changes

### 1. Project Structure Modernization

**Before:**
- Notebook-based code with utility files in root
- Conda environment setup
- No standardized project structure

**After:**
- Modern Python package structure with `src/` layout
- `pyproject.toml` for dependency and build configuration
- Modular code organization under `src/blindspot_denoise/`
- Entrypoint scripts for training and inference
- Test utilities and dataset generators

**Key Decision:** Used `src/` layout following PEP 420 guidelines for better separation of source code from tests and examples.

### 2. Dependency Management

**Before:**
- `environment_tt2022ssd.yml` (conda)
- Pinned dependencies with specific versions

**After:**
- `pyproject.toml` with unpinned dependencies (compatible with Python 3.8+)
- Support for modern Python versions including 3.12
- Uses `uv` or `pip` for installation

**Key Decision:** Unpinned dependencies to allow compatibility with modern Python versions while maintaining flexibility for users to choose specific versions if needed.

### 3. Code Organization

**Module Structure:**
```
src/blindspot_denoise/
├── __init__.py          # Package initialization
├── models.py            # UNet architecture
├── utils.py             # Utility functions (seed, weights, data loading, noise)
├── preprocessing.py     # Active pixel corruption functions
├── training.py          # Training and evaluation functions
├── config.py            # Pydantic configuration models
├── train.py             # Training entrypoint
└── infer.py             # Inference entrypoint
```

**Key Decisions:**
- Separated concerns: models, utilities, preprocessing, and training logic
- Extracted Notebook 3 code into reusable modules
- Maintained backward compatibility with original functionality

### 4. Configuration Management with Pydantic

**Implementation:**
- Created `TrainingConfig` and `InferenceConfig` classes using `pydantic-settings.BaseSettings`
- Type validation and automatic conversion
- Support for environment variables with prefixes (`BLINDSPOT_TRAIN_`, `BLINDSPOT_INFER_`)
- Field validators for path existence and value constraints

**Benefits:**
- Type safety and validation
- Environment variable support
- Clear documentation through field descriptions
- Automatic path validation and creation

**Example:**
```python
from blindspot_denoise.config import TrainingConfig

config = TrainingConfig(
    data="path/to/data.npy",
    n_epochs=20,
    learning_rate=1e-4
)
```

### 5. CLI with Typer and Pydantic Integration

**Implementation:**
- Converted from `argparse` to `typer` for modern CLI
- Integrated with Pydantic models for configuration
- Automatic help generation and type validation
- Support for both CLI arguments and environment variables

**Key Features:**
- Clean, modern CLI interface
- Automatic help text generation
- Type validation at CLI level
- Integration with Pydantic settings

**Example Usage:**
```bash
# Using CLI arguments
train --data data.npy --n-epochs 20 --learning-rate 1e-4

# Using environment variables
export BLINDSPOT_TRAIN_DATA=data.npy
export BLINDSPOT_TRAIN_N_EPOCHS=20
train
```

### 6. Entrypoint Scripts

**Training Entrypoint (`train`):**
- Full training pipeline with configurable parameters
- Automatic checkpoint saving
- Training history tracking
- Progress reporting

**Inference Entrypoint (`infer`):**
- Model loading and inference
- Support for single samples or batches
- Optional noise addition for testing
- Output saving

**Key Decision:** Separated training and inference into distinct entrypoints for clarity and maintainability.

### 7. Test Dataset Generator

**Implementation:**
- Created `tests/generate_test_data.py` for generating synthetic seismic-like data
- Supports multiple event types (hyperbolic, linear, point)
- Configurable parameters (number of samples, traces, time samples, noise level)
- Reproducible with seed control

**Key Decision:** Provides a way to test the system without requiring real seismic data, making the tutorial more accessible.

### 8. Deprecated Code Fixes

**Changes:**
- Fixed PyTorch deprecation warnings:
  - `nn.init.xavier_normal` → `nn.init.xavier_normal_`
  - `nn.init.constant` → `nn.init.constant_`
- Removed unused imports
- Code cleanup and formatting

## Architecture Decisions

### Why Pydantic Settings?

1. **Type Safety:** Automatic type validation and conversion
2. **Environment Variable Support:** Easy configuration via environment variables
3. **Documentation:** Field descriptions serve as inline documentation
4. **Validation:** Built-in validators for paths, ranges, and custom logic
5. **Modern Standard:** Industry-standard for Python configuration management

### Why Typer?

1. **Modern CLI Framework:** Built on Python type hints
2. **Pydantic Integration:** Seamless integration with Pydantic models
3. **Automatic Help:** Generates help text from type hints and docstrings
4. **Better UX:** Cleaner API than argparse

### Why Unpinned Dependencies?

1. **Modern Python Support:** Works with Python 3.8-3.12+
2. **Flexibility:** Users can choose specific versions if needed
3. **Maintenance:** Easier to keep dependencies up to date
4. **Compatibility:** Allows use of latest features and bug fixes

### Module Naming

**Decision:** Used `blindspot_denoise` as the module name
- Clear and descriptive
- Follows Python naming conventions
- Reflects the application domain

## Migration Guide

### For Users of the Original Notebook

1. **Install the package:**
   ```bash
   pip install -e .
   # or
   uv pip install -e .
   ```

2. **Generate test data:**
   ```bash
   python tests/generate_test_data.py --output tests/test_data.npy
   ```

3. **Train a model:**
   ```bash
   train --data tests/test_data.npy --output-dir checkpoints
   ```

4. **Run inference:**
   ```bash
   infer --model checkpoints/denoise_final.net --input data.npy --output denoised.npy
   ```

### For Developers

1. **Use the modules directly:**
   ```python
   from blindspot_denoise.models import UNet
   from blindspot_denoise.utils import add_trace_wise_noise
   from blindspot_denoise.config import TrainingConfig
   ```

2. **Customize configuration:**
   ```python
   config = TrainingConfig(
       data="path/to/data.npy",
       n_epochs=100,
       learning_rate=5e-5
   )
   ```

3. **Extend functionality:**
   - Add new models in `models.py`
   - Add new utilities in `utils.py`
   - Create new entrypoints following the same pattern

## Testing

### Test Dataset Generation

The test dataset generator creates synthetic seismic-like data with:
- Multiple event types (hyperbolic, linear, point)
- Configurable noise levels
- Reproducible results with seed control

### Usage in Notebooks

A companion notebook (`Tutorial 3 - Using Entrypoints.ipynb`) demonstrates:
- Using the entrypoint scripts
- Visualizing results
- Comparing with original Notebook 3 functionality

## Configuration Options

### Training Configuration

All training parameters can be configured via:
- CLI arguments (highest priority)
- Environment variables (with `BLINDSPOT_TRAIN_` prefix)
- Pydantic model defaults

### Inference Configuration

All inference parameters can be configured via:
- CLI arguments (highest priority)
- Environment variables (with `BLINDSPOT_INFER_` prefix)
- Pydantic model defaults

## Future Improvements

Potential enhancements:
1. **Configuration Files:** Support for YAML/TOML configuration files
2. **Logging:** Structured logging with logging configuration
3. **Metrics:** Integration with MLflow or Weights & Biases
4. **Distributed Training:** Support for multi-GPU training
5. **Model Registry:** Versioned model storage
6. **Data Pipeline:** Standardized data loading and preprocessing
7. **Testing:** Unit tests and integration tests
8. **Documentation:** API documentation with Sphinx

## Compatibility

### Python Versions
- Tested with Python 3.8+
- Compatible with Python 3.12
- Unpinned dependencies allow flexibility

### Original Functionality
- All original Notebook 3 functionality preserved
- Same training algorithm and model architecture
- Compatible output formats

## Contributing

When contributing:
1. Follow the established module structure
2. Use Pydantic models for configuration
3. Add type hints to all functions
4. Update documentation for new features
5. Maintain backward compatibility where possible

## References

- Original Tutorial: Transform 2022 Self-Supervised Denoising
- Pydantic Settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- Typer: https://typer.tiangolo.com/
- PEP 420: https://peps.python.org/pep-0420/

