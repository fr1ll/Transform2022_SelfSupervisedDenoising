# Architecture Documentation

## System Architecture

### Overview

The seismic trace denoising system implements a self-supervised learning approach for removing trace-wise noise from seismic data. The architecture is modular, extensible, and follows modern Python best practices.

## Component Architecture

### 1. Configuration Layer (`config.py`)

**Purpose:** Centralized configuration management using Pydantic BaseSettings

**Components:**
- `TrainingConfig`: Configuration for training pipeline
- `InferenceConfig`: Configuration for inference pipeline

**Features:**
- Type validation
- Environment variable support
- Path validation
- Default value management

**Design Decision:** Centralized configuration provides:
- Single source of truth for configuration
- Type safety and validation
- Easy testing and mocking
- Environment variable integration

### 2. Model Layer (`models.py`)

**Purpose:** Neural network architecture definitions

**Components:**
- `UNet`: Main denoising network
- `ContractingBlock`: Encoder blocks
- `ExpandingBlock`: Decoder blocks
- `FeatureMapBlock`: Output projection

**Architecture:**
```
Input → UpFeature → Contracting Path → Expanding Path → DownFeature → Output
```

**Design Decision:** Modular UNet design allows:
- Configurable depth (levels)
- Configurable width (hidden_channels)
- Easy experimentation with architecture variants

### 3. Data Processing Layer

#### Utilities (`utils.py`)
**Functions:**
- `set_seed()`: Reproducibility
- `weights_init()`: Model initialization
- `add_trace_wise_noise()`: Data augmentation
- `make_data_loader()`: PyTorch DataLoader creation

#### Preprocessing (`preprocessing.py`)
**Functions:**
- `multi_active_pixels()`: Active pixel corruption for self-supervised learning

**Design Decision:** Separated preprocessing from utilities for:
- Clear separation of concerns
- Easy testing of preprocessing logic
- Potential for different preprocessing strategies

### 4. Training Layer (`training.py`)

**Purpose:** Training and evaluation functions

**Components:**
- `n2v_train()`: Training loop with masked loss
- `n2v_evaluate()`: Validation loop with masked loss

**Key Features:**
- Masked loss computation (only on corrupted pixels)
- Progress tracking with tqdm
- Metrics computation (loss and RMSE)

**Design Decision:** Separated training functions from entrypoint for:
- Reusability in notebooks and scripts
- Easy testing
- Clear separation of training logic

### 5. Entrypoint Layer

#### Training Entrypoint (`train.py`)
**Flow:**
1. Parse configuration (Pydantic + Typer)
2. Load and preprocess data
3. Initialize model
4. Training loop
5. Save checkpoints and history

#### Inference Entrypoint (`infer.py`)
**Flow:**
1. Parse configuration (Pydantic + Typer)
2. Load model
3. Load and preprocess data
4. Run inference
5. Save results

**Design Decision:** Entrypoints use Typer for:
- Modern CLI experience
- Automatic help generation
- Type validation
- Integration with Pydantic

## Data Flow

### Training Data Flow

```
Raw Data (3D numpy array)
    ↓
add_trace_wise_noise() → Noisy Patches
    ↓
Shuffle
    ↓
For each epoch:
    multi_active_pixels() → Corrupted Patches + Masks
    ↓
make_data_loader() → DataLoaders
    ↓
n2v_train() → Update Model Weights
    ↓
n2v_evaluate() → Validation Metrics
    ↓
Save Checkpoints
```

### Inference Data Flow

```
Input Data (numpy array)
    ↓
Optional: add_trace_wise_noise()
    ↓
Convert to Tensor
    ↓
Model Forward Pass
    ↓
Convert to Numpy
    ↓
Save Output
```

## Design Patterns

### 1. Configuration Pattern
- Uses Pydantic BaseSettings for type-safe configuration
- Supports multiple configuration sources (CLI, env vars, defaults)
- Validation at configuration time

### 2. Dependency Injection
- Configuration objects passed to functions
- Easy to test with mock configurations
- Clear dependencies

### 3. Separation of Concerns
- Models: Architecture definition
- Training: Training logic
- Utils: Reusable utilities
- Entrypoints: CLI interface

### 4. Functional Approach
- Most functions are pure (no side effects)
- Easy to test and reason about
- Composable

## Extension Points

### Adding New Models
1. Create new model class in `models.py`
2. Follow UNet interface pattern
3. Update configuration if needed

### Adding New Preprocessing
1. Add function to `preprocessing.py`
2. Update training/inference if needed
3. Document parameters

### Adding New Training Strategies
1. Add function to `training.py`
2. Update training entrypoint
3. Add configuration options if needed

### Adding New Entrypoints
1. Create new file following `train.py` pattern
2. Define Pydantic config model
3. Use Typer for CLI
4. Register in `pyproject.toml`

## Testing Strategy

### Unit Testing
- Test individual functions
- Mock dependencies
- Test edge cases

### Integration Testing
- Test full pipeline
- Use test dataset generator
- Verify output formats

### Configuration Testing
- Test Pydantic validation
- Test environment variable loading
- Test default values

## Performance Considerations

### GPU Support
- Automatic GPU detection
- CUDA device selection
- Batch size optimization

### Memory Management
- Efficient data loading
- Gradient checkpointing (if needed)
- Model checkpointing

### Reproducibility
- Seed control at multiple levels
- Deterministic operations
- Fixed random generators

## Security Considerations

### Input Validation
- Path validation
- File existence checks
- Type validation via Pydantic

### Resource Limits
- Configurable batch sizes
- Memory-aware operations
- Error handling

## Maintenance

### Code Organization
- Clear module boundaries
- Consistent naming
- Documentation

### Versioning
- Semantic versioning
- Changelog maintenance
- Migration guides

### Dependencies
- Unpinned for flexibility
- Regular updates
- Security patches

