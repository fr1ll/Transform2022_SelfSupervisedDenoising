"""Configuration models using Pydantic BaseSettings with CliApp support."""

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    """Configuration for training the seismic denoising model."""
    
    model_config = SettingsConfigDict(
        env_prefix="BLINDSPOT_TRAIN_",
        case_sensitive=False,
        extra="ignore",
        cli_kebab_case=True,
    )
    
    # Data paths
    data: Path = Field(..., description="Path to input data file (numpy .npy file)")
    output_dir: Path = Field(
        default=Path("./checkpoints"),
        description="Directory to save model checkpoints"
    )
    
    # Training parameters
    n_epochs: int = Field(
        default=20,
        ge=1,
        description="Number of training epochs"
    )
    n_training: int = Field(
        default=2048,
        ge=1,
        description="Number of training samples"
    )
    n_test: int = Field(
        default=256,
        ge=1,
        description="Number of validation samples"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for training"
    )
    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        description="Learning rate"
    )
    
    # Model architecture
    hidden_channels: int = Field(
        default=32,
        ge=1,
        description="Number of hidden channels in UNet"
    )
    levels: int = Field(
        default=2,
        ge=1,
        description="Number of levels in UNet"
    )
    
    # Data augmentation
    num_noisy_traces: int = Field(
        default=5,
        ge=1,
        description="Number of noisy traces to add per shot"
    )
    noisy_trace_value: float = Field(
        default=0.0,
        description="Value for noisy traces"
    )
    num_realisations: int = Field(
        default=7,
        ge=1,
        description="Number of noisy realisations per shot"
    )
    
    # Preprocessing
    active_number: int = Field(
        default=15,
        ge=1,
        description="Number of active pixels for corruption"
    )
    noise_level: float = Field(
        default=0.25,
        ge=0.0,
        description="Noise level for corruption"
    )
    
    # Reproducibility
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    save_every: int = Field(
        default=1,
        ge=1,
        description="Save checkpoint every N epochs"
    )
    
    @field_validator("data", mode="before")
    @classmethod
    def validate_data_path(cls, v) -> Path:
        """Validate that data file exists."""
        if v is None:
            raise ValueError("Data file path is required")
        path = Path(v)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return path
    
    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, v) -> Path:
        """Ensure output directory exists."""
        if v is None:
            return Path("./checkpoints")
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path


class InferenceConfig(BaseSettings):
    """Configuration for running inference on seismic data."""
    
    model_config = SettingsConfigDict(
        env_prefix="BLINDSPOT_INFER_",
        case_sensitive=False,
        extra="ignore",
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )
    
    # Required paths
    model: Path = Field(..., description="Path to trained model checkpoint")
    input: Path = Field(..., description="Path to input data file (numpy .npy file)")
    output: Path = Field(..., description="Path to save denoised output (numpy .npy file)")
    
    # Optional parameters
    sample_index: Optional[int] = Field(
        default=None,
        description="Index of sample to denoise (if None, denoises all samples)"
    )
    add_noise: bool = Field(
        default=False,
        description="Add trace-wise noise to input data before denoising"
    )
    num_noisy_traces: int = Field(
        default=5,
        ge=1,
        description="Number of noisy traces to add (if add_noise is True)"
    )
    noisy_trace_value: float = Field(
        default=0.0,
        description="Value for noisy traces (if add_noise is True)"
    )
    
    @field_validator("model", mode="before")
    @classmethod
    def validate_model_path(cls, v) -> Path:
        """Validate that model file exists."""
        path = Path(v)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path
    
    @field_validator("input", mode="before")
    @classmethod
    def validate_input_path(cls, v) -> Path:
        """Validate that input file exists."""
        path = Path(v)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path
    
    @field_validator("output", mode="before")
    @classmethod
    def validate_output_path(cls, v) -> Path:
        """Ensure output directory exists."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class AddNoiseConfig(BaseSettings):
    """Configuration for adding trace-wise noise to seismic data."""

    model_config = SettingsConfigDict(
        env_prefix="BLINDSPOT_NOISE_",
        case_sensitive=False,
        extra="ignore",
        cli_kebab_case=True,
        cli_implicit_flags=True,
    )

    data: Path = Field(..., description="Path to input data file (numpy .npy file)")
    output: Path = Field(..., description="Path to save noisy data (numpy .npy file)")
    num_noisy_traces: int = Field(
        default=5,
        ge=1,
        description="Number of noisy traces to add per shot",
    )
    noisy_trace_value: float = Field(
        default=0.0,
        description="Value for noisy traces",
    )
    num_realisations: int = Field(
        default=1,
        ge=1,
        description="Number of noisy realisations per shot",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )

    @field_validator("data", mode="before")
    @classmethod
    def validate_input_path(cls, v) -> Path:
        """Validate that input data file exists."""
        if v is None:
            raise ValueError("Input data file path is required")
        path = Path(v)
        if not path.exists():
            raise FileNotFoundError(f"Input data file not found: {path}")
        return path

    @field_validator("output", mode="before")
    @classmethod
    def validate_output_path(cls, v) -> Path:
        """Ensure output directory exists."""
        if v is None:
            raise ValueError("Output path is required")
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

