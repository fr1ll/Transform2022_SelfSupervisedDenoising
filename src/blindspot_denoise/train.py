"""Training entrypoint for seismic trace denoising."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler

from pydantic_settings import CliApp

from blindspot_denoise.config import TrainingConfig
from blindspot_denoise.models import UNet
from blindspot_denoise.training_loop import n2v_evaluate, n2v_train
from blindspot_denoise.utils import (
    make_streaming_data_loader,
    set_seed,
    weights_init,
)


def get_device() -> torch.device:
    """Get the appropriate device for training."""
    device: torch.device = torch.device("cpu")
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = torch.device(torch.cuda.current_device())
        print(f'Device: {device} {torch.cuda.get_device_name(device)}')
    else:
        print("No GPU available!")
    return device


def train_model(config: TrainingConfig) -> None:
    """Train the denoising model with the given configuration."""
    # Set seed only when provided
    if config.seed is not None:
        set_seed(config.seed)

    # Load data (memory-mapped to reduce RAM usage)
    print(f"Loading data from {config.data}")
    d = np.load(config.data, mmap_mode='r')
    print(f"Data shape: {d.shape}")

    # Setup device
    device = get_device()

    # Build UNet
    print("Building UNet...")
    network = UNet(
        input_channels=1,
        output_channels=1,
        hidden_channels=config.hidden_channels,
        levels=config.levels,
    ).to(device)
    network = network.apply(weights_init)

    # Setup training
    criterion = nn.L1Loss()
    optim = torch.optim.Adam(
        network.parameters(),
        betas=(0.5, 0.999),
        lr=config.learning_rate
    )

    # Initialize arrays to keep track of metrics
    train_loss_history = np.zeros(config.n_epochs)
    train_accuracy_history = np.zeros(config.n_epochs)
    test_loss_history = np.zeros(config.n_epochs)
    test_accuracy_history = np.zeros(config.n_epochs)

    # Create torch generator with fixed seed for reproducibility (if provided)
    g = torch.Generator()
    if config.seed is not None:
        g.manual_seed(config.seed)

    # Build streaming data loaders (on-the-fly corruption & masking)
    train_loader, test_loader = make_streaming_data_loader(
        d,
        n_training=config.n_training,
        n_test=config.n_test,
        batch_size=config.batch_size,
        active_number=config.active_number,
        noise_level=config.noise_level,
        num_noisy_traces=config.num_noisy_traces,
        noisy_trace_value=config.noisy_trace_value,
        torch_generator=g,
        seed=config.seed,
        patch_time=config.patch_time,
        patch_traces=config.patch_traces,
        num_workers=config.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # Training loop
    print("Starting training...")
    scaler = GradScaler('cuda', enabled=(config.use_amp and device.type == 'cuda'))

    for ep in range(config.n_epochs):
        # Train
        train_loss, train_accuracy = n2v_train(
            network,
            criterion,
            optim,
            train_loader,
            device,
            use_amp=(config.use_amp and device.type == 'cuda'),
            scaler=scaler,
        )
        train_loss_history[ep] = train_loss
        train_accuracy_history[ep] = train_accuracy

        # Evaluate
        test_loss, test_accuracy = n2v_evaluate(
            network,
            criterion,
            test_loader,
            device,
            use_amp=(config.use_amp and device.type == 'cuda'),
        )
        test_loss_history[ep] = test_loss
        test_accuracy_history[ep] = test_accuracy

        # Save checkpoint
        if ep % config.save_every == 0:
            mod_name = f'denoise_ep{ep}.net'
            checkpoint_path = config.output_dir / mod_name
            torch.save(network, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Print progress
        print(
            f'Epoch {ep}, '
            f'Training Loss {train_loss:.4f}, Training Accuracy {train_accuracy:.4f}, '
            f'Test Loss {test_loss:.4f}, Test Accuracy {test_accuracy:.4f}'
        )

    # Save final model and training history
    final_model_path = config.output_dir / 'denoise_final.net'
    torch.save(network, final_model_path)
    print(f"Saved final model to {final_model_path}")

    history_path = config.output_dir / 'training_history.npz'
    np.savez(
        history_path,
        train_loss=train_loss_history,
        train_accuracy=train_accuracy_history,
        test_loss=test_loss_history,
        test_accuracy=test_accuracy_history,
    )
    print(f"Saved training history to {history_path}")


def main(args: list[str] | None = None) -> None:
    """
    Train a seismic trace denoising model using self-supervised learning.
    
    Configuration can be provided via command-line arguments, environment variables
    (with BLINDSPOT_TRAIN_ prefix), or a config file.
    
    Example:
        train --data path/to/data.npy --output-dir checkpoints --n-epochs 50
    """
    # Create config from command line arguments
    config = CliApp.run(TrainingConfig, cli_args=args)
    train_model(config)


if __name__ == "__main__":
    main()

