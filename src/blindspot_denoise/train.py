"""Training entrypoint for seismic trace denoising."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pydantic_settings import CliApp

from blindspot_denoise.config import TrainingConfig
from blindspot_denoise.models import UNet
from blindspot_denoise.preprocessing import multi_active_pixels
from blindspot_denoise.training_loop import n2v_evaluate, n2v_train
from blindspot_denoise.utils import (
    add_trace_wise_noise,
    make_data_loader,
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
    # Set seed for reproducibility
    set_seed(config.seed)

    # Load data
    print(f"Loading data from {config.data}")
    d = np.load(config.data)
    print(f"Data shape: {d.shape}")

    # Add trace-wise noise
    print("Adding trace-wise noise...")
    noisy_patches = add_trace_wise_noise(
        d,
        num_noisy_traces=config.num_noisy_traces,
        noisy_trace_value=config.noisy_trace_value,
        num_realisations=config.num_realisations,
    )

    # Randomise patch order
    shuffler = np.random.permutation(len(noisy_patches))
    noisy_patches = noisy_patches[shuffler]

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

    # Create torch generator with fixed seed for reproducibility
    g = torch.Generator()
    g.manual_seed(config.seed)

    # Training loop
    print("Starting training...")
    for ep in range(config.n_epochs):
        
        # Randomly corrupt the noisy patches
        corrupted_patches = np.zeros_like(noisy_patches)
        masks = np.zeros_like(corrupted_patches)
        for pi in range(len(noisy_patches)):
            corrupted_patches[pi], masks[pi] = multi_active_pixels(
                noisy_patches[pi],
                active_number=config.active_number,
                noise_level=config.noise_level,
            )

        # Make data loaders
        train_loader, test_loader = make_data_loader(
            noisy_patches,
            corrupted_patches,
            masks,
            config.n_training,
            config.n_test,
            batch_size=config.batch_size,
            torch_generator=g,
        )

        # Train
        train_loss, train_accuracy = n2v_train(
            network,
            criterion,
            optim,
            train_loader,
            device,
        )
        train_loss_history[ep] = train_loss
        train_accuracy_history[ep] = train_accuracy

        # Evaluate
        test_loss, test_accuracy = n2v_evaluate(
            network,
            criterion,
            test_loader,
            device,
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

