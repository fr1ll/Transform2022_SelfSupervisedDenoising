"""Training entrypoint for seismic trace denoising."""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import typer

from seismic_denoising.config import TrainingConfig
from seismic_denoising.models import UNet
from seismic_denoising.utils import (
    set_seed,
    weights_init,
    add_trace_wise_noise,
    make_data_loader,
)
from seismic_denoising.preprocessing import multi_active_pixels
from seismic_denoising.training import n2v_train, n2v_evaluate

app = typer.Typer(help="Train seismic trace denoising model")


def get_device():
    """Get the appropriate device for training."""
    device = 'cpu'
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
            optim,
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


@app.command()
def main(
    data: Path = typer.Option(..., "--data", "-d", help="Path to input data file (numpy .npy file)"),
    output_dir: Path = typer.Option("./checkpoints", "--output-dir", "-o", help="Directory to save model checkpoints"),
    n_epochs: int = typer.Option(20, "--n-epochs", "-e", help="Number of training epochs"),
    n_training: int = typer.Option(2048, "--n-training", "-t", help="Number of training samples"),
    n_test: int = typer.Option(256, "--n-test", help="Number of validation samples"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for training"),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", "-lr", help="Learning rate"),
    hidden_channels: int = typer.Option(32, "--hidden-channels", help="Number of hidden channels in UNet"),
    levels: int = typer.Option(2, "--levels", help="Number of levels in UNet"),
    num_noisy_traces: int = typer.Option(5, "--num-noisy-traces", help="Number of noisy traces to add per shot"),
    noisy_trace_value: float = typer.Option(0.0, "--noisy-trace-value", help="Value for noisy traces"),
    num_realisations: int = typer.Option(7, "--num-realisations", help="Number of noisy realisations per shot"),
    active_number: int = typer.Option(15, "--active-number", help="Number of active pixels for corruption"),
    noise_level: float = typer.Option(0.25, "--noise-level", help="Noise level for corruption"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducibility"),
    save_every: int = typer.Option(1, "--save-every", help="Save checkpoint every N epochs"),
) -> None:
    """
    Train a seismic trace denoising model using self-supervised learning.
    
    Configuration can be provided via command-line arguments, environment variables
    (with SEISMIC_TRAIN_ prefix), or a config file.
    """
    # Create config from arguments
    config = TrainingConfig(
        data=data,
        output_dir=output_dir,
        n_epochs=n_epochs,
        n_training=n_training,
        n_test=n_test,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_channels=hidden_channels,
        levels=levels,
        num_noisy_traces=num_noisy_traces,
        noisy_trace_value=noisy_trace_value,
        num_realisations=num_realisations,
        active_number=active_number,
        noise_level=noise_level,
        seed=seed,
        save_every=save_every,
    )
    
    train_model(config)


if __name__ == "__main__":
    app()
