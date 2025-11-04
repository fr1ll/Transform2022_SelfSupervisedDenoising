"""Inference entrypoint for seismic trace denoising."""

import numpy as np
import torch
from pathlib import Path
import typer

from seismic_denoising.config import InferenceConfig
from seismic_denoising.models import UNet
from seismic_denoising.utils import add_trace_wise_noise

app = typer.Typer(help="Run inference on seismic data")


def get_device():
    """Get the appropriate device for inference."""
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
        print(f'Using device: {device} {torch.cuda.get_device_name(device)}')
    else:
        print("Using CPU")
    return device


def run_inference(config: InferenceConfig) -> None:
    """Run inference with the given configuration."""
    # Setup device
    device = get_device()

    # Load model
    print(f"Loading model from {config.model}")
    network = torch.load(config.model, map_location=device)
    network.eval()

    # Load data
    print(f"Loading data from {config.input}")
    data = np.load(config.input)
    print(f"Input data shape: {data.shape}")

    # Add noise if requested
    if config.add_noise:
        print("Adding trace-wise noise...")
        noisy_data = add_trace_wise_noise(
            data,
            num_noisy_traces=config.num_noisy_traces,
            noisy_trace_value=config.noisy_trace_value,
            num_realisations=1,
        )
        if len(noisy_data.shape) == 3:
            # If add_trace_wise_noise returns 3D array, take first sample
            noisy_data = noisy_data[0]
    else:
        noisy_data = data

    # Select sample if specified
    if config.sample_index is not None:
        if len(noisy_data.shape) == 3:
            noisy_data = noisy_data[config.sample_index]
        else:
            print(f"Warning: sample_index specified but data is 2D, ignoring")

    # Ensure data is 2D
    if len(noisy_data.shape) == 3:
        print("Warning: Data is 3D, processing first sample only")
        noisy_data = noisy_data[0]

    # Convert to tensor
    print("Running inference...")
    torch_data = torch.from_numpy(
        np.expand_dims(np.expand_dims(noisy_data, axis=0), axis=0)
    ).float()

    # Run inference
    with torch.no_grad():
        prediction = network(torch_data.to(device))
        denoised = prediction.detach().cpu().numpy().squeeze()

    # Save output
    print(f"Saving denoised output to {config.output}")
    np.save(config.output, denoised)
    print(f"Output shape: {denoised.shape}")
    print("Done!")


@app.command()
def main(
    model: Path = typer.Option(..., "--model", "-m", help="Path to trained model checkpoint"),
    input: Path = typer.Option(..., "--input", "-i", help="Path to input data file (numpy .npy file)"),
    output: Path = typer.Option(..., "--output", "-o", help="Path to save denoised output (numpy .npy file)"),
    sample_index: int = typer.Option(None, "--sample-index", help="Index of sample to denoise (if None, denoises all samples)"),
    add_noise: bool = typer.Option(False, "--add-noise", help="Add trace-wise noise to input data before denoising"),
    num_noisy_traces: int = typer.Option(5, "--num-noisy-traces", help="Number of noisy traces to add (if add_noise is True)"),
    noisy_trace_value: float = typer.Option(0.0, "--noisy-trace-value", help="Value for noisy traces (if add_noise is True)"),
) -> None:
    """
    Run inference on seismic data to denoise it.
    
    Configuration can be provided via command-line arguments, environment variables
    (with SEISMIC_INFER_ prefix), or a config file.
    """
    # Create config from arguments
    config = InferenceConfig(
        model=model,
        input=input,
        output=output,
        sample_index=sample_index,
        add_noise=add_noise,
        num_noisy_traces=num_noisy_traces,
        noisy_trace_value=noisy_trace_value,
    )
    
    run_inference(config)


if __name__ == "__main__":
    app()
