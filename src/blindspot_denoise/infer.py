"""Inference entrypoint for seismic trace denoising."""

import numpy as np
import torch

from pydantic_settings import CliApp

from blindspot_denoise.config import InferenceConfig
from blindspot_denoise.utils import add_trace_wise_noise

def get_device() -> torch.device:
    """Get the appropriate device for inference."""
    device: torch.device = torch.device("cpu")
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


def main(args: list[str] | None = None) -> None:
    """
    Run inference on seismic data to denoise it.
    
    Configuration can be provided via command-line arguments, environment variables
    (with BLINDSPOT_INFER_ prefix), or a config file.
    """
    # Create config from command line arguments
    config = CliApp.run(InferenceConfig, cli_args=args)
    run_inference(config)


if __name__ == "__main__":
    main()

