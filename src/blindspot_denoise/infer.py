"""Inference entrypoint for seismic trace denoising."""

import numpy as np
import torch

from pydantic_settings import CliApp

from blindspot_denoise.config import InferenceConfig
from blindspot_denoise.models import UNet
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
    device = get_device()

    print(f"Loading model from {config.model}")
    ckpt = None
    try:
        ckpt = torch.load(config.model, map_location=device, weights_only=True, mmap=True)
    except Exception:
        try:
            # Allowlist UNet for safe weights-only unpickling if present in metadata
            from torch.serialization import add_safe_globals
            add_safe_globals([UNet])
            ckpt = torch.load(config.model, map_location=device, weights_only=True, mmap=True)
        except Exception:
            # As a last resort, allow full pickle load (trusted local file)
            ckpt = torch.load(config.model, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        arch = ckpt.get('arch', {})
        net_levels = int(arch.get('levels', 2))
        network = UNet(
            input_channels=arch.get('input_channels', 1),
            output_channels=arch.get('output_channels', 1),
            hidden_channels=arch.get('hidden_channels', 32),
            levels=net_levels,
        ).to(device)
        network.load_state_dict(ckpt['state_dict'])
    elif isinstance(ckpt, torch.nn.Module):
        network = ckpt
        # Best guess if levels not known; used only for padding logic
        net_levels = getattr(network, 'levels', 2)
    else:
        # Unsupported format
        raise RuntimeError("Unsupported checkpoint format. Expected dict with 'state_dict' or a torch.nn.Module.")

    network.eval()

    print(f"Loading data from {config.input}")
    data = np.load(config.input, mmap_mode='r')
    print(f"Input data shape: {data.shape}")

    if config.add_noise:
        print("Adding trace-wise noise...")
        noisy = add_trace_wise_noise(
            np.asarray(data),
            num_noisy_traces=config.num_noisy_traces,
            noisy_trace_value=config.noisy_trace_value,
            num_realisations=1,
        )
    else:
        noisy = np.asarray(data)

    if noisy.ndim == 2:
        noisy = noisy[None, ...]

    if config.sample_index is not None:
        noisy = noisy[config.sample_index:config.sample_index + 1]

    N, H, W = noisy.shape
    # Ensure compatibility with UNet down/upsampling strides (2**levels)
    stride = 2 ** int(net_levels)
    pad_h = (-H) % stride
    pad_w = (-W) % stride
    if pad_h or pad_w:
        noisy_padded = np.pad(noisy, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
    else:
        noisy_padded = noisy
    bs = int(config.batch_size)
    print("Running batched inference...")

    out = np.empty_like(noisy)

    with torch.no_grad():
        for start in range(0, N, bs):
            end = min(start + bs, N)
            # Copy to contiguous array to avoid non-writable numpy warning from memmap
            batch_np = np.ascontiguousarray(noisy_padded[start:end][:, None, :, :])
            batch = torch.from_numpy(batch_np).float().to(device, non_blocking=True)
            pred = network(batch).detach().cpu().numpy()[:, 0]
            # Crop back to original H, W in case we padded
            out[start:end] = pred[:, :H, :W]

    if config.sample_index is not None or data.ndim == 2:
        out_to_save = out[0]
    else:
        out_to_save = out

    print(f"Saving denoised output to {config.output}")
    np.save(config.output, out_to_save)
    print(f"Output shape: {out_to_save.shape}")
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

