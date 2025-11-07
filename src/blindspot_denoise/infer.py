"""Inference entrypoint for seismic trace denoising."""

import numpy as np
import torch

from pydantic_settings import CliApp

from blindspot_denoise.config import InferenceConfig
from blindspot_denoise.models import UNet
from blindspot_denoise.utils import add_trace_wise_noise
from dataclasses import dataclass
from pathlib import Path

def get_device() -> torch.device:
    """Auto-select CUDA if available, else CPU."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        print(f'Using device: {dev} {torch.cuda.get_device_name(dev)}')
    else:
        print("Using CPU")
    return dev


@dataclass
class DenoiseInferencer:
    model: Path
    add_noise: bool = False
    batch_size: int = 4
    num_noisy_traces: int = 5
    noisy_trace_value: float = 0.0
    device: object | None = None

    def __post_init__(self) -> None:
        self.model = Path(self.model)
        dev = self.device
        if dev is None or (isinstance(dev, str) and dev == "auto"):
            self.device = get_device()
        else:
            if isinstance(dev, int):
                self.device = torch.device(f"cuda:{dev}")
            elif isinstance(dev, torch.device):
                self.device = dev
            else:
                self.device = torch.device(dev)
        ckpt = None
        try:
            ckpt = torch.load(self.model, map_location=self.device, weights_only=True, mmap=True)
        except Exception:
            try:
                from torch.serialization import add_safe_globals
                add_safe_globals([UNet])
                ckpt = torch.load(self.model, map_location=self.device, weights_only=True, mmap=True)
            except Exception:
                ckpt = torch.load(self.model, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            arch = ckpt.get('arch', {})
            self._net_levels = int(arch.get('levels', 2))
            self.network = UNet(
                input_channels=arch.get('input_channels', 1),
                output_channels=arch.get('output_channels', 1),
                hidden_channels=arch.get('hidden_channels', 32),
                levels=self._net_levels,
            ).to(self.device)
            self.network.load_state_dict(ckpt['state_dict'])
        elif isinstance(ckpt, torch.nn.Module):
            self.network = ckpt
            self._net_levels = int(getattr(self.network, 'levels', 2))
        else:
            raise RuntimeError("Unsupported checkpoint format. Expected dict with 'state_dict' or a torch.nn.Module.")

        self.network.eval()

    @classmethod
    def from_config(cls, config: InferenceConfig) -> "DenoiseInferencer":
        return cls(
            model=config.model,
            add_noise=config.add_noise,
            batch_size=int(config.batch_size),
            num_noisy_traces=int(config.num_noisy_traces),
            noisy_trace_value=float(config.noisy_trace_value),
        )

    @classmethod
    def from_pretrained(cls, model: str | Path, **kwargs) -> "DenoiseInferencer":
        return cls(model=model, **kwargs)

    def infer_array(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError("Input must be 2D or 3D array")
        if self.add_noise:
            arr = add_trace_wise_noise(
                arr,
                num_noisy_traces=self.num_noisy_traces,
                noisy_trace_value=self.noisy_trace_value,
                num_realisations=1,
            )

        N, H, W = arr.shape
        stride = 2 ** int(self._net_levels)
        pad_h = (-H) % stride
        pad_w = (-W) % stride
        if pad_h or pad_w:
            arr_padded = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
        else:
            arr_padded = arr

        out = np.empty_like(arr)
        bs = int(self.batch_size)
        with torch.no_grad():
            for start in range(0, N, bs):
                end = min(start + bs, N)
                batch_np = np.ascontiguousarray(arr_padded[start:end][:, None, :, :])
                batch = torch.from_numpy(batch_np).float().to(self.device, non_blocking=True)
                pred = self.network(batch).detach().cpu().numpy()[:, 0]
                out[start:end] = pred[:, :H, :W]

        if data.ndim == 2:
            return out[0]
        return out

    def infer_files(self, input_path: Path, output_path: Path) -> None:
        print(f"Loading data from {input_path}")
        data = np.load(input_path, mmap_mode='r')
        print(f"Input data shape: {data.shape}")
        out = self.infer_array(np.asarray(data))
        print(f"Saving denoised output to {output_path}")
        np.save(output_path, out)
        print(f"Output shape: {out.shape}")

    def __call__(
        self,
        inputs,
        *,
        batch_size: int | None = None,
        output: str | Path | None = None,
    ):
        prev_bs = self.batch_size
        if batch_size is not None:
            self.batch_size = int(batch_size)
        try:
            if isinstance(inputs, (str, Path)):
                ip = Path(inputs)
                if output is not None:
                    op = Path(output)
                    self.infer_files(ip, op)
                    return op
                data = np.load(ip, mmap_mode='r')
                return self.infer_array(np.asarray(data))
            if isinstance(inputs, np.ndarray):
                return self.infer_array(inputs)
            if isinstance(inputs, (list, tuple)):
                results = []
                out_dir: Path | None = None
                if output is not None:
                    out_dir = Path(output)
                    out_dir.mkdir(parents=True, exist_ok=True)
                for item in inputs:
                    if isinstance(item, (str, Path)):
                        ip = Path(item)
                        if out_dir is not None:
                            op = out_dir / (ip.stem + "_denoised.npy")
                            self.infer_files(ip, op)
                            results.append(op)
                        else:
                            data = np.load(ip, mmap_mode='r')
                            results.append(self.infer_array(np.asarray(data)))
                    else:
                        results.append(self.infer_array(np.asarray(item)))
                return results
            raise TypeError("Unsupported input type")
        finally:
            self.batch_size = prev_bs


def run_inference(config: InferenceConfig) -> None:
    """Run inference with the given configuration."""
    infer = DenoiseInferencer.from_config(config)
    print("Running batched inference...")
    infer.infer_files(config.input, config.output)
    print("Done!")


def pipeline(
    task: str | None = None,
    *,
    model: str | Path | None = None,
    device: object | None = None,
    **kwargs,
) -> DenoiseInferencer:
    if model is None:
        raise ValueError("model is required")
    return DenoiseInferencer.from_pretrained(model, device=device, **kwargs)


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

