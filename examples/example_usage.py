"""Example usage of the seismic denoising package."""

import numpy as np
from pathlib import Path

# Example: Generate test data, train, and run inference
if __name__ == "__main__":
    print("This is an example script showing how to use the package.")
    print("\n1. Generate test data:")
    print("   python tests/generate_test_data.py --output tests/test_data.npy")
    print("\n2. Train a model:")
    print("   train --data tests/test_data.npy --output-dir checkpoints --n-epochs 20")
    print("\n3. Run inference:")
    print("   infer --model checkpoints/denoise_final.net --input tests/test_data.npy --output denoised.npy")
    print("\nSee USAGE.md for more details.")

