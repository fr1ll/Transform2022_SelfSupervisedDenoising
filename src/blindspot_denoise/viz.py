import numpy as np
import matplotlib.pyplot as plt


def imshow(ax, img: np.ndarray, title: str | None = None, cmap: str = "gray") -> None:
    ax.imshow(img, cmap=cmap, aspect="auto")
    if title:
        ax.set_title(title)
    ax.axis("off")


def show_pair(noisy: np.ndarray, denoised: np.ndarray, idx: int = 0) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    imshow(axes[0], noisy[idx], title="Noisy")
    imshow(axes[1], denoised[idx], title="Denoised")
    plt.tight_layout()


def show_samples(noisy: np.ndarray, denoised: np.ndarray, n: int = 4, indices: list[int] | None = None) -> None:
    if noisy.ndim == 2:
        noisy = noisy[None, ...]
    if denoised.ndim == 2:
        denoised = denoised[None, ...]
    N = min(len(noisy), len(denoised))
    if indices is None:
        rng = np.random.default_rng()
        indices = rng.choice(N, size=min(n, N), replace=False).tolist()
    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = np.array([axes])
    for row, i in enumerate(indices):
        imshow(axes[row, 0], noisy[i], title=f"Noisy[{i}]")
        imshow(axes[row, 1], denoised[i], title=f"Denoised[{i}]")
    plt.tight_layout()
