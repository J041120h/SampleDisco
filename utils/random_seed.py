import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42, use_gpu: bool = False, verbose: bool = True):
    """
    Set random seeds for reproducibility on CPU or GPU.

    Parameters
    ----------
    seed : int, default=42
        The seed to use for all RNGs.
    use_gpu : bool, default=False
        Whether to also set GPU-specific RNG and deterministic flags.
    verbose : bool, default=True
        Whether to print confirmation messages.
    """
    # --- Python, NumPy, and PyTorch (CPU) ---
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- GPU-specific controls ---
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # cuDNN: make deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Ensure reproducible convolution / pooling ops
        torch.use_deterministic_algorithms(True, warn_only=True)

        if verbose:
            print(f"[Seed Control] GPU deterministic mode enabled with seed={seed}")
    else:
        # Ensure CPU-only reproducibility
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        if verbose:
            print(f"[Seed Control] CPU deterministic mode enabled with seed={seed}")

    return seed
