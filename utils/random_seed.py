import os
import random
import numpy as np

def set_global_seed(seed: int = 42, verbose: bool = True):
    """
    Set random seeds for reproducibility on CPU or GPU.
    Automatically detects available GPU libraries.

    Parameters
    ----------
    seed : int, default=42
        The seed to use for all RNGs.
    verbose : bool, default=True
        Whether to print confirmation messages.
    """
    # --- Python and NumPy ---
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # --- PyTorch ---
    try:
        import torch
        torch.manual_seed(seed)
        
        # Check if CUDA is available for PyTorch
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # cuDNN: make deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Ensure reproducible convolution / pooling ops
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True, warn_only=True)
            
            if verbose:
                print(f"[Seed Control] PyTorch GPU deterministic mode enabled with seed={seed}")
        else:
            # CPU-only PyTorch
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False
            if verbose:
                print(f"[Seed Control] PyTorch CPU mode with seed={seed}")
    except ImportError:
        if verbose:
            print(f"[Seed Control] PyTorch not available, skipping torch seed")
    
    # --- CuPy (for RAPIDS) ---
    try:
        import cupy as cp
        cp.random.seed(seed)
        if verbose:
            print(f"[Seed Control] CuPy seed set to {seed}")
    except ImportError:
        # CuPy not installed, which is fine
        pass
    
    # --- Scanpy ---
    try:
        import scanpy as sc
        sc.settings.seed = seed
        if verbose:
            print(f"[Seed Control] Scanpy seed set to {seed}")
    except ImportError:
        pass
    
    return seed