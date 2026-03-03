import os
import random
import numpy as np


def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True, warn_only=True)
            try:
                import cupy as cp
                cp.random.seed(seed)
            except ImportError:
                pass
    except ImportError:
        pass
    try:
        import scanpy as sc
        sc.settings.seed = seed
    except ImportError:
        pass
    
    return seed