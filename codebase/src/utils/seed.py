import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed Python ``random``, NumPy and PyTorch (CPU, CUDA and MPS generators).

    With ``deterministic=True`` PyTorch is asked for deterministic kernels
    (``torch.use_deterministic_algorithms(True, warn_only=True)``): ops without a deterministic
    implementation warn instead of raising. Bit-for-bit reproducibility is only guaranteed for the
    same seed on the same device type; use ``--device cpu`` for runs that must match across machines.

    ``PYTHONHASHSEED`` is deliberately not set here: it only takes effect before the interpreter
    starts. MPS users who hit unsupported ops should export ``PYTORCH_ENABLE_MPS_FALLBACK=1`` in the
    shell before launching Python (see README).

    Args:
        seed (int): The random seed integer value (default: 42).
        deterministic (bool): Request deterministic PyTorch algorithms (default: True).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU, CUDA and MPS generators

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required by deterministic cuBLAS; must be set before the first cuBLAS call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.set_num_threads(4)

    print(f"[EEG-KAN] Global random seed locked to: {seed}")
