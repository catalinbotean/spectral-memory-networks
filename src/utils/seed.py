import os
import random
import numpy as np
import torch

from .logging import log


def set_seed(seed: int = 42):
    """Set all random seeds for full reproducibility and log the configuration."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    log(f"[Seed] All randomness set to seed = {seed}")
