import torch
import numpy as np
import random

def set_experiment_seeds(my_seed: int):
    """
    Sets seed value for PRNGS for reproducibility

    Args:
        my_seed (int)
    """

    torch.manual_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(my_seed)
        torch.cuda.manual_seed_all(my_seed)