import random

import torch
import numpy as np


def seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
