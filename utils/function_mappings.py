import torch.nn as nn

ACTIVATION_FUNCTION_MAPPING = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
