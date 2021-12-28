from typing import List

import torch.nn as nn


def mlp(
    layer_dims: List[int],
    activation: nn.Module,
    output_activation: nn.Module = nn.Identity,
    p_dropout: float = 0.1,
) -> nn.Sequential:
    layers = []
    for i in range(len(layer_dims) - 1):
        layers += [
            nn.Linear(in_features=layer_dims[i], out_features=layer_dims[i + 1]),
            nn.Dropout(p=p_dropout),
            activation() if i < len(layer_dims) - 2 else output_activation(),
        ]
    return nn.Sequential(*layers)
