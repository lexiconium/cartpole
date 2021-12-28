from typing import List

import torch
from torch import nn

from .core.nn import mlp


class Actor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_dims: List[int],
        action_dim: int,
        activation: nn.Module,
        p_dropout: float,
    ):
        super().__init__()
        self.action_dim = action_dim
        layer_dims = [observation_dim, *hidden_dims, action_dim]
        self.layers = mlp(layer_dims=layer_dims, activation=activation, p_dropout=p_dropout)

    def forward(self, observations: torch.Tensor):
        return self.layers(observations)


class Critic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_dims: List[int],
        activation: nn.Module,
        p_dropout: float,
    ):
        super().__init__()
        layer_dims = [observation_dim, *hidden_dims, 1]
        self.layers = mlp(layer_dims=layer_dims, activation=activation, p_dropout=p_dropout)

    def forward(self, observations: torch.Tensor):
        return self.layers(observations)
