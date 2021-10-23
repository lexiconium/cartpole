import random

import torch
from torch import nn

from .core import mlp


class DQN(nn.Module):
    def __init__(self, obs_dim, hidden_dims, act_dim, activation):
        super().__init__()
        self.act_dim = act_dim
        layer_dims = [obs_dim, *hidden_dims, act_dim]
        self.layers = mlp(layer_dims=layer_dims, activation=activation, p_dropout=0.1)

    def forward(self, obs):
        return self.layers(obs)

    @torch.no_grad()
    def action(self, obs, eps):
        if random.random() < eps:
            return torch.tensor(random.randrange(self.act_dim))
        else:
            return self.forward(obs).argmax()
