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


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_dims: List[int],
        action_dim: int,
        activation: nn.Module,
        p_dropout: float,
    ):
        super().__init__()
        common_layer_dims = [observation_dim, *hidden_dims]
        actor_layer_dims = [hidden_dims[-1], action_dim]
        critic_layer_dims = [hidden_dims[-1], 1]

        self.common_layers = mlp(
            layer_dims=common_layer_dims,
            activation=activation,
            output_activation=activation,
            p_dropout=p_dropout,
        )
        self.actor_layer = mlp(layer_dims=actor_layer_dims, p_dropout=p_dropout)
        self.critic_layer = mlp(layer_dims=critic_layer_dims, p_dropout=p_dropout)

    def actor(self, observations: torch.Tensor):
        x = self.common_layers(observations)
        return self.actor_layer(x)

    def critic(self, observations: torch.Tensor):
        x = self.common_layers(observations)
        return self.critic_layer(x)
