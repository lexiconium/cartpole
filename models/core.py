from collections import deque, namedtuple
import random

import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, capacity=int(1e6), field_names=["obs", "act", "rew", "next_obs"]):
        self.buffer = deque([], maxlen=capacity)
        self.Transition = namedtuple(typename="Transition", field_names=field_names)

    def push(self, *args):
        formatted_inputs = [torch.tensor(arg).view(1, -1) if arg is not None else arg for arg in args]
        self.buffer.append(self.Transition(*formatted_inputs))

    def sample(self, batch_size=64):
        sampled = random.sample(self.buffer, k=batch_size)
        return self.Transition(*zip(*sampled))

    def __len__(self):
        return len(self.buffer)


def mlp(layer_dims, activation, output_activation=nn.Identity, p_dropout=0.2):
    layers = []
    for i in range(len(layer_dims) - 1):
        layers += [
            nn.Linear(in_features=layer_dims[i], out_features=layer_dims[i + 1]),
            nn.Dropout(p=p_dropout),
            activation() if i < len(layer_dims) - 2 else output_activation(),
        ]
    return nn.Sequential(*layers)
