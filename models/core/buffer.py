from collections import deque, namedtuple
from typing import List
import random

import torch


class ReplayBuffer:
    def __init__(self, capacity=int(1e6), field_names=List[str]):
        self.buffer = deque([], maxlen=capacity)
        self.Transition = namedtuple(typename="Transition", field_names=field_names)

    def push(self, *args):
        formatted_inputs = [torch.tensor(arg).view(1, -1) if arg is not None else arg for arg in args]
        self.buffer.append(self.Transition(*formatted_inputs))

    def sample(self, size=64):
        sampled = random.sample(self.buffer, k=size)
        return self.Transition(*zip(*sampled))

    def __len__(self):
        return len(self.buffer)
