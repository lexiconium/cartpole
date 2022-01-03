import argparse
import logging
from itertools import count

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np


from models.actor_critic import Actor, Critic
from models.core.buffer import ReplayBuffer
from utils.parallelism import ParallelEnvs
from utils.logger import SimpleLoggerWrapper, ScoreLogger


TRAINING_STEPS = 100000


def compute_bellman(state_values, rewards_record, dones_record, gamma):
    targets = [
        (state_values := rewards + masks * gamma * state_values)
        for rewards, masks in zip(rewards_record[::-1], dones_record[::-1])
    ][::-1]
    return torch.tensor(targets).float()


def train(args: argparse.Namespace):
    envs = ParallelEnvs(id=args.env, num_parallel=args.num_parallel)
    config = {
        "observation_dim": envs.observation_space.shape[0],
        "hidden_dims": [128],
        "action_dim": envs.action_space.n,
        "activation": nn.GELU,
        "p_dropout": 0,
    }
    actor = Actor(**config)
    critic = Critic(**config)
    actor_optimizer = optim.AdamW(
        params=actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    critic_optimizer = optim.AdamW(
        params=critic.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    states = envs.reset()
    step = 0
    while step < TRAINING_STEPS:
        states_record, actions_record, rewards_record, dones_record = [], [], [], []
        for _ in range(args.update_interval):
            probabilities = actor(torch.tensor(states).float())
            actions = torch.multinomial(probabilities, num_samples=1).numpy()
            next_states, rewards, dones, _ = envs.step(actions)

            states_record.append(states)
            actions_record.append(actions)
            rewards_record.append(rewards / 100)
            dones_record.append(1 - dones)

            states = next_states

        step += args.update_interval

        final_state_values = critic(torch.tensor(states).float()).numpy()
        targets = compute_bellman(final_state_values, rewards_record, dones_record, args.gamma).view(-1)
        states_record = torch.tensor(states_record).float().view(-1, config["observation_dim"])
        advantages = targets - critic(states_record)
        # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--num_parallel", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--update_interval", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(args)
