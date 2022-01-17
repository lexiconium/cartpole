import argparse
from collections import namedtuple
from itertools import count
from typing import Iterable

import torch
from torch import nn, optim, distributions
from torch.nn import functional as F

import numpy as np

import gym

from models.actor_critic import Actor, Critic

N = 3  # num epochs
K = 3  # num discriminator update iteration

gamma = 0.98
eps = 0.1
lmbda = 0.99


class Memory:
    def __init__(self, field_names: Iterable[str]):
        self.Fields = namedtuple(typename="Fields", field_names=field_names)
        self.memory = []

    def push(self, *args):
        self.memory.append(self.Fields(*args))

    @staticmethod
    def _format(field):
        if isinstance(field[0], torch.Tensor):
            field = [e.detach().numpy() for e in field]
        field = torch.from_numpy(np.array(field)).float()
        return field.squeeze(dim=-1) if field.shape[-1] == 1 else field

    def rollout(self):
        return self.Fields(*[self._format(field) for field in zip(*self.memory)])

    def reset(self):
        self.memory = []


def test(id: str, actor: Actor):
    env = gym.make(id)

    durations = []
    for _ in range(10):
        state = env.reset()
        for t in count(1):
            output = actor(torch.tensor(state).float())
            action = output.argmax(dim=-1).item()
            next_state, reward, done, _ = env.step(action)

            if done:
                durations.append(t)
                break
            state = next_state

    env.close()
    print(f"Average Duration: {np.array(durations).mean(axis=-1)}")


def train(args: argparse.Namespace):
    env = gym.make(args.env)
    config = {
        "observation_dim": env.observation_space.shape[0],
        "hidden_dims": [128],
        "activation": nn.GELU,
        "p_dropout": 0,
    }
    actor = Actor(**config, action_dim=env.action_space.n)
    discriminator = Actor(**config, action_dim=env.action_space.n)
    critic = Critic(**config)
    actor_optimizer = optim.AdamW(
        params=actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    critic_optimizer = optim.AdamW(
        params=critic.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    discriminator_optimizer = optim.AdamW(
        params=discriminator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    memory = Memory(field_names=["state", "prob", "action", "reward", "next_state", "done"])
    state = env.reset()
    for episode in range(args.num_episodes):
        for _ in range(20):
            """
            generate a batch of M rollouts (M: number of environments)
            """
            action_probs = F.softmax(actor(torch.tensor(state).float()), dim=-1)
            action_distribution = distributions.Categorical(probs=action_probs)
            action = action_distribution.sample()

            next_state, reward, done, _ = env.step(action.item())

            memory.push(state, action_probs, action, reward / 100, next_state, 1 - done)
            state = next_state

            if done:
                break

        states, probs, actions, rewards, next_states, dones = memory.rollout()

        if done:
            memory.reset()
            state = env.reset()

        actions = actions.long().unsqueeze(dim=-1)
        for epoch in range(N):
            for _ in range(K):
                """
                compute and update discriminator
                """
                g_t = discriminator(states).gather(dim=-1, index=actions)

                action_probs = F.softmax(actor(next_states), dim=-1)
                action_distribution = distributions.Categorical(probs=action_probs)
                a_primes = action_distribution.sample().unsqueeze(dim=-1)
                g_t_plus_1 = discriminator(next_states).gather(dim=-1, index=a_primes)

                action_probs = F.softmax(actor(states[0]), dim=-1)
                action_distribution = distributions.Categorical(probs=action_probs)
                a_prime = action_distribution.sample()
                g_1 = discriminator(states[0]).gather(dim=-1, index=a_prime)

                discriminator_loss = (torch.exp(g_t - gamma * g_t_plus_1 - 1) - (1 - gamma) * g_1).mean()

                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()

            # compute value loss
            value_loss = F.mse_loss(critic(states), critic(next_states))

            # compute ppo loss
            deltas = (
                (rewards.unsqueeze(dim=-1) + gamma * critic(next_states) - critic(states))
                .squeeze(dim=-1)
                .detach()
                .numpy()
            )
            advantage = 0
            advantages = torch.from_numpy(
                np.array([(advantage := delta + gamma * lmbda * advantage) for delta in deltas[::-1]][::-1])
            )

            action_probs = F.softmax(actor(states), dim=-1).gather(dim=-1, index=actions)
            old_action_probs = probs.gather(dim=-1, index=actions)
            ratios = (action_probs / old_action_probs).squeeze(dim=-1)

            clipped_loss = torch.minimum(
                advantages * ratios, advantages * torch.clip(ratios, 1 - eps, 1 + eps)
            )
            discriminator_loss = discriminator_loss.detach().clone()
            negative_clipped_loss = -(clipped_loss + lmbda * discriminator_loss)

            # update value and policy functions
            critic_optimizer.zero_grad()
            value_loss.backward()
            critic_optimizer.step()

            actor_optimizer.zero_grad()
            negative_clipped_loss.mean().backward()
            actor_optimizer.step()

        test(id=args.env, actor=actor)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--k_epochs", type=int, default=3)
    parser.add_argument("--t_steps", type=int, default=20)
    parser.add_argument("--update_interval", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(args)
