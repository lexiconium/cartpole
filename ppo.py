import argparse
import logging
import multiprocessing
from itertools import count

from numpy.core.fromnumeric import squeeze

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import gym

from models.actor_critic import ActorCritic
from utils.parallelism import ParallelEnvs
from utils.logger import SimpleLoggerWrapper, ScoreLogger


def compute_advantages(deltas, gamma):
    advantages_t = 0
    advantages = [(advantages_t := deltas_t + gamma * advantages_t) for deltas_t in deltas[::-1]][::-1]
    return torch.from_numpy(np.array(advantages)).float()


def compute_bellman(state_values, reward_records, done_records, gamma):
    targets = [
        (state_values := rewards + masks * gamma * state_values)
        for rewards, masks in zip(reward_records[::-1], done_records[::-1])
    ][::-1]
    return torch.from_numpy(np.array(targets)).float()


def test(id: str, actor_critic: ActorCritic):
    env = gym.make(id)

    durations = []
    for _ in range(10):
        state = env.reset()
        for t in count(1):
            output = actor_critic.actor(torch.tensor(state).float())
            action = output.argmax(dim=-1).item()
            next_state, reward, done, _ = env.step(action)

            if done:
                durations.append(t)
                break
            state = next_state

    env.close()
    print(f"Average Duration: {np.array(durations).mean(axis=-1)}")


def train(args: argparse.Namespace):
    envs = ParallelEnvs(id=args.env, num_parallel=multiprocessing.cpu_count())
    config = {
        "observation_dim": envs.observation_space.shape[0],
        "hidden_dims": [128],
        "action_dim": envs.action_space.n,
        "activation": nn.GELU,
        "p_dropout": 0,
    }
    actor_critic = ActorCritic(**config)
    optimizer = optim.AdamW(
        params=actor_critic.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    states = envs.reset()
    for episode in range(args.num_episodes):
        (
            probability_records,
            state_records,
            next_state_records,
            action_records,
            reward_records,
            done_records,
        ) = ([], [], [], [], [], [])
        for _ in range(args.t_steps):
            probabilities = F.softmax(actor_critic.actor(torch.tensor(states).float()), dim=-1)
            actions = torch.multinomial(probabilities, num_samples=1).view(-1).numpy()
            next_states, rewards, dones = envs.step(actions)

            probability_records.append(probabilities.detach().numpy())
            state_records.append(states)
            next_state_records.append(next_states)
            action_records.append(actions)
            reward_records.append(rewards / 100)
            done_records.append(1 - dones)

            states = next_states

        state_records = torch.from_numpy(np.array(state_records)).float()
        next_state_records = torch.from_numpy(np.array(next_state_records)).float()
        action_records = torch.from_numpy(np.array(action_records)).unsqueeze(dim=-1)
        reward_records = torch.from_numpy(np.array(reward_records)).float()  # 20 X 16
        for epoch in range(args.k_epochs):
            deltas = (
                (
                    reward_records.unsqueeze(dim=-1)
                    + args.gamma * actor_critic.critic(next_state_records)
                    - actor_critic.critic(state_records)
                )
                .squeeze(dim=-1)
                .detach()
                .numpy()
            )
            advantages = compute_advantages(deltas=deltas, gamma=args.gamma)

            probabilities = F.softmax(actor_critic.actor(state_records), dim=-1).gather(
                dim=-1, index=action_records
            )
            old_probabilities = torch.from_numpy(np.array(probability_records)).gather(
                dim=-1, index=action_records
            )
            ratios = (probabilities / old_probabilities).squeeze(dim=-1)

            objective = torch.minimum(
                ratios * advantages, ratios.clip(min=1 - args.epsilon, max=1 + args.epsilon) * advantages
            )

            #TODO
            squared_err = None
            loss = (objective - squared_err).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test(id=args.env, actor_critic=actor_critic)

    envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--k_epochs", type=int, default=3)
    parser.add_argument("--t_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(args)
