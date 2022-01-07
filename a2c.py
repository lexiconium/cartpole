import argparse
import logging
import multiprocessing
from itertools import count

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import gym

from models.actor_critic import Actor, Critic
from utils.parallelism import ParallelEnvs
from utils.logger import SimpleLoggerWrapper, ScoreLogger

TRAINING_STEPS = 100000


def compute_bellman(state_values, reward_records, done_records, gamma):
    targets = [
        (state_values := rewards + masks * gamma * state_values)
        for rewards, masks in zip(reward_records[::-1], done_records[::-1])
    ][::-1]
    return torch.from_numpy(np.array(targets)).float()


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
    envs = ParallelEnvs(id=args.env, num_parallel=multiprocessing.cpu_count())
    config = {
        "observation_dim": envs.observation_space.shape[0],
        "hidden_dims": [128],
        "activation": nn.GELU,
        "p_dropout": 0,
    }
    actor = Actor(**config, action_dim=envs.action_space.n)
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
        state_records, action_records, reward_records, done_records = [], [], [], []
        for _ in range(args.update_interval):
            probabilities = F.softmax(actor(torch.tensor(states).float()), dim=-1)
            actions = torch.multinomial(probabilities, num_samples=1).view(-1).numpy()
            next_states, rewards, dones = envs.step(actions)

            state_records.append(states)
            action_records.append(actions)
            reward_records.append(rewards / 100)
            done_records.append(1 - dones)

            states = next_states

        step += args.update_interval

        final_state_values = critic(torch.tensor(states).float()).view(-1).detach().numpy()
        targets = compute_bellman(final_state_values, reward_records, done_records, args.gamma)

        state_records = torch.from_numpy(np.array(state_records)).float()
        state_values = critic(state_records).squeeze(dim=-1)
        
        advantages = targets - state_values

        action_records = torch.from_numpy(np.array(action_records)).unsqueeze(dim=-1)
        negative_log = (
            -torch.log_softmax(actor(state_records), dim=-1)
            .gather(dim=-1, index=action_records)
            .squeeze(dim=-1)
        )
        loss = (negative_log * advantages).mean() + F.smooth_l1_loss(state_values, targets)
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()

        if step % (args.update_interval * 10) == 0:
            test(id=args.env, actor=actor)

    envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="CartPole-v1")
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
