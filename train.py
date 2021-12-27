import argparse
import logging
from itertools import count
from collections import deque

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

import gym

from models.dqn import DQN
from models.core.buffer import ReplayBuffer
from utils.seed import seed
from utils.logger import SimpleLoggerWrapper, ScoreLogger

model_name = "dqn"

logger = SimpleLoggerWrapper.get_logger(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", name=model_name
)
scores = ScoreLogger()


def _train(
    policy: DQN,
    target: DQN,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    sample_size: int,
    gamma: float,
):
    if len(buffer) < 2000:
        return

    for _ in range(10):
        transitions = buffer.sample(size=sample_size)
        observations = torch.cat(transitions.observation).float()
        actions = torch.cat(transitions.action).long()
        rewards = torch.cat(transitions.reward).float()
        next_observations = torch.cat(transitions.next_observation).float()
        done_masks = 1 - torch.cat(transitions.done).float()

        state_action_values = policy(observations).gather(dim=1, index=actions)

        next_state_values = target(next_observations).max(dim=1)[0].unsqueeze(dim=1)
        expected_state_action_values = rewards + gamma * next_state_values * done_masks

        loss = F.smooth_l1_loss(input=state_action_values, target=expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(args: argparse.Namespace):
    env = gym.make(args.env)
    config = {
        "observation_dim": env.observation_space.shape[0],
        "hidden_dims": [128],
        "action_dim": env.action_space.n,
        "activation": nn.GELU,
        "p_dropout": 0,
    }
    policy = DQN(**config)
    target = DQN(**config)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.AdamW(params=policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    buffer = ReplayBuffer(
        capacity=int(5e4), field_names=["observation", "action", "reward", "next_observation", "done"]
    )

    for episode in range(args.num_episodes):
        observation = env.reset()
        epsilon = max(0.01, args.epsilon - 0.01 * (episode / 200))

        # fill replay buffer
        for t in count(1):
            action = policy.get_action(observation=torch.from_numpy(observation).float(), epsilon=epsilon)
            next_observation, reward, done, _ = env.step(action=action)
            buffer.push(observation, action, reward / 100, next_observation, done)

            if done:
                scores.append(t)
                break

            observation = next_observation

        # train
        _train(
            policy=policy,
            target=target,
            optimizer=optimizer,
            buffer=buffer,
            sample_size=args.batch_size,
            gamma=args.gamma,
        )

        if (episode + 1) % args.update_interval == 0:
            target.load_state_dict(policy.state_dict())
            mean = np.array(scores[-20:]).mean().item()
            logger.info(msg=f"{episode=}\t{mean=}")

    scores.draw(name=model_name)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--update_interval", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    seed(args.seed)
    train(args)
