import argparse
from itertools import count
import math
from collections import deque

import torch
from torch import nn, optim

import gym

from models.dqn import DQN
from models.core import ReplayBuffer
from utils.seed import seed
from utils.function_mappings import ACTIVATION_FUNCTION_MAPPING


def train(args):
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    field_names = ["obs", "act", "rew", "next_obs"]
    activation = ACTIVATION_FUNCTION_MAPPING[args.activation]

    policy = DQN(obs_dim=obs_dim, hidden_dims=args.hidden_dims, act_dim=act_dim, activation=activation)
    target = DQN(obs_dim=obs_dim, hidden_dims=args.hidden_dims, act_dim=act_dim, activation=activation)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.AdamW(params=policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=int(5e5))
    criterion = nn.SmoothL1Loss()
    buffer = ReplayBuffer(field_names=field_names)

    durations = deque([], maxlen=100)
    mean = 0
    for epi in range(args.num_episodes):
        obs = env.reset()
        for t in count(1):
            act = policy.action(
                obs=torch.tensor(obs, dtype=torch.float), eps=args.epsilon * math.exp(-epi / 3000)
            )
            next_obs, rew, done, _ = env.step(action=act.item())
            buffer.push(obs, act, rew, None if done else next_obs)

            if done:
                durations.append(t)
                mean = torch.tensor(durations).float().mean().item()
                print(f"{epi=}\t{mean=}\t{t=}\t{optimizer.param_groups[0]['lr']}")
                break
            obs = next_obs

        for _ in range(10):
            if len(buffer) < 10 * args.batch_size:
                continue

            transitions = buffer.sample(args.batch_size)
            observations = torch.cat(transitions.obs).float()
            actions = torch.cat(transitions.act).long()
            rewards = torch.cat(transitions.rew).float()

            state_action_values = policy(observations).gather(dim=1, index=actions)

            non_final_mask = torch.tensor(
                list(map(lambda obs: obs is not None, transitions.next_obs)), dtype=torch.bool
            )
            non_final_next_states = torch.cat(
                [obs for obs in transitions.next_obs if obs is not None]
            ).float()
            next_state_values = torch.zeros(args.batch_size)
            next_state_values[non_final_mask] = target(non_final_next_states).max(dim=1)[0]
            expected_state_action_values = rewards + args.gamma * next_state_values.view(args.batch_size, -1)

            loss = criterion(state_action_values, expected_state_action_values)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            scheduler.step()

        if (epi + 1) % args.update_interval == 0:
            target.load_state_dict(policy.state_dict())

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--hidden_dims", type=tuple, default=(128, 64))
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--update_interval", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--epsilon", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    seed(args.seed)
    train(args)

# env.action_space=Discrete(2)
# env.observation_space=Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)
# env.reward_range=(-inf, inf)
# env.metadata={'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
