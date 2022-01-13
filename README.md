# Cartpole

Simple Open-AI Gym CartPole experiments.

## Available Algorithms

### Deeq Q-Network
[DQN](https://github.com/lexiconium/cartpole/blob/main/dqn.py): [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
```
python dqn.py
```

### Advantage Actor-Critic
[A2C](https://github.com/lexiconium/cartpole/blob/main/a2c.py): [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
```
python a2c.py
```

### Proximal Policy Optimization
[PPO](https://github.com/lexiconium/cartpole/blob/main/ppo.py): [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
```
python ppo.py
```

### Additional Arguments

```
--env ENV
--batch_size BATCH_SIZE
--num_episodes NUM_EPISODES
--update_interval UPDATE_INTERVAL
--learning_rate LEARNING_RATE
--weight_decay WEIGHT_DECAY
--gamma GAMMA
--epsilon EPSILON
--seed SEED
```
