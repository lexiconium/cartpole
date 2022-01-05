import multiprocessing

import numpy as np
import gym


class ParallelEnvs:
    def __init__(self, id: str, num_parallel: int):
        self.id = id
        self.num_parallel = num_parallel
        self.pool = multiprocessing.Pool(num_parallel)
        self.envs = [gym.make(id) for _ in range(num_parallel)]
        self.pid = 0

        self._observation_space = None
        self._action_space = None

    def __repr__(self):
        return self.id

    def __len__(self):
        return self.num_parallel

    def _set_observation_space(self):
        self._observation_space = self.envs[0].observation_space

    @property
    def observation_space(self):
        if not self._observation_space:
            self._set_observation_space()
        return self._observation_space

    def _set_action_space(self):
        self._action_space = self.envs[0].action_space

    @property
    def action_space(self):
        if not self._action_space:
            self._set_action_space()
        return self._action_space

    @staticmethod
    def _step(env_action_pair):
        env, action = env_action_pair
        next_state, reward, done, _ = env.step(action)
        if done:
            env.reset()

        return env, next_state, reward, done

    def step(self, actions: np.ndarray):
        next_states, rewards, dones = [], [], []

        env_action_pairs = list(zip(self.envs, actions))
        for idx, (env, next_state, reward, done) in enumerate(self.pool.imap(self._step, env_action_pairs)):
            self.envs[idx] = env
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

        return np.array(next_states), np.array(rewards), np.array(dones)

    def reset(self):
        return np.stack([env.reset() for env in self.envs])

    def close(self):
        self.pool.close()
        self.pool.join()
