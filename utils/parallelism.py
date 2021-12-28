from typing import List
from multiprocessing.connection import Connection

from torch.multiprocessing import Process, Pipe
import numpy as np

import gym


class ParallelEnvs:
    def __init__(self, id: str, num_parallel: int):
        self.id = id
        self.num_parallel = num_parallel
        self.envs = []
        self._action_space = None
        self._observation_space = None

        self.parents = []
        self.childs = []
        self.processes = []

        def f(pid: int, child: Connection):
            env = self.envs[pid]
            
            cmd, recv = child.recv()
            if cmd == "action_space":
                action_space = env.action_space
                child.send(action_space)
            elif cmd == "observation_space":
                observation_space = env.observation_space
                child.send(observation_space)
            elif cmd == "step":
                observation, reward, done, info = env.step(recv)
                child.send([observation, reward, done, info])
            elif cmd == "reset":
                observation = env.reset()
                child.send(observation)
            else:
                env.close()

        for pid in range(num_parallel):
            env = gym.make(id)
            env.seed(pid)
            self.envs.append(env)

            parent, child = Pipe()
            self.parents.append(parent)
            self.childs.append(child)

            p = Process(target=f, args=(pid, child), daemon=True)
            p.start()
            self.processes.append(p)

    def __len__(self):
        return self.num_parallel

    def set_action_space(self):
        self._action_space = self.parents[0].send(["action_space", None])
        return self._action_space

    @property
    def action_space(self):
        if self._action_space is None:
            return self.set_action_space()
        return self._action_space

    def set_observation_space(self):
        self._observation_space = self.parents[0].send(["observation_space", None])
        return self._observation_space

    @property
    def observation_space(self):
        if self._observation_space is None:
            return self.set_observation_space()
        return self._observation_space

    def step(self, actions: List):
        for parent, action in zip(self.parents, actions):
            parent.send(["step", action])
        observations, rewards, dones, info = zip(*[parent.recv() for parent in self.parents])
        return np.stack(observations), np.stack(rewards), np.stack(dones), info

    def reset(self):
        for parent in self.parents:
            parent.send("reset", None)
        return np.stack(parent.recv() for parent in self.parents)

    def close(self):
        for parent in self.parents:
            parent.send("close", None)
        for child in self.childs:
            child.join()
