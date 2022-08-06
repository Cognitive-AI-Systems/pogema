from copy import deepcopy

import gym


class IsMultiAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.is_multiagent = True

    @property
    def num_agents(self):
        return self.get_num_agents()


class MetricsForwardingWrapper(gym.Wrapper):
    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        for info in infos:
            if 'metrics' in info:
                info.update(episode_extra_stats=deepcopy(info['metrics']))
        return observations, rewards, dones, infos


class AutoResetWrapper(gym.Wrapper):
    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        if all(dones):
            observations = self.env.reset()
        return observations, rewards, dones, infos
