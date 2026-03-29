from copy import deepcopy

from pogema.wrappers.base import PogemaWrapper


class IsMultiAgentWrapper(PogemaWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.is_multiagent = True

    @property
    def num_agents(self):
        return self.unwrapped.get_num_agents()


class MetricsForwardingWrapper(PogemaWrapper):
    def step(self, action):

        observations, rewards, terminated, truncated, infos = self.env.step(action)
        for info in infos:
            if 'metrics' in info:
                info.update(episode_extra_stats=deepcopy(info['metrics']))
        return observations, rewards, terminated, truncated, infos


class AutoResetWrapper(PogemaWrapper):
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if all(terminated) or all(truncated):
            observations, info = self.env.reset()
        return observations, rewards, terminated, truncated, infos
