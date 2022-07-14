import gym
import numpy as np


class NegativeTimeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_reward: float = -0.1

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)

        return obs, [r + self.step_reward for r in reward], done, infos


class NegativeBackReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.backward_step_reward: float = -0.1
        self.visited: list = [np.zeros((env.config.size, env.config.size)) for i in range(env.config.num_agents)]

    def step(self, action):
        for agent_idx in range(self.env.config.num_agents):
            pos = self.env.grid.positions_xy[agent_idx]
            self.visited[agent_idx][pos[0] - self.env.config.obs_radius, pos[1] - self.env.config.obs_radius] = 1

        obs, reward, done, infos = self.env.step(action)

        for agent_idx in range(self.env.config.num_agents):
            if done[agent_idx]:
                self.visited[agent_idx] = np.zeros_like(self.visited[agent_idx])
            pos = self.env.grid.positions_xy[agent_idx]
            if self.visited[agent_idx][pos[0] - self.env.config.obs_radius, pos[1] - self.env.config.obs_radius]:
                reward[agent_idx] += self.backward_step_reward
            self.visited[agent_idx][pos[0] - self.env.config.obs_radius, pos[1] - self.env.config.obs_radius] = 1

        return obs, reward, done, infos


