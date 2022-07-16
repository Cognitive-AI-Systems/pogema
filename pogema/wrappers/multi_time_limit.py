from gym.wrappers import TimeLimit
import gym
import numpy as np

class MultiTimeLimit(TimeLimit):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            for agent_idx in range(self.env.get_num_agents()):
                info[agent_idx]["TimeLimit.truncated"] = not done[agent_idx]
            done = [True] * self.env.get_num_agents()
        return observation, reward, done, info

class CoopRewardWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self.prev_observation = None
        self.elapsed_steps = 0
        self._max_episode_steps = max_episode_steps

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.elapsed_steps += 1
        centre = int(len(observation[0][0][0]) / 2)
        for agent_idx in range(self.env.get_num_agents()):
            if self.elapsed_steps >= self._max_episode_steps:
                reward[agent_idx] = 0.0
                if np.isclose(1.0, observation[agent_idx][2][centre][centre]):
                    reward[agent_idx] = 1.0
        return observation, reward, done, info

class NegativeCoopRewardWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self.prev_observation = None
        self.elapsed_steps = 0
        self._max_episode_steps = max_episode_steps

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.elapsed_steps += 1
        centre = int(len(observation[0][0][0]) / 2)
        for agent_idx in range(self.env.get_num_agents()):
            if self.elapsed_steps >= self._max_episode_steps:
                reward[agent_idx] = 0.0
                if np.isclose(1.0, observation[agent_idx][2][centre][centre]):
                    reward[agent_idx] = 1.0
                    if self.prev_observation is not None:
                        if np.isclose(1.0, observation[agent_idx][1][centre][centre+1]) and np.isclose(1.0, self.prev_observation[agent_idx][1][centre][centre+1]):
                            reward[agent_idx] -= 0.5
                        elif np.isclose(1.0, observation[agent_idx][1][centre][centre-1]) and np.isclose(1.0, self.prev_observation[agent_idx][1][centre][centre-1]):
                            reward[agent_idx] -= 0.5
                        elif np.isclose(1.0, observation[agent_idx][1][centre+1][centre]) and np.isclose(1.0, self.prev_observation[agent_idx][1][centre+1][centre]):
                            reward[agent_idx] -= 0.5
                        elif np.isclose(1.0, observation[agent_idx][1][centre-1][centre]) and np.isclose(1.0, self.prev_observation[agent_idx][1][centre-1][centre]):
                            reward[agent_idx] -= 0.5
        self.prev_observation = observation
        return observation, reward, done, info