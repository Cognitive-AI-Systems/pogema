from gym.wrappers import TimeLimit
import gym

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
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        for agent_idx in range(self.env.get_num_agents()):
            if not done[agent_idx]:
                reward[agent_idx] = 0.0
        return observation, reward, done, info