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

# class CoopRewardWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.prev_observation = None

#     def step(self, action):
#         observation, reward, done, info = self.env.step(action)
#         centre = int(len(observation[0][0][0]) / 2)
#         flag_all_on_target = True
#         for agent_idx in range(self.env.get_num_agents()):
#             if not done[agent_idx]:
#                 reward[agent_idx] = 0.0
            
#             if np.isclose(1.0, observation[agent_idx][2][centre][centre]):
#                 reward[agent_idx] = 1.0
#             else:
#                 flag_all_on_target = False
#                 if abs(agents_xy[0][0] - agents_xy[1][0]) + abs(agents_xy[0][1] - agents_xy[1][1]) == 1:
#                     reward[agent_idx] = 1.3
#             if len(self.previous_agents_xy) >= 2:
#                 if self.previous_agents_xy[-1][1-agent_idx] == self.previous_agents_xy[-2][1-agent_idx] and self.previous_agents_xy[-1][1-agent_idx] != agents_xy[1-agent_idx]:
#                     reward[agent_idx] = 1.5

#         self.previous_agents_xy.append(agents_xy)
#         return observation, reward, done, info


class NegativeCoopRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_observation = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        centre = int(len(observation[0][0][0]) / 2)
        flag_all_on_target = True
        for agent_idx in range(self.env.get_num_agents()):
            if not done[agent_idx]:
                reward[agent_idx] = 0.0
            # if np.isclose(1.0, observation[agent_idx][2][centre][centre]):
            #     reward[agent_idx] = 0.5
            # else:
            #     flag_all_on_target = False
            if self.prev_observation is not None:
                if np.isclose(1.0, observation[agent_idx][1][centre][centre+1]) and np.isclose(1.0, self.prev_observation[agent_idx][1][centre][centre+1]):
                    reward[agent_idx] -= 0.1
                elif np.isclose(1.0, observation[agent_idx][1][centre][centre-1]) and np.isclose(1.0, self.prev_observation[agent_idx][1][centre][centre-1]):
                    reward[agent_idx] -= 0.1
                elif np.isclose(1.0, observation[agent_idx][1][centre+1][centre]) and np.isclose(1.0, self.prev_observation[agent_idx][1][centre+1][centre]):
                    reward[agent_idx] -= 0.1
                elif np.isclose(1.0, observation[agent_idx][1][centre-1][centre]) and np.isclose(1.0, self.prev_observation[agent_idx][1][centre-1][centre]):
                    reward[agent_idx] -= 0.1
                    
        for agent_idx in range(self.env.get_num_agents()):
            if done[agent_idx] and np.isclose(1.0, observation[agent_idx][2][centre][centre]):
                reward[agent_idx] = 1.0
        self.prev_observation = observation
        return observation, reward, done, info 