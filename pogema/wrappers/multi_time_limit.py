from gym.wrappers import TimeLimit


class MultiTimeLimit(TimeLimit):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            for agent_idx in range(self.get_num_agents()):
                info[agent_idx]["TimeLimit.truncated"] = not done[agent_idx]
            done = [True] * self.get_num_agents()
        return observation, reward, done, info

    def set_elapsed_steps(self, elapsed_steps):
        if not self.grid_config.persistent:
            raise ValueError("Cannot set elapsed steps for non-persistent environment!")
        assert elapsed_steps >= 0
        self._elapsed_steps = elapsed_steps
