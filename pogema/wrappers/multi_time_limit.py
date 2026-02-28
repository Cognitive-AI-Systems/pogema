from pogema.wrappers.base import PogemaWrapper


class MultiTimeLimit(PogemaWrapper):
    def __init__(self, env, max_episode_steps: int):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = [True] * self.get_num_agents()
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def set_elapsed_steps(self, elapsed_steps: int) -> None:
        assert elapsed_steps >= 0
        self._elapsed_steps = elapsed_steps
