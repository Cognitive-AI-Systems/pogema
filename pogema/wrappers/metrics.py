import gym


class AbstractMetric(gym.Wrapper):
    def _compute_stats(self, step, is_on_goal, truncated):
        raise NotImplementedError

    def __init__(self, env):
        super().__init__(env)
        self._current_step = 0

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        truncated = all(done)
        metric = self._compute_stats(self._current_step, self.was_on_goal, all(done))
        self._current_step += 1
        if truncated:
            self._current_step = 0

        if metric:
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(**metric)

        return obs, reward, done, infos


class LifeLongAverageThroughputMetric(AbstractMetric):

    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, truncated):
        for agent_idx, on_goal in enumerate(is_on_goal):
            if on_goal:
                self._solved_instances += 1
        if truncated:
            result = {'avg_throughput': self._solved_instances / self.grid_config.max_episode_steps}
            self._solved_instances = 0
            return result


class NonDisappearCSRMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, truncated):
        if truncated:
            return {'CSR': float(all(is_on_goal))}


class NonDisappearISRMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, truncated):
        if truncated:
            return {'ISR': float(sum(is_on_goal)) / self.get_num_agents()}


class NonDisappearEpLengthMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, truncated):
        if truncated:
            return {'ep_length': step}


class EpLengthMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        # self._current_step = 0
        self._solve_time = [None for _ in range(self.get_num_agents())]

    def _compute_stats(self, step, is_on_goal, truncated):
        for idx, on_goal in enumerate(is_on_goal):
            if on_goal or truncated:
                self._solve_time[idx] = step

        if truncated:
            result = {'ep_length': sum(self._solve_time) / self.get_num_agents()}
            self._solve_time = [None for _ in range(self.get_num_agents())]
            return result


class CSRMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, truncated):
        self._solved_instances += sum(is_on_goal)
        if truncated:
            results = {'CSR': float(self._solved_instances == self.get_num_agents())}
            self._solved_instances = 0
            return results


class ISRMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, truncated):
        self._solved_instances += sum(is_on_goal)
        if truncated:
            results = {'ISR': self._solved_instances / self.get_num_agents()}
            self._solved_instances = 0
            return results
