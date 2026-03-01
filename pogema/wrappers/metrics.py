import time

import numpy as np

from pogema.wrappers.base import PogemaWrapper


class AbstractMetric(PogemaWrapper):
    def _compute_stats(self, step, is_on_goal, finished):
        raise NotImplementedError

    def __init__(self, env):
        super().__init__(env)
        self._current_step = 0

    def step(self, action):
        obs, reward, terminated, truncated, infos = self.env.step(action)
        finished = all(truncated) or all(terminated)

        metric = self._compute_stats(self._current_step, self.unwrapped.was_on_goal, finished)
        self._current_step += 1
        if finished:
            self._current_step = 0

        if metric:
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(**metric)

        return obs, reward, terminated, truncated, infos


class LifeLongAverageThroughputMetric(AbstractMetric):

    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, finished):
        for _agent_idx, on_goal in enumerate(is_on_goal):
            if on_goal:
                self._solved_instances += 1
        if finished:
            result = {'avg_throughput': self._solved_instances / self.unwrapped.grid_config.max_episode_steps}
            self._solved_instances = 0
            return result


class NonDisappearCSRMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, finished):
        if finished:
            return {'CSR': float(all(is_on_goal))}


class NonDisappearISRMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, finished):
        if finished:
            return {'ISR': float(sum(is_on_goal)) / self.unwrapped.get_num_agents()}


class NonDisappearEpLengthMetric(AbstractMetric):

    def _compute_stats(self, step, is_on_goal, finished):
        if finished:
            return {'ep_length': step + 1}


class EpLengthMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solve_time = [None for _ in range(self.unwrapped.get_num_agents())]

    def _compute_stats(self, step, is_on_goal, finished):
        for idx, on_goal in enumerate(is_on_goal):
            if self._solve_time[idx] is None:
                if on_goal or finished:
                    self._solve_time[idx] = step

        if finished:
            result = {'ep_length': sum(self._solve_time) / self.unwrapped.get_num_agents() + 1}
            self._solve_time = [None for _ in range(self.unwrapped.get_num_agents())]
            return result


class CSRMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, finished):
        self._solved_instances += sum(is_on_goal)
        if finished:
            results = {'CSR': float(self._solved_instances == self.unwrapped.get_num_agents())}
            self._solved_instances = 0
            return results


class ISRMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solved_instances = 0

    def _compute_stats(self, step, is_on_goal, finished):
        self._solved_instances += sum(is_on_goal)
        if finished:
            results = {'ISR': self._solved_instances / self.unwrapped.get_num_agents()}
            self._solved_instances = 0
            return results


class SumOfCostsAndMakespanMetric(AbstractMetric):
    def __init__(self, env):
        super().__init__(env)
        self._solve_time = [None for _ in range(self.unwrapped.get_num_agents())]

    def _compute_stats(self, step, is_on_goal, finished):
        for idx, on_goal in enumerate(is_on_goal):
            if self._solve_time[idx] is None and (on_goal or finished):
                self._solve_time[idx] = step
            if not on_goal and not finished:
                self._solve_time[idx] = None

        if finished:
            result = {'SoC': sum(self._solve_time) + self.unwrapped.get_num_agents(), 'makespan': max(self._solve_time) + 1}
            self._solve_time = [None for _ in range(self.unwrapped.get_num_agents())]
            return result


class AgentsDensityWrapper(PogemaWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._avg_agents_density = None

    def count_agents(self, observations):
        avg_agents_density = []
        for obs in observations:
            traversable_cells = np.size(obs['obstacles']) - np.count_nonzero(obs['obstacles'])
            avg_agents_density.append(np.count_nonzero(obs['agents']) / traversable_cells)
        self._avg_agents_density.append(np.mean(avg_agents_density))

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        self.count_agents(observations)
        if all(terminated) or all(truncated):
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(avg_agents_density=float(np.mean(self._avg_agents_density)))
        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        self._avg_agents_density = []
        observations, info = self.env.reset(**kwargs)
        self.count_agents(observations)
        return observations, info


class RuntimeMetricWrapper(PogemaWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._start_time = None
        self._env_step_time = None

    def step(self, actions):
        env_step_start = time.monotonic()
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        env_step_end = time.monotonic()
        self._env_step_time += env_step_end - env_step_start
        if all(terminated) or all(truncated):
            final_time = time.monotonic() - self._start_time - self._env_step_time
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(runtime=final_time)
        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._start_time = time.monotonic()
        self._env_step_time = 0.0
        return obs
