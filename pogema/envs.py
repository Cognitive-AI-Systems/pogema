from typing import Optional

import numpy as np
import gym
from gym.error import ResetNeeded

from pogema.grid import Grid, GridLifeLong, CooperativeGrid
from pogema.grid_config import GridConfig
from pogema.wrappers.metrics import LifeLongAverageThroughputMetric, NonDisappearEpLengthMetric, \
    NonDisappearCSRMetric, NonDisappearISRMetric, EpLengthMetric, ISRMetric, CSRMetric
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.generator import generate_new_target
from pogema.wrappers.persistence import PersistentWrapper


class ActionsSampler:
    """
    Samples the random actions for the given number of agents using the given seed.
    """

    def __init__(self, num_actions, seed=42):
        self._num_actions = num_actions
        self._rnd = None
        self.update_seed(seed)

    def update_seed(self, seed=None):
        self._rnd = np.random.default_rng(seed)

    def sample_actions(self, dim=1):
        return self._rnd.integers(self._num_actions, size=dim)


class PogemaBase(gym.Env):
    """
    Abstract class of the Pogema environment.
    """
    metadata = {
        "render_modes": ["ansi"],
        # "render_fps": 4,
    }

    def step(self, action):
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, ):
        raise NotImplementedError

    def __init__(self, grid_config: GridConfig = GridConfig()):
        # noinspection PyTypeChecker
        self.grid: Grid = None
        self.grid_config = grid_config

        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(len(self.grid_config.MOVES))
        self._multi_action_sampler = ActionsSampler(self.action_space.n, seed=self.grid_config.seed)

    def _get_agents_obs(self, agent_id=0):
        """
        Returns the observation of the agent with the given id.
        :param agent_id:
        :return:
        """
        return np.concatenate([
            self.grid.get_obstacles_for_agent(agent_id)[None],
            self.grid.get_positions(agent_id)[None],
            self.grid.get_square_target(agent_id)[None]
        ])

    def check_reset(self):
        """
        Checks if the reset needed.
        :return:
        """
        if self.grid is None:
            raise ResetNeeded("Please reset environment first!")

    def render(self, mode='human'):
        """
        Renders the environment using ascii graphics.
        :param mode:
        :return:
        """
        self.check_reset()
        return self.grid.render(mode=mode)

    def sample_actions(self):
        """
        Samples the random actions for the given number of agents.
        :return:
        """
        return self._multi_action_sampler.sample_actions(dim=self.grid_config.num_agents)

    def get_num_agents(self):
        """
        Returns the number of agents in the environment.
        :return:
        """
        return self.grid_config.num_agents


class Pogema(PogemaBase):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.was_on_goal = None
        full_size = self.grid_config.obs_radius * 2 + 1
        if self.grid_config.observation_type == 'default':
            self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3, full_size, full_size))
        elif self.grid_config.observation_type == 'POMAPF':
            self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
                obstacles=gym.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gym.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                xy=gym.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gym.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            )
        elif self.grid_config.observation_type == 'MAPF':
            self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
                obstacles=gym.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gym.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                xy=gym.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gym.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                # global_obstacles=None, # todo define shapes of global state variables
                # global_xy=None,
                # global_target_xy=None,
            )
        else:
            raise ValueError(f"Unknown observation type: {self.grid.config.observation_type}")

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = []

        terminated = []

        self.move_agents(action)
        self.update_was_on_goal()

        for agent_idx in range(self.grid_config.num_agents):

            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.grid.is_active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            terminated.append(on_goal)

        for agent_idx in range(self.grid_config.num_agents):
            if self.grid.on_goal(agent_idx):
                self.grid.hide_agent(agent_idx)
                self.grid.is_active[agent_idx] = False

        infos = self._get_infos()

        observations = self._obs()
        return observations, rewards, terminated, infos

    def _initialize_grid(self):
        self.grid: Grid = Grid(grid_config=self.grid_config)

    def update_was_on_goal(self):
        self.was_on_goal = [self.grid.on_goal(agent_idx) and self.grid.is_active[agent_idx]
                            for agent_idx in range(self.grid_config.num_agents)]

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, ):
        self._initialize_grid()
        self.update_was_on_goal()

        if seed is not None:
            self.grid.seed = seed

        if return_info:
            return self._obs(), self._get_infos()
        return self._obs()

    def _obs(self):
        if self.grid_config.observation_type == 'default':
            return [self._get_agents_obs(index) for index in range(self.grid_config.num_agents)]
        elif self.grid_config.observation_type == 'POMAPF':
            return self._pomapf_obs()

        elif self.grid_config.observation_type == 'MAPF':
            results = self._pomapf_obs()
            global_obstacles = self.grid.get_obstacles()
            global_agents_xy = self.grid.get_agents_xy()
            global_targets_xy = self.grid.get_targets_xy()

            for agent_idx in range(self.grid_config.num_agents):
                result = results[agent_idx]
                result.update(global_obstacles=global_obstacles)
                result['global_xy'] = global_agents_xy[agent_idx]
                result['global_target_xy'] = global_targets_xy[agent_idx]

            return results
        else:
            raise ValueError(f"Unknown observation type: {self.grid.config.observation_type}")

    def _pomapf_obs(self):
        results = []
        agents_xy_relative = self.grid.get_agents_xy_relative()
        targets_xy_relative = self.grid.get_targets_xy_relative()

        for agent_idx in range(self.grid_config.num_agents):
            result = {}
            result['obstacles'] = self.grid.get_obstacles_for_agent(agent_idx)

            result['agents'] = self.grid.get_positions(agent_idx)
            result['xy'] = agents_xy_relative[agent_idx]
            result['target_xy'] = targets_xy_relative[agent_idx]
            results.append(result)
        return results

    def _get_infos(self):
        infos = [dict() for _ in range(self.grid_config.num_agents)]
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]
        return infos

    def move_agents(self, actions):
        if self.grid.config.collision_system == 'priority':
            for agent_idx in range(self.grid_config.num_agents):
                if self.grid.is_active[agent_idx]:
                    self.grid.move(agent_idx, actions[agent_idx])
        elif self.grid.config.collision_system == 'block_both':
            used_cells = {}
            agents_xy = self.grid.get_agents_xy()
            for agent_idx, (x, y) in enumerate(agents_xy):
                if self.grid.is_active[agent_idx]:
                    dx, dy = self.grid_config.MOVES[actions[agent_idx]]
                    used_cells[x + dx, y + dy] = 'blocked' if (x + dx, y + dy) in used_cells else 'visited'
                    used_cells[x, y] = 'blocked'
            for agent_idx in range(self.grid_config.num_agents):
                if self.grid.is_active[agent_idx]:
                    x, y = agents_xy[agent_idx]
                    dx, dy = self.grid_config.MOVES[actions[agent_idx]]
                    if used_cells.get((x + dx, y + dy), None) != 'blocked':
                        self.grid.move(agent_idx, actions[agent_idx])
        else:
            raise ValueError('Unknown collision system: {}'.format(self.grid.config.collision_system))

    def get_agents_xy_relative(self):
        return self.grid.get_agents_xy_relative()

    def get_targets_xy_relative(self):
        return self.grid.get_targets_xy_relative()

    def get_obstacles(self, ignore_borders=False):
        return self.grid.get_obstacles(ignore_borders=ignore_borders)

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_agents_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_targets_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_state(self, ignore_borders=False, as_dict=False):
        return self.grid.get_state(ignore_borders=ignore_borders, as_dict=as_dict)


class PogemaLifeLong(Pogema):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.random_generators: list = [np.random.default_rng(grid_config.seed + i) for i in range(grid_config.num_agents)]

    def _initialize_grid(self):
        self.grid: GridLifeLong = GridLifeLong(grid_config=self.grid_config)

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = []

        infos = [dict() for _ in range(self.grid_config.num_agents)]

        dones = [False] * self.grid_config.num_agents

        self.move_agents(action)
        self.update_was_on_goal()

        for agent_idx in range(self.grid_config.num_agents):
            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.grid.is_active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

            if self.grid.on_goal(agent_idx):
                self.grid.finishes_xy[agent_idx] = generate_new_target(self.random_generators[agent_idx],
                                                                       self.grid.point_to_component,
                                                                       self.grid.component_to_points,
                                                                       self.grid.positions_xy[agent_idx])

        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]

        obs = self._obs()
        return obs, rewards, dones, infos


class PogemaCoopFinish(Pogema):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.num_agents = self.grid_config.num_agents
        self.is_multiagent = True

    def _initialize_grid(self):
        self.grid: CooperativeGrid = CooperativeGrid(grid_config=self.grid_config)

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = [0.0 for _ in range(self.grid_config.num_agents)]

        infos = [dict() for _ in range(self.grid_config.num_agents)]

        self.move_agents(action)
        self.update_was_on_goal()

        is_task_solved = all(self.was_on_goal)
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]

        obs = self._obs()

        dones = [is_task_solved] * self.grid_config.num_agents
        return obs, rewards, dones, infos


class CoopFinishRewardWrapper(gym.Wrapper):
    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        truncated = all(dones)
        rewards = [1.0 if on_goal and truncated else 0.0 for on_goal in self.was_on_goal]
        return observations, rewards, dones, infos


def _make_pogema(grid_config):
    if grid_config.on_target == 'restart':
        env = PogemaLifeLong(grid_config=grid_config)
    elif grid_config.on_target == 'nothing':
        env = PogemaCoopFinish(grid_config=grid_config)
    elif grid_config.on_target == 'finish':
        env = Pogema(grid_config=grid_config)
    else:
        raise KeyError(f'Unknown on_target option: {grid_config.on_target}')

    env = MultiTimeLimit(env, grid_config.max_episode_steps)
    if env.grid_config.on_target == 'nothing':
        env = CoopFinishRewardWrapper(env)
    if env.grid_config.persistent:
        env = PersistentWrapper(env)
    else:
        # adding metrics wrappers
        if grid_config.on_target == 'restart':
            env = LifeLongAverageThroughputMetric(env)
        elif grid_config.on_target == 'nothing':
            env = NonDisappearISRMetric(env)
            env = NonDisappearCSRMetric(env)
            env = NonDisappearEpLengthMetric(env)
        elif grid_config.on_target == 'finish':
            env = ISRMetric(env)
            env = CSRMetric(env)
            env = EpLengthMetric(env)
        else:
            raise KeyError(f'Unknown on_target option: {grid_config.on_target}')

    return env
