from typing import Optional
import warnings
import numpy as np
import gymnasium
from gymnasium.error import ResetNeeded

from pogema.grid import Grid, GridLifeLong
from pogema.grid_config import GridConfig
from pogema.wrappers.metrics import LifeLongAverageThroughputMetric, NonDisappearEpLengthMetric, \
    NonDisappearCSRMetric, NonDisappearISRMetric, EpLengthMetric, ISRMetric, CSRMetric, SumOfCostsAndMakespanMetric
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.generator import generate_new_target, generate_from_possible_targets
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


class PogemaBase(gymnasium.Env):
    """
    Abstract class of the Pogema environment.
    """
    metadata = {"render_modes": ["ansi"], }

    def step(self, action):
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, ):
        raise NotImplementedError

    def __init__(self, grid_config: GridConfig = GridConfig()):
        # noinspection PyTypeChecker
        self.grid: Grid = None
        self.grid_config = grid_config

        self.action_space: gymnasium.spaces.Discrete = gymnasium.spaces.Discrete(len(self.grid_config.MOVES))
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
            self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(3, full_size, full_size))
        elif self.grid_config.observation_type == 'POMAPF':
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            )
        elif self.grid_config.observation_type == 'MAPF':
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
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
        truncated = [False] * self.grid_config.num_agents
        return observations, rewards, terminated, truncated, infos

    def _initialize_grid(self):
        self.grid: Grid = Grid(grid_config=self.grid_config)

    def update_was_on_goal(self):
        self.was_on_goal = [self.grid.on_goal(agent_idx) and self.grid.is_active[agent_idx]
                            for agent_idx in range(self.grid_config.num_agents)]

    def reset(self, seed: Optional[int] = None, return_info: bool = True, options: Optional[dict] = None, ):
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
            result = {'obstacles': self.grid.get_obstacles_for_agent(agent_idx),
                      'agents': self.grid.get_positions(agent_idx),
                      'xy': agents_xy_relative[agent_idx],
                      'target_xy': targets_xy_relative[agent_idx]}

            results.append(result)
        return results

    def _get_infos(self):
        infos = [dict() for _ in range(self.grid_config.num_agents)]
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]
        return infos

    def _revert_action(self, agent_idx, used_cells, cell, actions):
        actions[agent_idx] = 0
        used_cells[cell].remove(agent_idx)
        new_cell = self.grid.positions_xy[agent_idx]
        if new_cell in used_cells and len(used_cells[new_cell]) > 0:
            used_cells[new_cell].append(agent_idx)
            return self._revert_action(used_cells[new_cell][0], used_cells, new_cell, actions)
        else:
            used_cells.setdefault(new_cell, []).append(agent_idx)
        return actions, used_cells

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
        elif self.grid.config.collision_system == 'soft':
            used_cells = dict()
            used_edges = dict()
            agents_xy = self.grid.get_agents_xy()
            for agent_idx, (x, y) in enumerate(agents_xy):
                if self.grid.is_active[agent_idx]:
                    dx, dy = self.grid.config.MOVES[actions[agent_idx]]
                    used_cells.setdefault((x + dx, y + dy), []).append(agent_idx)
                    used_edges[x, y, x + dx, y + dy] = [agent_idx]
                    if dx != 0 or dy != 0:
                        used_edges.setdefault((x + dx, y + dy, x, y), []).append(agent_idx)
            for agent_idx, (x, y) in enumerate(agents_xy):
                if self.grid.is_active[agent_idx]:
                    dx, dy = self.grid.config.MOVES[actions[agent_idx]]
                    if len(used_edges[x, y, x + dx, y + dy]) > 1:
                        used_cells[x + dx, y + dy].remove(agent_idx)
                        used_cells.setdefault((x, y), []).append(agent_idx)
                        actions[agent_idx] = 0
            for agent_idx in reversed(range(len(agents_xy))):
                x, y = agents_xy[agent_idx]
                if self.grid.is_active[agent_idx]:
                    dx, dy = self.grid.config.MOVES[actions[agent_idx]]
                    if len(used_cells[x + dx, y + dy]) > 1 or self.grid.has_obstacle(x + dx, y + dy):
                        actions, used_cells = self._revert_action(agent_idx, used_cells, (x + dx, y + dy), actions)
            for agent_idx in range(self.grid_config.num_agents):
                if self.grid.is_active[agent_idx]:
                    self.grid.move_without_checks(agent_idx, actions[agent_idx])
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
        self.current_goal_indices = [0] * grid_config.num_agents
        self.has_custom_sequences = grid_config.targets_xy is not None

    def _initialize_grid(self):
        self.grid: GridLifeLong = GridLifeLong(grid_config=self.grid_config)

        main_rng = np.random.default_rng(self.grid_config.seed)
        seeds = main_rng.integers(np.iinfo(np.int32).max, size=self.grid_config.num_agents)
        self.random_generators = [np.random.default_rng(seed) for seed in seeds]
        
    def get_lifelong_targets_xy(self, ignore_borders=False):
        if self.has_custom_sequences:
            if ignore_borders:
                return self.grid_config.targets_xy
            else:
                return [[[x + self.grid_config.obs_radius, y + self.grid_config.obs_radius] for x, y in sequence] 
                        for sequence in self.grid_config.targets_xy]
        
        sequences = []

        main_rng = np.random.default_rng(self.grid_config.seed)
        seeds = main_rng.integers(np.iinfo(np.int32).max, size=self.grid_config.num_agents)
        temp_generators = [np.random.default_rng(seed) for seed in seeds]

        for agent_idx in range(self.grid_config.num_agents):
            agent_sequence = []
            start_pos = self.get_agents_xy(ignore_borders=ignore_borders)[agent_idx]
            initial_target = self.get_targets_xy(ignore_borders=ignore_borders)[agent_idx]
            agent_sequence.append(initial_target)
            current_pos = initial_target
            total_distance = abs(start_pos[0] - initial_target[0]) + abs(start_pos[1] - initial_target[1])
            
            while total_distance < self.grid_config.max_episode_steps:
                if ignore_borders:
                    generator_pos = (current_pos[0] + self.grid_config.obs_radius, 
                                   current_pos[1] + self.grid_config.obs_radius)
                else:
                    generator_pos = tuple(current_pos)

                if self.grid_config.possible_targets_xy is not None:
                    new_goal = generate_from_possible_targets(
                        temp_generators[agent_idx], 
                        self.grid_config.possible_targets_xy, 
                        generator_pos
                    )
                    if ignore_borders:
                        goal_coords = list(new_goal)
                    else:
                        goal_coords = [new_goal[0] + self.grid_config.obs_radius, 
                                       new_goal[1] + self.grid_config.obs_radius]
                else:
                    new_goal = generate_new_target(
                        temp_generators[agent_idx],
                        self.grid.point_to_component,
                        self.grid.component_to_points,
                        generator_pos
                    )
                    if ignore_borders:
                        goal_coords = [new_goal[0] - self.grid_config.obs_radius, 
                                       new_goal[1] - self.grid_config.obs_radius]
                    else:
                        goal_coords = list(new_goal)

                agent_sequence.append(goal_coords)
                total_distance += abs(current_pos[0] - goal_coords[0]) + abs(current_pos[1] - goal_coords[1])
                current_pos = goal_coords
            sequences.append(agent_sequence)
        return sequences

    def reset(self, seed: Optional[int] = None, return_info: bool = True, options: Optional[dict] = None):
        self.current_goal_indices = [0] * self.grid_config.num_agents
        return super().reset(seed=seed, return_info=return_info, options=options)

    def _generate_new_target(self, agent_idx):
        if self.has_custom_sequences:
            agent_targets = self.grid_config.targets_xy[agent_idx]
            current_idx = self.current_goal_indices[agent_idx]
            next_target = agent_targets[(current_idx + 1) % len(agent_targets)]
            
            self.current_goal_indices[agent_idx] = (current_idx + 1) % len(agent_targets)
            
            if self.current_goal_indices[agent_idx] == 0 and current_idx == len(agent_targets) - 1:
                warnings.warn(
                    f"Agent {agent_idx} has completed all {len(agent_targets)} provided targets and "
                    f"is cycling back to the beginning. Provide more targets for the "
                    f"{self.grid_config.max_episode_steps} episode length.",
                    UserWarning,
                    stacklevel=2
                )
            
            return (next_target[0] + self.grid_config.obs_radius, 
                   next_target[1] + self.grid_config.obs_radius)
        elif self.grid_config.possible_targets_xy is not None:
            new_goal = generate_from_possible_targets(self.random_generators[agent_idx], 
                                                     self.grid_config.possible_targets_xy, 
                                                     self.grid.positions_xy[agent_idx])
            return (new_goal[0] + self.grid_config.obs_radius, new_goal[1] + self.grid_config.obs_radius)
        else:
            return generate_new_target(self.random_generators[agent_idx],
                                    self.grid.point_to_component,
                                    self.grid.component_to_points,
                                    self.grid.positions_xy[agent_idx])

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = []

        infos = [dict() for _ in range(self.grid_config.num_agents)]

        self.move_agents(action)
        self.update_was_on_goal()

        for agent_idx in range(self.grid_config.num_agents):
            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.grid.is_active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

            if self.grid.on_goal(agent_idx):
                self.grid.finishes_xy[agent_idx] = self._generate_new_target(agent_idx)

        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]

        obs = self._obs()

        terminated = [False] * self.grid_config.num_agents
        truncated = [False] * self.grid_config.num_agents
        return obs, rewards, terminated, truncated, infos


class PogemaCoopFinish(Pogema):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.num_agents = self.grid_config.num_agents
        self.is_multiagent = True

    def _initialize_grid(self):
        self.grid: Grid = Grid(grid_config=self.grid_config)

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents

        infos = [dict() for _ in range(self.grid_config.num_agents)]

        self.move_agents(action)
        self.update_was_on_goal()

        is_task_solved = all(self.was_on_goal)
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]

        obs = self._obs()

        terminated = [is_task_solved] * self.grid_config.num_agents
        truncated = [False] * self.grid_config.num_agents
        rewards = [1.0 if is_task_solved else 0.0 for _ in range(self.grid_config.num_agents)]
        return obs, rewards, terminated, truncated, infos


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
            env = SumOfCostsAndMakespanMetric(env)
        elif grid_config.on_target == 'finish':
            env = ISRMetric(env)
            env = CSRMetric(env)
            env = EpLengthMetric(env)
        else:
            raise KeyError(f'Unknown on_target option: {grid_config.on_target}')

    return env
