from copy import deepcopy
import warnings

import numpy as np

from pogema.generator import generate_obstacles, generate_positions_and_targets_fast, \
    get_components, generate_from_possible_positions
from .grid_config import GridConfig
from .grid_registry import in_registry, get_grid
from .utils import render_grid


class Grid:

    def __init__(self, grid_config: GridConfig, add_artificial_border: bool = True, num_retries=10):

        self.config = grid_config
        self.rnd = np.random.default_rng(grid_config.seed)
        if self.config.map is None:
            self.obstacles = generate_obstacles(self.config)
        else:
            self.obstacles = np.array([np.array(line) for line in self.config.map])
        if in_registry(self.config.map_name):
            self.obstacles = get_grid(self.config.map_name).get_obstacles()
        self.obstacles = self.obstacles.astype(np.int32)

        if grid_config.targets_xy and grid_config.agents_xy:
            self.starts_xy = grid_config.agents_xy

            if isinstance(grid_config.targets_xy[0][0], (list, tuple)):
                self.finishes_xy = [sequence[0] for sequence in grid_config.targets_xy]
            else:
                self.finishes_xy = grid_config.targets_xy
                
            if len(self.starts_xy) != len(self.finishes_xy):
                raise IndexError("Can't create task. Please provide agents_xy and targets_xy of the same size.")
            if grid_config.num_agents > len(self.starts_xy):
                raise IndexError(f"Not enough agents_xy and targets_xy to place {grid_config.num_agents} agents")
            self.starts_xy = self.starts_xy[:grid_config.num_agents]
            self.finishes_xy = self.finishes_xy[:grid_config.num_agents]
            for start_xy, finish_xy in zip(self.starts_xy, self.finishes_xy):
                s_x, s_y = start_xy
                f_x, f_y = finish_xy
                if self.config.map is not None and self.obstacles[s_x, s_y] == grid_config.OBSTACLE:
                    warnings.warn(f"There is an obstacle on a start point ({s_x}, {s_y}), replacing with free cell",
                                  Warning, stacklevel=2)
                self.obstacles[s_x, s_y] = grid_config.FREE
                if self.config.map is not None and self.obstacles[f_x, f_y] == grid_config.OBSTACLE:
                    warnings.warn(f"There is an obstacle on a finish point ({f_x}, {f_y}), replacing with free cell",
                                  Warning, stacklevel=2)
                self.obstacles[f_x, f_y] = grid_config.FREE
        elif grid_config.possible_agents_xy and grid_config.possible_targets_xy:
            self.starts_xy, self.finishes_xy = generate_from_possible_positions(self.config)
        else:
            self.starts_xy, self.finishes_xy = generate_positions_and_targets_fast(self.obstacles, self.config)

        if len(self.starts_xy) != len(self.finishes_xy):
            for attempt in range(num_retries):
                if len(self.starts_xy) == len(self.finishes_xy):
                    warnings.warn(f'Created valid configuration only with {attempt} attempts.', Warning, stacklevel=2)
                    break
                if self.config.map is None:
                    self.obstacles = generate_obstacles(self.config)
                self.starts_xy, self.finishes_xy = generate_positions_and_targets_fast(self.obstacles, self.config)

        if not self.starts_xy or not self.finishes_xy or len(self.starts_xy) != len(self.finishes_xy):
            raise OverflowError(
                "Can't create task. Please check grid grid_config, especially density, num_agent and map.")

        if add_artificial_border:
            self.add_artificial_border()

        filled_positions = np.zeros(self.obstacles.shape)
        for x, y in self.starts_xy:
            filled_positions[x, y] = 1

        self.positions = filled_positions
        self.positions_xy = self.starts_xy
        self._initial_xy = deepcopy(self.starts_xy)
        self.is_active = {agent_id: True for agent_id in range(self.config.num_agents)}

    def add_artificial_border(self):
        gc = self.config
        r = gc.obs_radius
        if gc.empty_outside:
            filled_obstacles = np.zeros(np.array(self.obstacles.shape) + r * 2)
        else:
            filled_obstacles = self.rnd.binomial(1, gc.density, np.array(self.obstacles.shape) + r * 2)

        height, width = filled_obstacles.shape
        filled_obstacles[r - 1, r - 1:width - r + 1] = gc.OBSTACLE
        filled_obstacles[r - 1:height - r + 1, r - 1] = gc.OBSTACLE
        filled_obstacles[height - r, r - 1:width - r + 1] = gc.OBSTACLE
        filled_obstacles[r - 1:height - r + 1, width - r] = gc.OBSTACLE
        filled_obstacles[r:height - r, r:width - r] = self.obstacles

        self.obstacles = filled_obstacles

        self.starts_xy = [(x + r, y + r) for x, y in self.starts_xy]
        self.finishes_xy = [(x + r, y + r) for x, y in self.finishes_xy]

    def get_obstacles(self, ignore_borders=False):
        gc = self.config
        if ignore_borders:
            return self.obstacles[gc.obs_radius:-gc.obs_radius, gc.obs_radius:-gc.obs_radius].copy()
        return self.obstacles.copy()

    @staticmethod
    def _cut_borders_xy(positions, obs_radius):
        return [[x - obs_radius, y - obs_radius] for x, y in positions]

    @staticmethod
    def _filter_inactive(pos, active_flags):
        return [pos for idx, pos in enumerate(pos) if active_flags[idx]]

    def get_grid_config(self):
        return deepcopy(self.config)

    # def _get_grid_config(self) -> GridConfig:
    #     return self.env.grid_config

    def _prepare_positions(self, positions, only_active, ignore_borders):
        gc = self.config

        if only_active:
            positions = self._filter_inactive(positions, [idx for idx, active in self.is_active.items() if active])

        if ignore_borders:
            positions = self._cut_borders_xy(positions, gc.obs_radius)

        return positions

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self._prepare_positions(deepcopy(self.positions_xy), only_active, ignore_borders)

    @staticmethod
    def to_relative(coordinates, offset):
        result = deepcopy(coordinates)
        for idx, _ in enumerate(result):
            x, y = result[idx]
            dx, dy = offset[idx]
            result[idx] = x - dx, y - dy
        return result

    def get_agents_xy_relative(self):
        return self.to_relative(self.positions_xy, self._initial_xy)

    def get_targets_xy_relative(self):
        return self.to_relative(self.finishes_xy, self._initial_xy)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self._prepare_positions(deepcopy(self.finishes_xy), only_active, ignore_borders)

    def _normalize_coordinates(self, coordinates):
        gc = self.config

        x, y = coordinates

        x -= gc.obs_radius
        y -= gc.obs_radius

        x /= gc.height - 1
        y /= gc.width - 1

        return x, y

    def get_state(self, ignore_borders=False, as_dict=False):
        agents_xy = list(map(self._normalize_coordinates, self.get_agents_xy(ignore_borders)))
        targets_xy = list(map(self._normalize_coordinates, self.get_targets_xy(ignore_borders)))

        obstacles = self.get_obstacles(ignore_borders)

        if as_dict:
            return {"obstacles": obstacles, "agents_xy": agents_xy, "targets_xy": targets_xy}

        return np.concatenate(list(map(lambda x: np.array(x).flatten(), [agents_xy, targets_xy, obstacles])))

    def get_observation_shape(self):
        full_radius = self.config.obs_radius * 2 + 1
        return 2, full_radius, full_radius

    def get_num_actions(self):
        return len(self.config.MOVES)

    def get_obstacles_for_agent(self, agent_id):
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        return self.obstacles[x - r:x + r + 1, y - r:y + r + 1].astype(np.float32)

    def get_positions(self, agent_id):
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        return self.positions[x - r:x + r + 1, y - r:y + r + 1].astype(np.float32)

    def get_target(self, agent_id):

        x, y = self.positions_xy[agent_id]
        fx, fy = self.finishes_xy[agent_id]
        if x == fx and y == fy:
            return 0.0, 0.0
        rx, ry = fx - x, fy - y
        dist = np.sqrt(rx ** 2 + ry ** 2)
        return rx / dist, ry / dist

    def get_square_target(self, agent_id):
        c = self.config
        full_size = self.config.obs_radius * 2 + 1
        result = np.zeros((full_size, full_size))
        x, y = self.positions_xy[agent_id]
        fx, fy = self.finishes_xy[agent_id]
        dx, dy = x - fx, y - fy

        dx = min(dx, c.obs_radius) if dx >= 0 else max(dx, -c.obs_radius)
        dy = min(dy, c.obs_radius) if dy >= 0 else max(dy, -c.obs_radius)
        result[c.obs_radius - dx, c.obs_radius - dy] = 1
        return result.astype(np.float32)

    def render(self, mode='human', border='thin'):
        gc = self.config
        r = gc.obs_radius

        if border == 'none':
            trim = r
        elif border == 'full':
            trim = 0
        else:
            trim = max(r - 1, 0)

        if trim == 0:
            obs = self.obstacles.copy()
            agents = deepcopy(self.positions_xy)
            targets = deepcopy(self.finishes_xy)
        else:
            obs = self.obstacles[trim:-trim, trim:-trim].copy()
            agents = [[x - trim, y - trim] for x, y in self.positions_xy]
            targets = [[x - trim, y - trim] for x, y in self.finishes_xy]

        render_grid(obs, agents, targets, self.is_active, mode=mode)

    def move_agent_to_cell(self, agent_id, x, y):
        if self.positions[self.positions_xy[agent_id]] == self.config.FREE:
            raise KeyError("Agent {} is not in the map".format(agent_id))
        self.positions[self.positions_xy[agent_id]] = self.config.FREE
        if self.obstacles[x, y] != self.config.FREE or self.positions[x, y] != self.config.FREE:
            raise ValueError(f"Can't force agent to blocked position {x} {y}")
        self.positions_xy[agent_id] = x, y
        self.positions[self.positions_xy[agent_id]] = self.config.OBSTACLE

    def has_obstacle(self, x, y):
        return self.obstacles[x, y] == self.config.OBSTACLE

    def move_without_checks(self, agent_id, action):
        x, y = self.positions_xy[agent_id]
        dx, dy = self.config.MOVES[action]
        self.positions[x, y] = self.config.FREE
        self.positions[x+dx, y+dy] = self.config.OBSTACLE
        self.positions_xy[agent_id] = (x+dx, y+dy)

    def move(self, agent_id, action):
        x, y = self.positions_xy[agent_id]
        dx, dy = self.config.MOVES[action]
        if self.obstacles[x + dx, y + dy] == self.config.FREE:
            if self.positions[x + dx, y + dy] == self.config.FREE:
                self.positions[x, y] = self.config.FREE
                x += dx
                y += dy
                self.positions[x, y] = self.config.OBSTACLE
        self.positions_xy[agent_id] = (x, y)

    def on_goal(self, agent_id):
        return self.positions_xy[agent_id] == self.finishes_xy[agent_id]

    def is_active(self, agent_id):
        return self.is_active[agent_id]

    def hide_agent(self, agent_id):
        if not self.is_active[agent_id]:
            return False
        self.is_active[agent_id] = False

        self.positions[self.positions_xy[agent_id]] = self.config.FREE

        return True

    def show_agent(self, agent_id):
        if self.is_active[agent_id]:
            return False

        self.is_active[agent_id] = True
        if self.positions[self.positions_xy[agent_id]] == self.config.OBSTACLE:
            raise KeyError("The cell is already occupied")
        self.positions[self.positions_xy[agent_id]] = self.config.OBSTACLE
        return True


class GridLifeLong(Grid):
    def __init__(self, grid_config: GridConfig, add_artificial_border: bool = True, num_retries=10):

        super().__init__(grid_config, add_artificial_border, num_retries)

        self.component_to_points, self.point_to_component = get_components(grid_config, self.obstacles,
                                                                           self.positions_xy, self.finishes_xy)

        for i in range(len(self.positions_xy)):
            position, target = self.positions_xy[i], self.finishes_xy[i]
            if self.point_to_component[position] != self.point_to_component[target]:
                warnings.warn(f"The start point ({position[0]}, {position[1]}) and the goal"
                              f" ({target[0]}, {target[1]}) are in different components. The goal is changed.",
                              Warning, stacklevel=2)
