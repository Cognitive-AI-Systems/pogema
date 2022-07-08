import string
import sys
from contextlib import closing
from copy import deepcopy
import warnings

import numpy as np
from gym import utils
from io import StringIO

from pogema.generator import generate_obstacles, generate_positions_and_targets_fast
from .grid_config import GridConfig


class Grid:

    def __init__(self, grid_config: GridConfig, add_artificial_border: bool = True, num_retries=10):

        self.config = grid_config
        self.rnd = np.random.default_rng(grid_config.seed)

        if self.config.map is None:
            obstacles = generate_obstacles(self.config)
        else:
            obstacles = np.array([np.array(line) for line in self.config.map])

        if grid_config.targets_xy and grid_config.agents_xy:
            starts_xy, finishes_xy = grid_config.agents_xy, grid_config.targets_xy
            if len(starts_xy) != len(finishes_xy):
                raise IndexError("Can't create task. Please provide agents_xy and targets_xy of the same size.")
            grid_config.num_agents = len(starts_xy)
            for start_xy, finish_xy in zip(starts_xy, finishes_xy):
                s_x, s_y = start_xy
                f_x, f_y = finish_xy
                if self.config.map is not None and obstacles[s_x, s_y] == grid_config.OBSTACLE:
                    warnings.warn(f"There is an obstacle on a start point ({s_x}, {s_y}), replacing with free cell",
                                  Warning, stacklevel=2)
                obstacles[s_x, s_y] = grid_config.FREE
                if self.config.map is not None and obstacles[f_x, f_y] == grid_config.OBSTACLE:
                    warnings.warn(f"There is an obstacle on a finish point ({s_x}, {s_y}), replacing with free cell",
                                  Warning, stacklevel=2)
                obstacles[f_x, f_y] = grid_config.FREE
        else:
            starts_xy, finishes_xy = generate_positions_and_targets_fast(obstacles, self.config)

        if len(starts_xy) != len(finishes_xy):
            for attempt in range(num_retries):
                if len(starts_xy) == len(finishes_xy):
                    warnings.warn(f'Created valid configuration only with {attempt} attempts.', Warning, stacklevel=2)
                    break
                if self.config.map is None:
                    obstacles = generate_obstacles(self.config)
                starts_xy, finishes_xy = generate_positions_and_targets_fast(obstacles, self.config)

        if not starts_xy or not finishes_xy or len(starts_xy) != len(finishes_xy):
            raise OverflowError("Can't create task. Please check grid config, especially density, num_agent and map.")

        if add_artificial_border:
            r = self.config.obs_radius
            if grid_config.empty_outside:
                filled_obstacles = np.zeros(np.array(obstacles.shape) + r * 2)
            else:
                filled_obstacles = self.rnd.binomial(1, grid_config.density, np.array(obstacles.shape) + r * 2)

            height, width = filled_obstacles.shape
            filled_obstacles[r - 1, r - 1:width - r + 1] = grid_config.OBSTACLE
            filled_obstacles[r - 1:height - r + 1, r - 1] = grid_config.OBSTACLE
            filled_obstacles[height - r, r - 1:width - r + 1] = grid_config.OBSTACLE
            filled_obstacles[r - 1:height - r + 1, width - r] = grid_config.OBSTACLE
            filled_obstacles[r:height - r, r:width - r] = obstacles

            obstacles = filled_obstacles

            starts_xy = [(x + r, y + r) for x, y in starts_xy]
            finishes_xy = [(x + r, y + r) for x, y in finishes_xy]

        filled_positions = np.zeros(obstacles.shape)
        for x, y in starts_xy:
            filled_positions[x, y] = 1

        self.obstacles = obstacles
        self.positions = filled_positions
        self.finishes_xy = finishes_xy
        self.positions_xy = starts_xy
        self._initial_xy = deepcopy(starts_xy)
        self.inactive = set()
        self.active = set([agent_id for agent_id in range(self.config.num_agents)])

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
    #     return self.env.config

    def _prepare_positions(self, positions, only_active, ignore_borders):
        gc = self.config

        if only_active:
            positions = self._filter_inactive(positions, list(self.active))

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

        x /= gc.size - 1
        y /= gc.size - 1

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
        return self.obstacles[x - r:x + r + 1, y - r:y + r + 1]

    def get_positions(self, agent_id):
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        return self.positions[x - r:x + r + 1, y - r:y + r + 1]

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
        return result

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        chars = string.digits + string.ascii_letters + string.punctuation
        positions_map = {(x, y): id_ for id_, (x, y) in enumerate(self.positions_xy) if id_ not in self.inactive}
        finishes_map = {(x, y): id_ for id_, (x, y) in enumerate(self.finishes_xy) if id_ not in self.inactive}
        for line_index, line in enumerate(self.obstacles):
            out = ''
            for cell_index, cell in enumerate(line):
                if cell == self.config.FREE:
                    agent_id = positions_map.get((line_index, cell_index), None)
                    finish_id = finishes_map.get((line_index, cell_index), None)

                    if agent_id is not None:
                        out += str(utils.colorize(' ' + chars[agent_id % len(chars)] + ' ', color='red', bold=True,
                                                  highlight=False))
                    elif finish_id is not None:
                        out += str(
                            utils.colorize('|' + chars[finish_id % len(chars)] + '|', 'white', highlight=False))
                    else:
                        out += str(utils.colorize(str(' . '), 'white', highlight=False))
                else:
                    out += str(utils.colorize(str('   '), 'cyan', bold=False, highlight=True))
            out += '\n'
            outfile.write(out)

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def move(self, agent_id, action):
        x, y = self.positions_xy[agent_id]

        self.positions[x, y] = self.config.FREE

        dx, dy = self.config.MOVES[action]
        if self.obstacles[x + dx, y + dy] == self.config.FREE and self.positions[x + dx, y + dy] == self.config.FREE:
            x += dx
            y += dy

        self.positions_xy[agent_id] = (x, y)
        # TODO: we need to recognize whether its an agent standing or an obstacle
        self.positions[x, y] = self.config.OBSTACLE

    def on_goal(self, agent_id):
        return self.positions_xy[agent_id] == self.finishes_xy[agent_id]

    def is_active(self, agent_id):
        return agent_id in self.inactive

    def hide_agent(self, agent_id):
        if agent_id in self.inactive:
            return False

        self.inactive.add(agent_id)
        self.active.remove(agent_id)

        x, y = self.positions_xy[agent_id]
        self.positions[x, y] = self.config.FREE

        return True


class CooperativeGrid(Grid):
    def __init__(self, grid_config: GridConfig, add_artificial_border: bool = True, num_retries=10):
        super().__init__(grid_config)

    def move(self, agent_id, action):
        x, y = self.positions_xy[agent_id]
        fx, fy = self.finishes_xy[agent_id]
        dx, dy = self.config.MOVES[action]
        if self.obstacles[x + dx, y + dy] == self.config.FREE:
            if self.positions[x + dx, y + dy] == self.config.FREE:
                self.positions[x, y] = self.config.FREE
                x += dx
                y += dy
                self.positions[x, y] = self.config.OBSTACLE
        self.positions_xy[agent_id] = (x, y)
        return x == fx and y == fy
