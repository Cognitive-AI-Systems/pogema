import numpy as np

from pogema.utils import check_grid, render_grid

GRID_STR_REGISTRY = {}


def in_registry(name):
    return name in GRID_STR_REGISTRY


def get_grid(name):
    if in_registry(name):
        return GRID_STR_REGISTRY[name]
    else:
        raise KeyError(f"Grid with name {name} not found")


class RegisteredGrid:
    FREE = 0
    OBSTACLE = 1

    def str_to_grid(self, grid_str):
        obstacles = []
        agents = {}
        targets = {}
        for idx, line in enumerate(grid_str.split()):
            row = []
            for char in line:
                if char == '.':
                    row.append(self.FREE)
                elif char == '#':
                    row.append(self.OBSTACLE)
                elif 'A' <= char <= 'Z':
                    targets[char.lower()] = len(obstacles), len(row)
                    row.append(self.FREE)
                elif 'a' <= char <= 'z':
                    agents[char.lower()] = len(obstacles), len(row)
                    row.append(self.FREE)
                else:
                    raise KeyError(f"Unsupported symbol '{char}' at line {idx}")
            if row:
                if obstacles:
                    if len(obstacles[-1]) != len(row):
                        raise ValueError(f"Row {idx} has {len(row)} columns, expected {len(obstacles[-1])}")
                obstacles.append(row)
        return obstacles, agents, targets

    def __init__(self, name: str, grid_str: str = None, agents_positions: list = None, agents_targets: list = None):
        self.name = name
        self.grid_str = grid_str
        self.agents_positions = agents_positions
        self.agents_targets = agents_targets

        self.obstacles, agents, targets = self.str_to_grid(grid_str)
        self.obstacles = np.array(self.obstacles, dtype=np.int32)

        if agents_positions and agents:
            raise ValueError("Agents positions are already defined in the grid string!")
        if agents_targets and targets:
            raise ValueError("Agents targets are already defined in the grid string!")

        if agents:
            self.agents_xy = []
            for _, (x, y) in sorted(agents.items()):
                self.agents_xy.append([x, y])
        else:
            self.agents_xy = agents_positions

        if targets:
            self.targets_xy = []
            for _, (x, y) in sorted(targets.items()):
                self.targets_xy.append([x, y])
        else:
            self.targets_xy = agents_targets
        if in_registry(name):
            raise ValueError(f"Grid with name {self.name} already registered!")
        check_grid(self.obstacles, self.agents_xy, self.targets_xy)

        register_grid(self)

    def get_obstacles(self):
        return self.obstacles

    def get_agents_xy(self):
        return self.agents_xy

    def get_targets_xy(self):
        return self.targets_xy

    def render(self):
        render_grid(obstacles=self.get_obstacles(), positions_xy=self.get_agents_xy(), targets_xy=self.get_targets_xy())


def register_grid(rg: RegisteredGrid):
    if in_registry(rg.name):
        raise KeyError(f"Grid with name {rg.name} already registered")
    GRID_STR_REGISTRY[rg.name] = rg
