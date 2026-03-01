import sys
from typing import Literal

from pydantic import BaseModel


class AgentsTargetsSizeError(Exception):
    pass


def grid_to_str(grid):
    return '\n'.join(''.join('.' if cell == 0 else '#' for cell in row) for row in grid)


def check_grid(obstacles, agents_xy, targets_xy):
    if bool(agents_xy) != bool(targets_xy):
        raise AgentsTargetsSizeError("Agents and targets must be defined together/undefined together!")

    if not agents_xy or not targets_xy:
        return

    if len(agents_xy) != len(targets_xy):
        raise IndexError("Can't create task. Please provide agents_xy and targets_xy of the same size.")

    # check overlapping of agents
    for i in range(len(agents_xy)):
        for j in range(i + 1, len(agents_xy)):
            if agents_xy[i] == agents_xy[j]:
                raise ValueError(f"Agents can't overlap! {agents_xy[i]} is in both {i} and {j} position.")

    for start_xy, finish_xy in zip(agents_xy, targets_xy):
        s_x, s_y = start_xy
        if obstacles[s_x, s_y]:
            raise ValueError(f'Cell ({s_x}, {s_y}) is occupied by an obstacle.')
        f_x, f_y = finish_xy
        if obstacles[f_x, f_y]:
            raise ValueError(f'Cell ({f_x}, {f_y}) is occupied by an obstacle.')

    # Connectivity check not implemented: GridLifeLong already warns on unreachable
    # targets, and adding a mandatory check to Grid would be a breaking change for
    # users who intentionally place agents/targets in disconnected components.


def render_grid(obstacles, positions_xy=None, targets_xy=None, is_active=None, mode='human'):
    if positions_xy is None:
        positions_xy = []
    if targets_xy is None:
        targets_xy = []
    if is_active is None:
        if positions_xy:
            is_active = [True] * len(positions_xy)
        else:
            is_active = []
    import string
    from contextlib import closing
    from io import StringIO

    from gymnasium import utils as gym_utils

    outfile = StringIO() if mode == 'ansi' else sys.stdout
    chars = string.digits + string.ascii_letters + string.punctuation
    positions_map = {(x, y): id_ for id_, (x, y) in enumerate(positions_xy) if is_active[id_]}
    finishes_map = {(x, y): id_ for id_, (x, y) in enumerate(targets_xy) if is_active[id_]}
    for line_index, line in enumerate(obstacles):
        out = ''
        for cell_index, cell in enumerate(line):
            if cell == CommonSettings().FREE:
                agent_id = positions_map.get((line_index, cell_index), None)
                finish_id = finishes_map.get((line_index, cell_index), None)

                if agent_id is not None:
                    out += str(gym_utils.colorize(' ' + chars[agent_id % len(chars)] + ' ', color='red', bold=True,
                                                  highlight=False))
                elif finish_id is not None:
                    out += str(
                        gym_utils.colorize('|' + chars[finish_id % len(chars)] + '|', 'white', highlight=False))
                else:
                    out += str(gym_utils.colorize(' . ', 'white', highlight=False))
            else:
                out += str(gym_utils.colorize('   ', 'cyan', bold=False, highlight=True))
        out += '\n'
        outfile.write(out)

    if mode != 'human':
        with closing(outfile):
            return outfile.getvalue()


class CommonSettings(BaseModel):
    MOVES: list = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], ]
    FREE: Literal[0] = 0
    OBSTACLE: Literal[1] = 1
    empty_outside: bool = True
