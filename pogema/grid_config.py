import sys
from typing import Optional, Union
from pydantic import BaseModel, validator

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class GridConfig(BaseModel, ):
    FREE: Literal[0] = 0
    OBSTACLE: Literal[1] = 1
    MOVES: list = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], ]
    pogema_type: Literal['disappearing', 'life_long'] = 'disappearing'
    seed: Optional[int] = None
    size: int = 8
    density: float = 0.3
    num_agents: int = 1
    obs_radius: int = 5
    agents_xy: Optional[list] = None
    targets_xy: Optional[list] = None

    map: Union[list, str] = None

    disappear_on_goal: Literal[True] = True
    empty_outside: bool = True

    map_name: str = None

    integration: Literal['SampleFactory', 'PyMARL', 'rllib', 'gym', 'PettingZoo'] = None
    max_episode_steps: int = 64
    steps_before_renew_target: Optional[int] = None

    @validator('seed')
    def seed_initialization(cls, v):
        assert v is None or (0 <= v < sys.maxsize), "seed must be in [0, " + str(sys.maxsize) + ']'
        return v

    @validator('size')
    def size_restrictions(cls, v):
        assert 2 <= v <= 1024, "size must be in [2, 1024]"
        return v

    @validator('density')
    def density_restrictions(cls, v):
        assert 0.0 <= v <= 1, "density must be in [0, 1]"
        return v

    @validator('num_agents')
    def num_agents_must_be_positive(cls, v):
        assert 1 <= v <= 10000, "num_agents must be in [1, 10000]"
        return v

    @validator('obs_radius')
    def obs_radius_must_be_positive(cls, v):
        assert 1 <= v <= 128, "obs_radius must be in [1, 128]"
        return v

    @validator('map', always=True)
    def map_validation(cls, v, values, ):
        if v is None:
            return None
        if isinstance(v, str):
            v, agents_xy, targets_xy = cls.str_map_to_list(v, values['FREE'], values['OBSTACLE'])
            if agents_xy and targets_xy and values['agents_xy'] is not None and values['targets_xy'] is not None:
                raise KeyError("""Can't create task. Please provide agents_xy and targets_xy only ones.
                Either with parameters or with a map.""")
            elif agents_xy and targets_xy:
                values['agents_xy'] = agents_xy
                values['targets_xy'] = targets_xy
                values['num_agents'] = len(agents_xy)
        size = len(v)
        area = 0
        for line in v:
            size = max(size, len(line))
            area += len(line)
        values['size'] = size
        values['density'] = sum([sum(line) for line in v]) / area
        return v

    @validator('agents_xy')
    def agents_xy_validation(cls, v, values):
        if v is not None:
            cls.check_positions(v, values['size'])
            values['num_agents'] = len(v)
        return v

    @validator('targets_xy')
    def targets_xy_validation(cls, v, values):
        if v is not None:
            cls.check_positions(v, values['size'])
            values['num_agents'] = len(v)
        return v

    @staticmethod
    def check_positions(v, size):
        for position in v:
            x, y = position
            if not (0 <= x < size and 0 <= y < size):
                raise IndexError("Position is out of bounds!")

    @staticmethod
    def str_map_to_list(str_map, free, obstacle):
        obstacles = []
        agents = {}
        targets = {}
        for idx, line in enumerate(str_map.split()):
            row = []
            for char in line:
                if char == '.':
                    row.append(free)
                elif char == '#':
                    row.append(obstacle)
                elif 'A' <= char <= 'Z':
                    targets[char.lower()] = len(obstacles), len(row)
                    row.append(free)
                elif 'a' <= char <= 'z':
                    agents[char.lower()] = len(obstacles), len(row)
                    row.append(free)
                else:
                    raise KeyError(f"Unsupported symbol '{char}' at line {idx}")
            if row:
                if obstacles:
                    assert len(obstacles[-1]) == len(row), f"Wrong string size for row {idx};"
                obstacles.append(row)

        targets_xy = []
        agents_xy = []
        for _, (x, y) in sorted(agents.items()):
            agents_xy.append([x, y])
        for _, (x, y) in sorted(targets.items()):
            targets_xy.append([x, y])

        assert len(targets_xy) == len(agents_xy)
        return obstacles, agents_xy, targets_xy
