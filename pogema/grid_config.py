import sys
from typing import Optional, Union
from pydantic import validator, root_validator
from pydantic import BaseModel, model_validator

from pogema.utils import CommonSettings

from typing_extensions import Literal

import sys
from typing import Optional, Union
from pydantic import validator, model_validator
from pogema.utils import CommonSettings
from typing_extensions import Literal


class GridConfig(CommonSettings):
    on_target: Literal['finish', 'nothing', 'restart'] = 'finish'
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size: int = 8
    density: float = 0.3
    obs_radius: int = 5
    agents_xy: Optional[list] = None
    targets_xy: Optional[list] = None
    num_agents: Optional[int] = None
    possible_agents_xy: Optional[list] = None
    possible_targets_xy: Optional[list] = None
    collision_system: Literal['block_both', 'priority', 'soft'] = 'priority'
    persistent: bool = False
    observation_type: Literal['POMAPF', 'MAPF', 'default'] = 'default'
    map: Optional[Union[list, str]] = None
    map_name: Optional[str] = None
    integration: Optional[Literal['SampleFactory', 'PyMARL', 'rllib', 'gymnasium', 'PettingZoo']] = None
    max_episode_steps: int = 64
    auto_reset: Optional[bool] = None

    @model_validator(mode='after')
    def validate_dimensions_and_positions(cls, model):
        # Use getattr for safe access, with default fallback
        width = getattr(model, 'width', None)
        height = getattr(model, 'height', None)
        size = getattr(model, 'size', 8)

        width_provided = width is not None and width > 0
        height_provided = height is not None and height > 0

        if width_provided and not height_provided:
            raise ValueError("Invalid dimension configuration: width provided but height missing.")
        if height_provided and not width_provided:
            raise ValueError("Invalid dimension configuration: height provided but width missing.")

        if not width_provided and not height_provided:
            fallback_size = size if size >= 2 else 8
            width = fallback_size
            height = fallback_size

        if width <= 0:
            width = 8
        if height <= 0:
            height = 8

        size = max(width, height, 2)

        setattr(model, 'width', width)
        setattr(model, 'height', height)
        setattr(model, 'size', size)

        if not (1 <= width <= 4096):
            raise ValueError(f"width must be in [1, 4096], got {width}")
        if not (1 <= height <= 4096):
            raise ValueError(f"height must be in [1, 4096], got {height}")
        if not (2 <= size <= 4096):
            raise ValueError(f"size must be in [2, 4096], got {size}")

        # Validate positions
        agents_xy = getattr(model, 'agents_xy', None)
        targets_xy = getattr(model, 'targets_xy', None)

        if agents_xy is not None:
            cls.check_positions(agents_xy, width, height)

        if targets_xy is not None:
            first_element = targets_xy[0]
            if isinstance(first_element[0], (list, tuple)):
                for agent_goals in targets_xy:
                    cls.check_positions(agent_goals, width, height)
            else:
                cls.check_positions(targets_xy, width, height)

        return model

    @validator('seed')
    def seed_initialization(cls, v):
        assert v is None or (0 <= v < sys.maxsize), f"seed must be in [0, {sys.maxsize}]"
        return v

    @staticmethod
    def _validate_dimension(v, field_name):
        if v is not None:
            if field_name == 'size':
                assert 2 <= v <= 4096, f"{field_name} must be in [2, 4096]"
            else:
                assert 1 <= v <= 4096, f"{field_name} must be in [1, 4096]"
        return v

    @validator('size')
    def size_restrictions(cls, v):
        return cls._validate_dimension(v, 'size')

    @validator('width')
    def width_restrictions(cls, v):
        return cls._validate_dimension(v, 'width')

    @validator('height')
    def height_restrictions(cls, v):
        return cls._validate_dimension(v, 'height')

    @validator('density')
    def density_restrictions(cls, v):
        assert 0.0 <= v <= 1, "density must be in [0, 1]"
        return v

    @validator('agents_xy')
    def agents_xy_validation(cls, v):
        if v is not None:
            if not isinstance(v, (list, tuple)):
                raise ValueError("agents_xy must be a list")
            for position in v:
                if not isinstance(position, (list, tuple)) or len(position) != 2:
                    raise ValueError("Position must be a list/tuple of length 2")
                if not all(isinstance(coord, int) for coord in position):
                    raise ValueError("Position coordinates must be integers")
        return v

    @validator('targets_xy')
    def targets_xy_validation(cls, v, values):
        if v is not None:
            if not v or not isinstance(v, (list, tuple)):
                raise ValueError("targets_xy must be a list")

            first_element = v[0]
            if not isinstance(first_element, (list, tuple)):
                raise ValueError("Invalid targets_xy format")

            if isinstance(first_element[0], (list, tuple)):
                for agent_goals in v:
                    if not isinstance(agent_goals, (list, tuple)) or len(agent_goals) < 2:
                        raise ValueError("Each agent must have at least two goals in the sequence")
                    for position in agent_goals:
                        if not isinstance(position, (list, tuple)) or len(position) != 2:
                            raise ValueError("Position must be a list/tuple of length 2")
                        if not all(isinstance(coord, int) for coord in position):
                            raise ValueError("Position coordinates must be integers")
            else:
                on_target = values.get('on_target', 'finish')
                if on_target == 'restart':
                    raise ValueError(
                        "on_target='restart' requires goal sequences, not single goals. "
                        "Use format: targets_xy: [[[x1,y1],[x2,y2]], [[x3,y3],[x4,y4]]]"
                    )
                for position in v:
                    if not isinstance(position, (list, tuple)) or len(position) != 2:
                        raise ValueError("Position must be a list/tuple of length 2")
                    if not all(isinstance(coord, int) for coord in position):
                        raise ValueError("Position coordinates must be integers")
        return v

    @staticmethod
    def check_positions(v, width, height):
        for position in v:
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                raise ValueError("Position must be a list/tuple of length 2")
            x, y = position
            if not isinstance(x, int) or not isinstance(y, int):
                raise ValueError("Position coordinates must be integers")
            if not (0 <= x < height and 0 <= y < width):
                raise IndexError(
                    f"Position is out of bounds! {position} is not in [{0}, {height}] x [{0}, {width}]"
                )

    @staticmethod
    def check_positions(v, width, height):
        for position in v:
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                raise ValueError("Position must be a list/tuple of length 2")
            x, y = position
            if not isinstance(x, int) or not isinstance(y, int):
                raise ValueError("Position coordinates must be integers")
            if not (0 <= x < height and 0 <= y < width):
                raise IndexError(f"Position is out of bounds! {position} is not in [{0}, {height}] x [{0}, {width}]")


    @validator('num_agents', always=True)
    def num_agents_must_be_positive(cls, v, values):
        if v is None:
            if values['agents_xy']:
                v = len(values['agents_xy'])
            else:
                v = 1
        assert 1 <= v <= 10000000, "num_agents must be in [1, 10000000]"
        return v

    @validator('obs_radius')
    def obs_radius_must_be_positive(cls, v):
        assert 1 <= v <= 128, "obs_radius must be in [1, 128]"
        return v

    @validator('map', always=True)
    def map_validation(cls, v, values):
        if v is None:
            return None
        if isinstance(v, str):
            v, agents_xy, targets_xy, possible_agents_xy, possible_targets_xy = cls.str_map_to_list(v, values['FREE'],
                                                                                                    values['OBSTACLE'])
            if agents_xy and targets_xy and values.get('agents_xy') is not None and values.get(
                    'targets_xy') is not None:
                raise KeyError("""Can't create task. Please provide agents_xy and targets_xy only once.
                Either with parameters or with a map.""")
            if (agents_xy or targets_xy) and (possible_agents_xy or possible_targets_xy):
                raise KeyError("""Can't create task. Mark either possible locations or precise ones.""")
            elif agents_xy and targets_xy:
                values['agents_xy'] = agents_xy
                values['targets_xy'] = targets_xy
                values['num_agents'] = len(agents_xy)
            elif (values.get('agents_xy') is None or values.get(
                    'targets_xy') is None) and possible_agents_xy and possible_targets_xy:
                values['possible_agents_xy'] = possible_agents_xy
                values['possible_targets_xy'] = possible_targets_xy

        height = len(v)
        width = 0
        area = 0
        for line in v:
            width = max(width, len(line))
            area += len(line)

        values['size'] = max(width, height)
        values['width'] = width
        values['height'] = height
        values['density'] = sum([sum(line) for line in v]) / area

        return v

    # @validator('possible_agents_xy')
    # def possible_agents_xy_validation(cls, v):
    #     return v
    #
    # @validator('possible_targets_xy')
    # def possible_targets_xy_validation(cls, v):
    #     return v

    @staticmethod
    def str_map_to_list(str_map, free, obstacle):
        obstacles = []
        agents = {}
        targets = {}
        possible_agents_xy = []
        possible_targets_xy = []
        special_chars = {'@', '$', '!'}

        for row_idx, line in enumerate(str_map.split()):
            row = []
            for col_idx, char in enumerate(line):
                position = (row_idx, col_idx)

                if char == '.':
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                elif char == '#':
                    row.append(obstacle)
                elif char in special_chars:
                    row.append(free)
                    if char == '@':
                        possible_agents_xy.append(position)
                    elif char == '$':
                        possible_targets_xy.append(position)
                elif 'A' <= char <= 'Z':
                    targets[char.lower()] = position
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                elif 'a' <= char <= 'z':
                    agents[char.lower()] = position
                    row.append(free)
                    possible_agents_xy.append(position)
                    possible_targets_xy.append(position)
                else:
                    raise KeyError(f"Unsupported symbol '{char}' at line {row_idx}")

            if row:
                assert len(obstacles[-1]) == len(row) if obstacles else True, f"Wrong string size for row {row_idx};"
                obstacles.append(row)

        agents_xy = [[x, y] for _, (x, y) in sorted(agents.items())]
        targets_xy = [[x, y] for _, (x, y) in sorted(targets.items())]

        assert len(targets_xy) == len(agents_xy), "Mismatch in number of agents and targets."

        if not any(char in special_chars for char in str_map):
            possible_agents_xy, possible_targets_xy = None, None

        return obstacles, agents_xy, targets_xy, possible_agents_xy, possible_targets_xy

    def update_config(self, **kwargs):
        current_values = self.dict()

        if 'size' in kwargs:
            current_values.pop('width', None)
            current_values.pop('height', None)
        elif 'width' in kwargs or 'height' in kwargs:
            current_values.pop('size', None)
        current_values.update(kwargs)
        new_instance = GridConfig(**current_values)

        for field_name, field_value in new_instance.__dict__.items():
            setattr(self, field_name, field_value)
