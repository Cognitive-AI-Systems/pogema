import sys
from typing import Literal

from pydantic import Field, field_validator, model_validator

from pogema.utils import CommonSettings


class GridConfig(CommonSettings):
    on_target: Literal['finish', 'nothing', 'restart'] = Field(
        'finish',
        description="Behavior when agent reaches target: "
                    "'finish' (agent disappears), 'nothing' (agent stays, all must reach simultaneously), "
                    "'restart' (agent gets new target, lifelong MAPF).",
    )
    seed: int | None = Field(None, description="Random seed for reproducibility.")
    width: int | None = Field(None, description="Grid width. Must be paired with height.")
    height: int | None = Field(None, description="Grid height. Must be paired with width.")
    size: int = Field(8, description="Grid size (used as both width and height when they are not set).")
    density: float = Field(0.3, description="Obstacle density in [0, 1] for random map generation.")
    obs_radius: int = Field(
        5, description="Observation radius. Each agent sees a (2*obs_radius+1) x (2*obs_radius+1) window.",
    )
    agents_xy: list[list[int]] | None = Field(
        None, description="Fixed agent start positions as [[row, col], ...].",
    )
    targets_xy: list[list[int]] | list[list[list[int]]] | None = Field(
        None,
        description="Target positions: [[row, col], ...] for single targets, "
                    "or [[[r1,c1],[r2,c2],...], ...] for goal sequences (lifelong MAPF).",
    )
    num_agents: int | None = Field(None, description="Number of agents. Auto-inferred from agents_xy or map if not set.")
    possible_agents_xy: list[list[int]] | None = Field(
        None, description="Pool of positions to randomly sample agent starts from.",
    )
    possible_targets_xy: list[list[int]] | None = Field(
        None, description="Pool of positions to randomly sample targets from.",
    )
    collision_system: Literal['block_both', 'priority', 'soft'] = Field(
        'priority',
        description="Collision resolution: 'priority' (higher index wins), "
                    "'block_both' (both agents stay), 'soft' (vertex and edge collision avoidance).",
    )
    persistent: bool = Field(False, description="Deprecated. Use env.enable_animation() instead.")
    observation_type: Literal['POMAPF', 'MAPF', 'default'] = Field(
        'default',
        description="Observation format: 'default' (3-channel array), "
                    "'POMAPF' (dict with obstacles/agents/xy/target_xy), "
                    "'MAPF' (POMAPF + global state).",
    )
    map: list[list[int]] | str | None = Field(
        None,
        description="Custom map as a 2D list of 0/1 (free/obstacle) or a string with special characters "
                    "(. # @ $ ! a-z A-Z).",
    )
    map_name: str | None = Field(None, description="Name of a registered map from the grid registry.")
    integration: Literal['SampleFactory', 'gymnasium', 'PettingZoo'] | None = Field(
        None, description="Framework integration: None (raw multi-agent), 'gymnasium', 'PettingZoo', or 'SampleFactory'.",
    )
    max_episode_steps: int = Field(64, description="Maximum number of steps per episode before truncation.")
    auto_reset: bool | None = Field(None, description="Auto-reset on episode end (SampleFactory only).")

    @model_validator(mode='before')
    @classmethod
    def process_map_and_defaults(cls, data):
        if isinstance(data, dict):
            # Process string map into list and extract agents/targets
            map_val = data.get('map')
            if map_val is not None and isinstance(map_val, str):
                free = CommonSettings().FREE
                obstacle = CommonSettings().OBSTACLE
                map_val, agents_xy, targets_xy, possible_agents_xy, possible_targets_xy = cls.str_map_to_list(
                    map_val, free, obstacle
                )
                if agents_xy and targets_xy and data.get('agents_xy') is not None and data.get(
                        'targets_xy') is not None:
                    raise ValueError("Can't create task. Please provide agents_xy and targets_xy only once: "
                                     "either with parameters or with a map.")
                if (agents_xy or targets_xy) and (possible_agents_xy or possible_targets_xy):
                    raise ValueError("Can't create task. Mark either possible locations or precise ones.")
                elif agents_xy and targets_xy:
                    data['agents_xy'] = agents_xy
                    data['targets_xy'] = targets_xy
                    data['num_agents'] = len(agents_xy)
                elif (data.get('agents_xy') is None or data.get(
                        'targets_xy') is None) and possible_agents_xy and possible_targets_xy:
                    data['possible_agents_xy'] = possible_agents_xy
                    data['possible_targets_xy'] = possible_targets_xy

                data['map'] = map_val

            # Compute map-derived dimensions
            if map_val is not None and not isinstance(map_val, str):
                height = len(map_val)
                width = 0
                area = 0
                for line in map_val:
                    width = max(width, len(line))
                    area += len(line)
                data['size'] = max(width, height)
                data['width'] = width
                data['height'] = height
                data['density'] = sum([sum(line) for line in map_val]) / area

            # Default num_agents
            if data.get('num_agents') is None:
                if data.get('agents_xy'):
                    data['num_agents'] = len(data['agents_xy'])
                else:
                    data['num_agents'] = 1

        return data

    @model_validator(mode='after')
    def validate_dimensions_and_positions(self):
        width = self.width
        height = self.height
        size = self.size

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

        self.width = width
        self.height = height
        self.size = size

        if not (1 <= width <= 8_388_608):
            raise ValueError(f"width must be in [1, 8_388_608], got {width}")
        if not (1 <= height <= 8_388_608):
            raise ValueError(f"height must be in [1, 8_388_608], got {height}")
        if not (2 <= size <= 8_388_608):
            raise ValueError(f"size must be in [2, 8_388_608], got {size}")

        # Validate positions
        agents_xy = self.agents_xy
        targets_xy = self.targets_xy

        if agents_xy is not None:
            self.check_positions(agents_xy, width, height)

        if targets_xy is not None:
            first_element = targets_xy[0]
            if isinstance(first_element[0], (list, tuple)):
                for agent_goals in targets_xy:
                    self.check_positions(agent_goals, width, height)
            else:
                self.check_positions(targets_xy, width, height)

        return self

    @field_validator('seed')
    @classmethod
    def seed_initialization(cls, v):
        if v is not None and not (0 <= v < sys.maxsize):
            raise ValueError(f"seed must be in [0, {sys.maxsize})")
        return v

    @staticmethod
    def _validate_dimension(v, field_name):
        if v is not None:
            if field_name == 'size':
                if not (2 <= v <= 8_388_608):
                    raise ValueError(f"{field_name} must be in [2, 8_388_608]")
            else:
                if not (1 <= v <= 8_388_608):
                    raise ValueError(f"{field_name} must be in [1, 8_388_608]")
        return v

    @field_validator('size')
    @classmethod
    def size_restrictions(cls, v):
        return cls._validate_dimension(v, 'size')

    @field_validator('width')
    @classmethod
    def width_restrictions(cls, v):
        return cls._validate_dimension(v, 'width')

    @field_validator('height')
    @classmethod
    def height_restrictions(cls, v):
        return cls._validate_dimension(v, 'height')

    @field_validator('density')
    @classmethod
    def density_restrictions(cls, v):
        if not (0.0 <= v <= 1):
            raise ValueError("density must be in [0, 1]")
        return v

    @field_validator('agents_xy')
    @classmethod
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

    @field_validator('targets_xy')
    @classmethod
    def targets_xy_validation(cls, v, info):
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
                on_target = info.data.get('on_target', 'finish')
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
                raise IndexError(f"Position {position} is out of bounds: row must be in [0, {height}), col in [0, {width})")

    @field_validator('num_agents')
    @classmethod
    def num_agents_must_be_positive(cls, v):
        if not (1 <= v <= 10_000_000):
            raise ValueError("num_agents must be in [1, 10_000_000]")
        return v

    @field_validator('obs_radius')
    @classmethod
    def obs_radius_must_be_positive(cls, v):
        if not (1 <= v <= 128):
            raise ValueError("obs_radius must be in [1, 128]")
        return v

    @field_validator('map')
    @classmethod
    def map_validation(cls, v):
        if v is None:
            return None
        # String maps are already processed in model_validator(mode='before')
        # At this point v should be a list
        return v

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
                    raise ValueError(f"Unsupported symbol '{char}' at line {row_idx}")

            if row:
                if obstacles and len(obstacles[-1]) != len(row):
                    raise ValueError(f"Inconsistent row width at row {row_idx}: expected {len(obstacles[-1])}, got {len(row)}")
                obstacles.append(row)

        agents_xy = [[x, y] for _, (x, y) in sorted(agents.items())]
        targets_xy = [[x, y] for _, (x, y) in sorted(targets.items())]

        if len(targets_xy) != len(agents_xy):
            raise ValueError(f"Mismatch in number of agents ({len(agents_xy)}) and targets ({len(targets_xy)}) in map.")

        if not any(char in special_chars for char in str_map):
            possible_agents_xy, possible_targets_xy = None, None

        return obstacles, agents_xy, targets_xy, possible_agents_xy, possible_targets_xy

    def update_config(self, **kwargs):
        current_values = self.model_dump()

        if 'size' in kwargs:
            current_values.pop('width', None)
            current_values.pop('height', None)
        elif 'width' in kwargs or 'height' in kwargs:
            current_values.pop('size', None)
        current_values.update(kwargs)
        new_instance = GridConfig(**current_values)

        for field_name, field_value in new_instance.__dict__.items():
            setattr(self, field_name, field_value)
