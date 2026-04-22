# Environment API

## Core Methods

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(num_agents=2, size=8, seed=42))

# Reset environment
obs, info = env.reset(seed=None)

# Take a step (actions: list of ints, length = num_agents)
obs, reward, terminated, truncated, info = env.step(env.sample_actions())

# Sample random valid actions
actions = env.sample_actions()  # np.ndarray of shape (num_agents,)

# Get number of agents
n = env.get_num_agents()

# Render to terminal
env.render()
```

## State Access Methods

<!--pytest-codeblocks:cont-->
```python
# Agent and target positions (list of [x, y])
agents = env.get_agents_xy()
targets = env.get_targets_xy()

# Positions relative to initial placement
agents_rel = env.get_agents_xy_relative()
targets_rel = env.get_targets_xy_relative()

# Obstacle grid (numpy array, 1 = obstacle)
obstacles = env.get_obstacles()

# Full state as array or dict
state = env.get_state()
state_dict = env.get_state(as_dict=True)

# Per-agent goal status from last step
was_on_goal = env.was_on_goal  # list of bool
```

## Spaces

<!--pytest-codeblocks:cont-->
```python
env.observation_space  # gymnasium.spaces.Box or Dict
env.action_space       # gymnasium.spaces.Discrete(5)
```

## Method Parameters

### `reset()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | int \| None | Override environment seed for this reset |
| `options` | dict \| None | Additional options (unused currently) |

### `step()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `action` | list[int] | One action per agent (0=wait, 1=up, 2=down, 3=left, 4=right) |

### `get_agents_xy()` / `get_targets_xy()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `only_active` | bool | False | Only return active (non-finished) agents |
| `ignore_borders` | bool | False | Return positions without border offset |
