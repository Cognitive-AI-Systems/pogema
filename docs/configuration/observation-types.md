# Observation Types

## `'default'`

Array of shape `(3, 2R+1, 2R+1)` with three channels:

```
Channel 0: Obstacles (1.0 = obstacle, 0.0 = free, -1.0 = out of bounds)
Channel 1: Other agents (1.0 = agent present)
Channel 2: Target direction (encodes relative goal position)
```

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(observation_type='default', obs_radius=5))
obs, info = env.reset()
assert obs[0].shape == (3, 11, 11)
```

Best for: Neural network policies (CNN input).

## `'POMAPF'`

Dictionary observation with explicit coordinates:

<!--pytest-codeblocks:skip-->
```python
{
    'obstacles': np.array(...),    # (2R+1, 2R+1) float32
    'agents': np.array(...),       # (2R+1, 2R+1) float32
    'xy': [x, y],                  # Agent position (int)
    'target_xy': [tx, ty],         # Target position (int)
}
```

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(observation_type='POMAPF'))
obs, info = env.reset()
assert 'xy' in obs[0] and 'target_xy' in obs[0]
```

Best for: Classical planners (like A*) used in decentralized mode, that need coordinates.

## `'MAPF'`

Extends POMAPF with full global state:

<!--pytest-codeblocks:skip-->
```python
{
    'obstacles': ...,              # Local (same as POMAPF)
    'agents': ...,
    'xy': [x, y],
    'target_xy': [tx, ty],
    'global_obstacles': np.array(...),  # Full map
    'global_xy': [x, y],               # Same as xy
    'global_target_xy': [tx, ty],      # Same as target_xy
}
```

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(observation_type='MAPF'))
obs, info = env.reset()
assert 'global_obstacles' in obs[0]
```

Best for: approaches that introduce their own observation (like MAPF-GPT) or centralized planners.
