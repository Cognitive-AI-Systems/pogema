# GridConfig Reference

`GridConfig` is a Pydantic model that controls all environment parameters. Pass it to `pogema_v0()` to create an environment.

```python
from pogema import GridConfig

config = GridConfig(
    num_agents=4,
    size=16,
    density=0.3,
    seed=42,
)
```

## Map Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | int | 8 | Square grid dimension (size x size) |
| `width` | int \| None | None | Grid width (overrides `size` for non-square grids) |
| `height` | int \| None | None | Grid height (overrides `size` for non-square grids) |
| `density` | float | 0.3 | Obstacle density, range [0, 1] |
| `map` | str \| list \| None | None | Custom map definition (see [Custom Maps](custom-maps.md)) |
| `map_name` | str \| None | None | Load a registered map by name |

## Agent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_agents` | int \| None | 1 | Number of agents |
| `agents_xy` | list \| None | None | Exact starting positions `[[x1,y1], [x2,y2], ...]` |
| `targets_xy` | list \| None | None | Exact goal positions |
| `possible_agents_xy` | list \| None | None | Pool of valid start positions to sample from |
| `possible_targets_xy` | list \| None | None | Pool of valid goal positions to sample from |

## Environment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `on_target` | str | `'finish'` | Task mode: `'nothing'` (classical MAPF), `'restart'` (lifelong), or `'finish'` (disappear) |
| `collision_system` | str | `'priority'` | Collision mode: `'block_both'`, `'priority'`, or `'soft'` |
| `obs_radius` | int | 5 | Observation radius (field of view = 2R+1) |
| `max_episode_steps` | int | 64 | Maximum steps per episode |
| `seed` | int \| None | None | Random seed (`None` = random each reset) |

## Observation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observation_type` | str | `'default'` | Format: `'default'`, `'POMAPF'`, or `'MAPF'` |

## Integration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `integration` | str \| None | None | Framework: `'PettingZoo'`, `'SampleFactory'`, `'gymnasium'` |
| `auto_reset` | bool \| None | None | Enable auto-reset (SampleFactory only) |


