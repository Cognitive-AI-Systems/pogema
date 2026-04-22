# Collision Systems

Collisions occur when two agents attempt to occupy the same cell. POGEMA supports three modes.

## `block_both` (Conservative)

Both agents remain in their previous positions. No agent moves into the contested cell.

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(collision_system='block_both', num_agents=4))
```

**Use case**: Robotics-inspired settings where physical collisions must be avoided.

## `priority` (Default)

The agent with the higher index moves; the other stays. This creates an implicit priority ordering.

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(collision_system='priority', num_agents=4))
```

**Use case**: Asymmetric settings or when a deterministic resolution is needed.

## `soft`

Agents can freely overlap. No collision penalties or blocking.

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(collision_system='soft', num_agents=4))
```

**Use case**: Simplified settings, large-scale experiments, or when collision avoidance is not the focus.
