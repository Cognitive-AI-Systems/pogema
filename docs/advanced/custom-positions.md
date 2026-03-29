# Custom Agent and Target Positions

## Fixed Positions

Set exact starting and goal positions:

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(
    size=8,
    agents_xy=[[1, 1], [6, 6]],
    targets_xy=[[6, 6], [1, 1]],
    num_agents=2,
    seed=42,
))
```

## Sampling from Pools

Define valid regions for agent/target placement:

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(
    size=8,
    possible_agents_xy=[[1,1], [1,2], [2,1], [2,2]],
    possible_targets_xy=[[6,6], [6,7], [7,6], [7,7]],
    num_agents=2,
    seed=42,
))
```

Each reset samples new positions from the pools.

## Lifelong Target Sequences

For `on_target='restart'`, provide sequences of targets:

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(
    on_target='restart',
    agents_xy=[[1, 1]],
    targets_xy=[[[3, 3], [5, 5], [7, 7]]],  # Sequence per agent
    num_agents=1,
))
```
