# A* Baseline Policy

POGEMA includes a built-in A* pathfinding baseline for evaluation and benchmarking. The approach is fully decentralized and works under partial observability by reconstructing the map in memory.

## BatchAStarAgent (Multi-Agent)

```python
from pogema import pogema_v0, GridConfig, BatchAStarAgent

env = pogema_v0(GridConfig(
    num_agents=4,
    size=16,
    observation_type='POMAPF',  # Required — A* needs coordinates
    seed=42,
))
agent = BatchAStarAgent()
obs, info = env.reset()

while True:
    actions = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(actions)
    if all(terminated) or all(truncated):
        break

agent.reset_states()  # Call between episodes
print(info[0]['metrics'])
```

## AStarAgent (Single-Agent)

<!--pytest-codeblocks:skip-->
```python
from pogema import AStarAgent

agent = AStarAgent()
action = agent.act(obs_dict)  # Single observation dict
agent.clear_state()           # Call between episodes
```

## How It Works

1. Each agent maintains a `GridMemory` — a sparse map built from observations
2. At each step, the agent updates its memory with newly observed obstacles
3. A* search finds the shortest path from current position to target
4. If the path is blocked or the agent is stuck, it takes a random action
5. Each agent plans independently (no communication)

## Observation Requirements

A* agents require `observation_type='POMAPF'` or `'MAPF'`, which provides:

- `'xy'`: Agent's current position
- `'target_xy'`: Target position
- `'obstacles'`: Local obstacle map
- `'agents'`: Local agent positions
