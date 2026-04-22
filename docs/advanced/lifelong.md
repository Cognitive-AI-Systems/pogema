# Lifelong MAPF

In lifelong mode, agents continuously receive new goals. This models real-world scenarios like warehouse robots.

## Setup

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(
    on_target='restart',
    num_agents=8,
    size=16,
    max_episode_steps=256,
    seed=42,
))
```

## Behavior

1. When an agent reaches its goal, a new goal is generated in the same connected component
2. The agent immediately begins navigating to the new goal
3. The episode runs for the full `max_episode_steps`
4. Performance is measured by throughput

## Metrics

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(on_target='restart', num_agents=4, size=8, max_episode_steps=64, seed=42))
obs, info = env.reset()

while True:
    obs, reward, terminated, truncated, info = env.step(env.sample_actions())
    if all(terminated) or all(truncated):
        break

throughput = info[0]['metrics']['avg_throughput']
# = (total goals reached across all agents) / max_episode_steps
```

## Accessing Current Targets

<!--pytest-codeblocks:cont-->
```python
# Get current targets (changes as agents reach goals)
targets = env.get_targets_xy()
```

## With A* Baseline

```python
from pogema import pogema_v0, GridConfig, BatchAStarAgent

env = pogema_v0(GridConfig(
    on_target='restart',
    num_agents=4,
    size=16,
    max_episode_steps=256,
    observation_type='POMAPF',
    seed=42,
))
agent = BatchAStarAgent()
obs, info = env.reset()

while True:
    actions = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(actions)
    if all(terminated) or all(truncated):
        break

print(f"Throughput: {info[0]['metrics']['avg_throughput']:.3f}")
```
