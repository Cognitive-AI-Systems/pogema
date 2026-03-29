# PettingZoo

POGEMA implements the PettingZoo Parallel API for multi-agent RL frameworks.

## Basic Usage

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(integration='PettingZoo', num_agents=4, size=8, seed=42))
obs, infos = env.reset()
# obs: {'player_0': array, 'player_1': array, ...}

# Step with dict of actions
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, rewards, terminated, truncated, infos = env.step(actions)

# Properties
env.agents              # List of currently active agent names
env.possible_agents     # List of all agent names
env.observation_space('player_0')
env.action_space('player_0')
```

## Agent Naming

Agents are named `'player_0'`, `'player_1'`, ..., `'player_{n-1}'`.

## Full Episode Loop

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(integration='PettingZoo', num_agents=4, size=8, seed=42))
obs, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminated, truncated, infos = env.step(actions)
```
