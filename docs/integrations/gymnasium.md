# Gymnasium (Single-Agent)

Wraps POGEMA as a standard single-agent Gymnasium environment. Only works with `num_agents=1`.

## Basic Usage

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(
    integration='gymnasium',
    num_agents=1,
    size=8,
    density=0.3,
))

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## With Stable-Baselines3

<!--pytest-codeblocks:skip-->
```python
from stable_baselines3 import PPO
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(integration='gymnasium', num_agents=1, size=8))
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100_000)
```

## Key Differences from Multi-Agent API

- `step()` takes a single int action, not a list
- `reset()` returns a single observation, not a list
- `observation_space` and `action_space` are for one agent
- Compatible with any Gymnasium-based RL library
