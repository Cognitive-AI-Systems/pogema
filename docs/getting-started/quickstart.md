# Quick Start

## Your First Environment

```python
from pogema import pogema_v0, GridConfig

# Create environment with 4 agents on an 8x8 grid
env = pogema_v0(GridConfig(num_agents=4, size=8, seed=42))

# Reset returns observations and info for each agent
obs, info = env.reset()

# Run episode with random actions
while True:
    obs, reward, terminated, truncated, info = env.step(env.sample_actions())
    if all(terminated) or all(truncated):
        break
```

## Understanding the Output

Each `step()` returns lists of length `num_agents`:

| Return | Type | Description |
|--------|------|-------------|
| `obs` | list of arrays | Per-agent partial observations |
| `reward` | list of float | 1.0 when agent reaches goal |
| `terminated` | list of bool | True when agent finishes |
| `truncated` | list of bool | True when episode time limit hit |
| `info` | list of dict | Metrics dict on final step |

## Recording an Animation

```python exec="on" source="above"
import re  # markdown-exec: hide
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(num_agents=4, size=8, seed=42))
env.enable_animation()
obs, info = env.reset()

while True:
    obs, reward, terminated, truncated, info = env.step(env.sample_actions())
    if all(terminated) or all(truncated):
        break

svg = env.render_animation()._repr_html_()  # markdown-exec: hide
svg = re.sub(r'\n\s+', '\n', svg[svg.index('<svg'):])  # markdown-exec: hide
print(f'<div class="pogema-anim">{svg}</div>')  # markdown-exec: hide
```

## Using the A* Baseline

```python exec="on" source="above"
import re  # markdown-exec: hide
from pogema import pogema_v0, GridConfig, BatchAStarAgent

env = pogema_v0(GridConfig(
    num_agents=4, size=8, seed=42,
    observation_type='POMAPF',  # A* needs xy coordinates
))
env.enable_animation()
agent = BatchAStarAgent()
obs, info = env.reset()

while True:
    actions = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(actions)
    if all(terminated) or all(truncated):
        break

agent.reset_states()
svg = env.render_animation()._repr_html_()  # markdown-exec: hide
svg = re.sub(r'\n\s+', '\n', svg[svg.index('<svg'):])  # markdown-exec: hide
print(f'<div class="pogema-anim">{svg}</div>')  # markdown-exec: hide
```

## Terminal Rendering

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(num_agents=2, size=6, seed=1))
obs, info = env.reset()
env.render()  # Prints ASCII grid to terminal
```
