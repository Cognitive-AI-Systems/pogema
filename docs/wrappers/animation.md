# Animation

POGEMA can record episodes as SVG animations. The `AnimationWrapper` is included by default but inactive — it adds zero overhead until enabled.

## Basic Usage

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

## AnimationConfig

```python
from pogema import AnimationConfig

config = AnimationConfig(
    directory='renders/',              # Output directory
    show_agents=True,                  # Render agents on grid
    egocentric_idx=None,               # Follow specific agent (int or None)
    static_frame_idx=None,             # Render single frame instead of animation
    show_grid_lines=True,              # Show grid lines
    save_every_idx_episode=1,          # Save every Nth episode
)
```

## Passing Config

```python
from pogema import pogema_v0, GridConfig, AnimationConfig

env = pogema_v0(GridConfig(num_agents=4, size=8, seed=42))

# At enable time
env.enable_animation(animation_config=AnimationConfig(egocentric_idx=0))
obs, info = env.reset()

# Or at save time (after running an episode)
while True:
    obs, reward, terminated, truncated, info = env.step(env.sample_actions())
    if all(terminated) or all(truncated):
        break

env.save_animation('render.svg', animation_config=AnimationConfig(show_grid_lines=False))
```

## Egocentric View

Follow a specific agent's perspective:

```python exec="on" source="above"
import re  # markdown-exec: hide
from pogema import pogema_v0, GridConfig, AnimationConfig

env = pogema_v0(GridConfig(num_agents=4, size=8, seed=42, obs_radius=3))
env.enable_animation()
obs, info = env.reset()

while True:
    obs, reward, terminated, truncated, info = env.step(env.sample_actions())
    if all(terminated) or all(truncated):
        break

svg = env.render_animation(AnimationConfig(egocentric_idx=0))._repr_html_()  # markdown-exec: hide
svg = re.sub(r'\n\s+', '\n', svg[svg.index('<svg'):])  # markdown-exec: hide
print(f'<div class="pogema-anim">{svg}</div>')  # markdown-exec: hide
```

## Static Frame

Render a single timestep instead of an animation:

```python exec="on" source="above"
import re  # markdown-exec: hide
from pogema import pogema_v0, GridConfig, AnimationConfig

env = pogema_v0(GridConfig(num_agents=4, size=8, seed=42))
env.enable_animation()
obs, info = env.reset()

while True:
    obs, reward, terminated, truncated, info = env.step(env.sample_actions())
    if all(terminated) or all(truncated):
        break

svg = env.render_animation(AnimationConfig(static_frame_idx=0))._repr_html_()  # markdown-exec: hide
svg = re.sub(r'\n\s+', '\n', svg[svg.index('<svg'):])  # markdown-exec: hide
print(f'<div class="pogema-anim">{svg}</div>')  # markdown-exec: hide
```

## A* Baseline

See agents solving the task optimally:

```python exec="on" source="above"
import re  # markdown-exec: hide
from pogema import pogema_v0, GridConfig, BatchAStarAgent

env = pogema_v0(GridConfig(
    num_agents=4, size=8, seed=42,
    observation_type='POMAPF',
))
env.enable_animation()
agent = BatchAStarAgent()
obs, info = env.reset()

while True:
    obs, reward, terminated, truncated, info = env.step(agent.act(obs))
    if all(terminated) or all(truncated):
        break

agent.reset_states()
svg = env.render_animation()._repr_html_()  # markdown-exec: hide
svg = re.sub(r'\n\s+', '\n', svg[svg.index('<svg'):])  # markdown-exec: hide
print(f'<div class="pogema-anim">{svg}</div>')  # markdown-exec: hide
```

## Enable / Disable

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(num_agents=2, size=8, seed=42))
env.enable_animation()          # Start recording
assert env.animation_is_active  # Check status (bool)
env.disable_animation()         # Stop recording (zero overhead resumes)
```
