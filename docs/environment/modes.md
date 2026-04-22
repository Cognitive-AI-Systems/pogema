# Environment Modes

The `on_target` parameter controls agent behavior upon reaching its goal, defining the MAPF task variant.

## `'finish'` — Finish and Disappear (default)

Agent disappears from the grid upon reaching its goal. The vacated cell becomes free, so other agents can pass through it. Each agent terminates independently.

This is the default mode and a simplified variant that removes the coordination challenge of classical MAPF — agents don't need to worry about blocking others once they've finished.

```python exec="on" source="above"
import re  # markdown-exec: hide
from pogema import pogema_v0, GridConfig, BatchAStarAgent

env = pogema_v0(GridConfig(
    on_target='finish',
    num_agents=4,
    size=8,
    density=0.3,
    seed=42,
    observation_type='POMAPF',
    max_episode_steps=64,
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

Notice how agents vanish from the grid the moment they reach their targets. Once gone, other agents can freely pass through those cells.

**Metrics**: CSR, ISR, EpLength, SumOfCosts, Makespan

## `'nothing'` — Classical MAPF

The standard multi-agent pathfinding formulation. Agents remain on the grid after reaching their goals and continue to act as obstacles for others. The episode succeeds only when **all agents occupy their goals simultaneously**.

This is the most challenging mode — agents must coordinate not just their paths but also their timing, since an agent sitting on its goal can still block others.

```python exec="on" source="above"
import re  # markdown-exec: hide
from pogema import pogema_v0, GridConfig, BatchAStarAgent

env = pogema_v0(GridConfig(
    on_target='nothing',
    num_agents=4,
    size=8,
    density=0.3,
    seed=42,
    observation_type='POMAPF',
    max_episode_steps=64,
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

Here agents stay on the grid even after reaching their goals — they can still block others. The episode only ends when all agents are on their targets at the same time.

**Metrics**: CSR, ISR, EpLength, SumOfCosts, Makespan

## `'restart'` — Lifelong MAPF

Agents continuously receive new goals upon reaching their current ones. The episode always runs for the full `max_episode_steps` horizon — no agent ever terminates. Performance is measured by how many goals agents collectively reach.

This models real-world scenarios like warehouse robots that perpetually pick up and deliver items.

```python exec="on" source="above"
import re  # markdown-exec: hide
from pogema import pogema_v0, GridConfig, BatchAStarAgent

env = pogema_v0(GridConfig(
    on_target='restart',
    num_agents=4,
    size=8,
    density=0.3,
    seed=42,
    observation_type='POMAPF',
    max_episode_steps=64,
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

Watch how agents receive new targets as soon as they reach their current ones — their target markers jump to new positions. The episode runs for the full time horizon.

**Metrics**: LifeLongAverageThroughput (total goals reached / max_steps)

See [Lifelong MAPF](../advanced/lifelong.md) for more details on lifelong mode.
