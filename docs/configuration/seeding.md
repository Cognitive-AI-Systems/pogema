# Seeding

Seeds control random map generation, agent placement, and target assignment. Setting a seed makes experiments fully reproducible.

## Reproducible Environments

Pass `seed` to `GridConfig` for deterministic behavior. Two environments with the same config produce identical episodes:

```python
from pogema import pogema_v0, GridConfig

cfg = GridConfig(seed=42, num_agents=4, size=10, density=0.3)

env1 = pogema_v0(cfg)
obs1, _ = env1.reset()

env2 = pogema_v0(cfg)
obs2, _ = env2.reset()

assert all((o1 == o2).all() for o1, o2 in zip(obs1, obs2))
```

The seed determines obstacle layout, agent start positions, and target positions.

## Different Seeds — Different Environments

Each seed produces a unique obstacle layout, agent placement, and target assignment. Here are three environments generated with different seeds on the same grid size:

=== "seed=42"

    ```python exec="on" source="above"
    import re  # markdown-exec: hide
    from pogema import pogema_v0, GridConfig, BatchAStarAgent

    env = pogema_v0(GridConfig(
        seed=42, num_agents=4, size=8, density=0.3,
        observation_type='POMAPF', max_episode_steps=64,
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

=== "seed=7"

    ```python exec="on" source="above"
    import re  # markdown-exec: hide
    from pogema import pogema_v0, GridConfig, BatchAStarAgent

    env = pogema_v0(GridConfig(
        seed=7, num_agents=4, size=8, density=0.3,
        observation_type='POMAPF', max_episode_steps=64,
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

=== "seed=256"

    ```python exec="on" source="above"
    import re  # markdown-exec: hide
    from pogema import pogema_v0, GridConfig, BatchAStarAgent

    env = pogema_v0(GridConfig(
        seed=256, num_agents=4, size=8, density=0.3,
        observation_type='POMAPF', max_episode_steps=64,
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

Notice how each seed produces a completely different map, start positions, and targets.

## Reseeding on Reset

Use `env.reset(seed=...)` to change the seed at runtime (standard Gymnasium API). Each new seed produces a new deterministic environment:

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(num_agents=2))

obs_a, _ = env.reset(seed=100)
obs_b, _ = env.reset(seed=200)  # different environment
obs_c, _ = env.reset(seed=100)  # same as obs_a

assert all((o1 == o2).all() for o1, o2 in zip(obs_a, obs_c))
```

## Random Mode

When `seed=None` (default), a different random environment is generated on each reset. This is useful for training:

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(seed=None, num_agents=4))

obs1, _ = env.reset()
obs2, _ = env.reset()  # likely different map and positions
```

## Seeds with Custom Maps

When a `map` is provided, obstacles are fixed regardless of the seed. The seed still controls agent and target placement on free cells:

=== "seed=1"

    ```python exec="on" source="above"
    import re  # markdown-exec: hide
    from pogema import pogema_v0, GridConfig, BatchAStarAgent

    grid = ".....#.....\n.....#.....\n...........\n.....#....."  # markdown-exec: hide
    env = pogema_v0(GridConfig(
        map=grid, num_agents=4, seed=1,
        observation_type='POMAPF', max_episode_steps=64,
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

=== "seed=2"

    ```python exec="on" source="above"
    import re  # markdown-exec: hide
    from pogema import pogema_v0, GridConfig, BatchAStarAgent

    grid = ".....#.....\n.....#.....\n...........\n.....#....."  # markdown-exec: hide
    env = pogema_v0(GridConfig(
        map=grid, num_agents=4, seed=2,
        observation_type='POMAPF', max_episode_steps=64,
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

The obstacle layout (the `#` wall) stays the same — only agent and target positions change between seeds.

When positions are fully specified via named agents (`a`/`A`, `b`/`B`, ...), the seed has no effect — everything is deterministic by definition:

```python exec="on" source="above"
import re  # markdown-exec: hide
from pogema import pogema_v0, GridConfig, BatchAStarAgent

grid = "a..#..A\n.......\nb..#..B"  # markdown-exec: hide
env = pogema_v0(GridConfig(
    map=grid,
    observation_type='POMAPF', max_episode_steps=64,
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

Here agents `a`/`b` navigate to their targets `A`/`B` on a fully specified map — no randomness involved.

## Seeds in Lifelong Mode

In lifelong mode (`on_target='restart'`), the seed controls initial placement and the sequence of new targets. A per-agent RNG is derived from the main seed, so each agent gets a reproducible but independent stream of targets:

```python exec="on" source="above"
import re  # markdown-exec: hide
from pogema import pogema_v0, GridConfig, BatchAStarAgent

env = pogema_v0(GridConfig(
    on_target='restart', seed=42, num_agents=4, size=8,
    density=0.3, observation_type='POMAPF', max_episode_steps=64,
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

The seed ensures the same sequence of regenerated targets on every run.
