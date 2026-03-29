# Wrappers

POGEMA uses Gymnasium's wrapper pattern. All wrappers extend `PogemaWrapper`, which forwards POGEMA-specific methods through the wrapper stack.

## PogemaWrapper

Base class for all POGEMA wrappers. Automatically forwards:

- `get_num_agents()`, `get_agents_xy()`, `get_targets_xy()`
- `get_obstacles()`, `get_state()`
- `sample_actions()`
- `enable_animation()`, `disable_animation()`, `save_animation()`
- `grid_config`, `grid`, `was_on_goal` properties
- And more

## Writing a Custom Wrapper

```python
from pogema import PogemaWrapper

class MyWrapper(PogemaWrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Custom logic here
        return obs, reward, terminated, truncated, info
```

All POGEMA methods remain accessible through your wrapper automatically.

## Default Wrapper Stack

When you call `pogema_v0()`, the following wrappers are applied automatically:

1. **AnimationWrapper** — SVG recording (inactive by default, zero overhead)
2. **MultiTimeLimit** — Episode time limit enforcement
3. **Metric wrappers** — CSR, ISR, EpLength, etc. (selected based on `on_target`)
