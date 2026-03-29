# Key Concepts

## Agents and Goals

Each agent has a start position and a target position on the grid. The agent's objective is to navigate from start to target while avoiding obstacles and other agents.

## Partial Observability

Agents can only see a local region around themselves, controlled by `obs_radius`. The default radius is 5, giving an 11x11 observation window. Agents have no knowledge of the global map or other agents outside their field of view.

## Actions

5 discrete actions:

| Action | ID | Movement |
|--------|----|----------|
| Wait | 0 | Stay in place |
| Up | 1 | y - 1 |
| Down | 2 | y + 1 |
| Left | 3 | x - 1 |
| Right | 4 | x + 1 |

## Observations

Default observations are arrays of shape `(3, 2*R+1, 2*R+1)` where `R = obs_radius`:

| Channel | Content | Values |
|---------|---------|--------|
| 0 | Obstacles | 1.0 = obstacle, 0.0 = free |
| 1 | Other agents | 1.0 = agent present |
| 2 | Target direction | Encodes relative goal position |

See [Observation Types](../configuration/observation-types.md) for alternative formats.

## Task Modes (`on_target`)

| Mode | Behavior | Use Case |
|------|----------|----------|
| `'nothing'` | Agent stays, all must reach goals simultaneously | Classical MAPF |
| `'restart'` | Agent gets a new goal upon reaching current one | Lifelong MAPF |
| `'finish'` | Agent disappears upon reaching goal | Simplified MAPF |

See [Task Modes](../environment/modes.md) for details.

## Collision Systems

| System | Behavior |
|--------|----------|
| `'block_both'` | Both colliding agents stay in place |
| `'priority'` | Higher-index agent moves, lower is blocked |
| `'soft'` | Agents can overlap freely |

See [Collision Systems](../configuration/collision-systems.md) for details.

## Episode Lifecycle

1. `env.reset()` — generate or load map, place agents and targets
2. `env.step(actions)` — all agents move simultaneously
3. Episode ends when all agents are terminated or truncated
4. Metrics available in `info[0]['metrics']` on the final step
