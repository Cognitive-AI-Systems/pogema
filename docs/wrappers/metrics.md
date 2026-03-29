# Metrics

Metrics are automatically computed and available in `info[0]['metrics']` on the final step of an episode.

## Classical MAPF Metrics (`on_target='nothing'`)

| Metric | Key | Description | Range |
|--------|-----|-------------|-------|
| CSR | `'CSR'` | 1.0 if all agents are on goals simultaneously | {0, 1} |
| ISR | `'ISR'` | Fraction of agents on goals at episode end | [0, 1] |
| EpLength | `'EpLength'` | Total episode steps | [1, max_steps] |
| SumOfCosts | `'SumOfCosts'` | Total steps across all agents | [num_agents, ...] |
| Makespan | `'Makespan'` | Steps until last agent finishes | [1, max_steps] |

## Lifelong Metrics (`on_target='restart'`)

| Metric | Key | Description |
|--------|-----|-------------|
| LifeLongAverageThroughput | `'avg_throughput'` | (Total goals reached) / max_steps |

## Finish Metrics (`on_target='finish'`)

| Metric | Key | Description | Range |
|--------|-----|-------------|-------|
| CSR | `'CSR'` | 1.0 if ALL agents reached goals | {0, 1} |
| ISR | `'ISR'` | Fraction of agents that reached goals | [0, 1] |
| EpLength | `'EpLength'` | Average steps to reach goal per agent | [1, max_steps] |
| SumOfCosts | `'SumOfCosts'` | Total steps across all agents | [num_agents, ...] |
| Makespan | `'Makespan'` | Steps until last agent finishes | [1, max_steps] |

## Accessing Metrics

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(num_agents=4, size=8, seed=42))
obs, info = env.reset()

while True:
    obs, reward, terminated, truncated, info = env.step(env.sample_actions())
    if all(terminated) or all(truncated):
        break

metrics = info[0]['metrics']
print(f"CSR: {metrics['CSR']}, ISR: {metrics['ISR']}")
```
