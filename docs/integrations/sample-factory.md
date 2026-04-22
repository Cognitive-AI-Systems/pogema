# SampleFactory

Integration for the [SampleFactory](https://github.com/alex-petrenko/sample-factory) framework with auto-reset and metrics forwarding.

## Basic Usage

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(
    integration='SampleFactory',
    auto_reset=True,         # Auto-reset on episode end
    num_agents=4,
    size=8,
))
```

## Features

- **AutoResetWrapper**: Automatically resets the environment when an episode ends
- **IsMultiAgentWrapper**: Adds `.num_agents` and `.is_multiagent` attributes
- **MetricsForwardingWrapper**: Exposes metrics via `episode_extra_stats` for SampleFactory logging

## Properties

<!--pytest-codeblocks:skip-->
```python
env.num_agents      # Number of agents (via IsMultiAgentWrapper)
env.is_multiagent   # Always True
```
