# POGEMA

**Partially-Observable Grid Environment for Multiple Agents**

[![Paper](https://img.shields.io/badge/Paper-ICLR%202025-1f4b99?logo=openreview&logoColor=white)](https://openreview.net/forum?id=6VgwE2tCRm)

POGEMA is a fast, flexible benchmarking platform for cooperative multi-agent pathfinding (MAPF). Agents navigate grid maps under partial observability, making decentralized decisions at each time step.

## Installation

```bash
pip install git+https://github.com/Cognitive-AI-Systems/pogema.git
```


Python 3.10+ required. See [Installation](getting-started/installation.md) for extras and development setup.


## Highlights

- **Fast**: Lightweight NumPy-based grid engine — thousands of steps per second
- **Flexible**: Random or custom maps, 3 collision modes, 3 task modes, configurable observations
- **Multi-framework**: Native integrations with Gymnasium, PettingZoo, and SampleFactory
- **Visualization**: Built-in SVG animation recorder with zero overhead when inactive
- **Reproducible**: Seed-controlled generation for maps, agent positions, and targets

## Quick Example

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(num_agents=4, size=8, seed=42))
obs, info = env.reset()

while True:
    obs, reward, terminated, truncated, info = env.step(env.sample_actions())
    if all(terminated) or all(truncated):
        break

print(info[0]['metrics'])
# {'CSR': 0.0, 'ISR': 0.25, 'EpLength': 64.0, ...}
```

## Citation

If you use POGEMA in your research, please cite:

```bibtex
@inproceedings{skrynnik2025pogema,
  title={POGEMA: A Benchmark Platform for Cooperative Multi-Agent Pathfinding},
  author={Skrynnik, Alexey and Andreychuk, Anton and Borzilov, Anatolii and Chernyavskiy, Alexander and Yakovlev, Konstantin and Panov, Aleksandr},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
