# Custom Maps

## Map String Format

Maps are defined as multi-line strings. Each character represents a cell:

| Symbol | Meaning |
|--------|---------|
| `.` | Free cell |
| `#` | Obstacle |
| `@` | Agent start position (unnamed) |
| `$` | Target position (unnamed) |
| `a`-`z` | Named agent start (agent `a`, `b`, etc.) |
| `A`-`Z` | Named target (target `A` matches agent `a`) |
| `!` | Flexible start position |

## Basic Custom Map

```python
from pogema import pogema_v0, GridConfig

grid = """
.....#.....
.....#.....
...........
.....#.....
.....#.....
#.####.....
.....###.##
.....#.....
.....#.....
...........
.....#.....
"""

env = pogema_v0(GridConfig(map=grid, num_agents=8))
```

Agents and targets are placed randomly on free cells.

## Map with Fixed Positions

```python
from pogema import pogema_v0, GridConfig

grid = """
a..#..A
.......
b..#..B
"""

# Agent 'a' starts top-left, targets top-right (A)
# Agent 'b' starts bottom-left, targets bottom-right (B)
env = pogema_v0(GridConfig(map=grid))
```

## Map as List of Lists

```python
from pogema import pogema_v0, GridConfig

# 0 = free, 1 = obstacle
grid = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
]

env = pogema_v0(GridConfig(map=grid, num_agents=2))
```

## Non-Square Maps

```python
from pogema import pogema_v0, GridConfig

env = pogema_v0(GridConfig(width=20, height=10, density=0.2, num_agents=4))
```
