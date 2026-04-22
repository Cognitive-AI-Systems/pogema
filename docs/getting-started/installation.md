# Installation

## From PyPI

```bash
pip install pogema
```

## From Source (development)

```bash
git clone https://github.com/Cognitive-AI-Systems/pogema.git
cd pogema
uv sync --extra test --extra dev
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| gymnasium | >= 1.2.3 | Environment interface |
| numpy | >= 2.0 | Grid computations |
| pydantic | >= 2.12.5 | Configuration validation |
| pettingzoo | >= 1.24, < 1.25 | Multi-agent API |

## Optional Dependencies

```bash
# For running tests
pip install pogema[test]

# For development (linting)
pip install pogema[dev]
```

## Verify Installation

```python
import pogema
print(pogema.__version__)
```
