from gymnasium import register
from pogema.grid_config import GridConfig
from pogema.integrations.make_pogema import pogema_v0
from pogema.svg_animation.animation_wrapper import AnimationMonitor, AnimationConfig
from pogema.a_star_policy import AStarAgent, BatchAStarAgent

__version__ = '1.4.0'

__all__ = [
    'GridConfig',
    'pogema_v0',
    'AStarAgent', 'BatchAStarAgent',
    "AnimationMonitor", "AnimationConfig",
]

register(
    id="Pogema-v0",
    entry_point="pogema.integrations.make_pogema:make_single_agent_gym",
)
