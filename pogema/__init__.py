from gymnasium import register
from pogema.grid_config import GridConfig
from pogema.integrations.make_pogema import pogema_v0
from pogema.svg_animation.animation_wrapper import AnimationMonitor
from pogema.svg_animation.animation_drawer import AnimationConfig
from pogema.a_star_policy import AStarAgent, BatchAStarAgent
from pogema.wrappers.base import PogemaWrapper
from pogema.wrappers.animation import AnimationWrapper
from pogema.wrappers.persistence import PersistentWrapper

__version__ = '1.4.0'

__all__ = [
    'GridConfig',
    'pogema_v0',
    'AStarAgent', 'BatchAStarAgent',
    "PogemaWrapper",
    "AnimationWrapper",
    "PersistentWrapper",
]

register(
    id="Pogema-v0",
    entry_point="pogema.integrations.make_pogema:make_single_agent_gym",
)
