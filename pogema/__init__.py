from gymnasium import register

from pogema.grid_config import GridConfig
from pogema.a_star_policy import AStarAgent, BatchAStarAgent
from pogema.integrations.make_pogema import pogema_v0, SingleAgentWrapper
from pogema.grid_registry import register_grid, get_grid
from pogema.svg_animation.animation_drawer import AnimationConfig
from pogema.svg_animation.animation_wrapper import AnimationMonitor
from pogema.wrappers.animation import AnimationWrapper, SvgAnimation
from pogema.wrappers.base import PogemaWrapper
from pogema.wrappers.metrics import (
    CSRMetric,
    ISRMetric,
    EpLengthMetric,
    NonDisappearCSRMetric,
    NonDisappearISRMetric,
    NonDisappearEpLengthMetric,
    SumOfCostsAndMakespanMetric,
    LifeLongAverageThroughputMetric,
    RuntimeMetricWrapper,
)
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.wrappers.persistence import PersistentWrapper

__version__ = '1.4.0'

__all__ = [
    'GridConfig',
    'pogema_v0',
    'AnimationMonitor',
    'AnimationConfig',
    'AStarAgent', 'BatchAStarAgent',
    'PogemaWrapper',
    'AnimationWrapper',
    'SvgAnimation',
    'PersistentWrapper',
    'SingleAgentWrapper',
    'MultiTimeLimit',
    'register_grid', 'get_grid',
    'CSRMetric', 'ISRMetric', 'EpLengthMetric',
    'NonDisappearCSRMetric', 'NonDisappearISRMetric', 'NonDisappearEpLengthMetric',
    'SumOfCostsAndMakespanMetric', 'LifeLongAverageThroughputMetric',
    'RuntimeMetricWrapper',
]

register(
    id="Pogema-v0",
    entry_point="pogema.integrations.make_pogema:make_single_agent_gym",
    order_enforce=False,
    disable_env_checker=True,
)
