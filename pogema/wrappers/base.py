from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium import Wrapper

if TYPE_CHECKING:
    from pogema.grid import Grid
    from pogema.grid_config import GridConfig

_POGEMA_FORWARDED = frozenset({
    # PogemaBase methods
    'get_num_agents', 'get_obstacles', 'get_agents_xy', 'get_targets_xy',
    'get_state', 'get_agents_xy_relative', 'get_targets_xy_relative',
    'sample_actions',
    # PogemaBase attributes
    'grid_config', 'grid', 'was_on_goal',
    # PogemaLifeLong
    'get_lifelong_targets_xy',
    # AnimationWrapper
    'enable_animation', 'disable_animation', 'save_animation', 'animation_is_active',
})


class PogemaWrapper(Wrapper):
    env: PogemaWrapper

    # -- Gymnasium overrides with multi-agent types --

    def step(self, action) -> tuple[list, list[float], list[bool], list[bool], list[dict]]:
        return self.env.step(action)

    def reset(self, **kwargs) -> tuple[list, list[dict]]:
        return self.env.reset(**kwargs)

    # -- PogemaBase methods --

    def get_num_agents(self) -> int:
        return self.env.get_num_agents()

    def get_obstacles(self, ignore_borders: bool = False) -> np.ndarray:
        return self.env.get_obstacles(ignore_borders=ignore_borders)

    def get_agents_xy(self, only_active: bool = False, ignore_borders: bool = False) -> list:
        return self.env.get_agents_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_targets_xy(self, only_active: bool = False, ignore_borders: bool = False) -> list:
        return self.env.get_targets_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_state(self, ignore_borders: bool = False, as_dict: bool = False):
        return self.env.get_state(ignore_borders=ignore_borders, as_dict=as_dict)

    def get_agents_xy_relative(self) -> list:
        return self.env.get_agents_xy_relative()

    def get_targets_xy_relative(self) -> list:
        return self.env.get_targets_xy_relative()

    def sample_actions(self) -> np.ndarray:
        return self.env.sample_actions()

    # -- PogemaBase attributes --

    @property
    def grid_config(self) -> GridConfig:
        return self.unwrapped.grid_config

    @property
    def grid(self) -> Grid:
        return self.unwrapped.grid

    @property
    def was_on_goal(self) -> list:
        return self.unwrapped.was_on_goal

    # -- PogemaLifeLong --

    def get_lifelong_targets_xy(self, ignore_borders: bool = False) -> list:
        return self.env.get_lifelong_targets_xy(ignore_borders=ignore_borders)

    # -- MultiTimeLimit --

    def set_elapsed_steps(self, elapsed_steps: int) -> None:
        return self.env.set_elapsed_steps(elapsed_steps)

    # -- AnimationWrapper --

    def enable_animation(self, animation_config=None):
        return self.env.enable_animation(animation_config)

    def disable_animation(self):
        return self.env.disable_animation()

    def save_animation(self, name='render.svg', animation_config=None):
        return self.env.save_animation(name, animation_config=animation_config)

    @property
    def animation_is_active(self):
        return self.env.animation_is_active

    # -- Fallback for any remaining forwarded names --

    def __getattr__(self, name):
        if name in _POGEMA_FORWARDED:
            return getattr(self.env, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
