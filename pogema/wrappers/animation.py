import os
from itertools import cycle

from pogema.wrappers.base import PogemaWrapper
from pogema.wrappers.persistence import AgentState, decompress_history


class SvgAnimation:
    def __init__(self, svg_str):
        self._svg_str = svg_str

    def _repr_html_(self):
        return self._svg_str

    def save(self, path='render.svg'):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(self._svg_str)

    def __str__(self):
        return self._svg_str

    def __repr__(self):
        return f"SvgAnimation({len(self._svg_str)} chars)"


class AnimationWrapper(PogemaWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._active = False
        self._animation_config = None
        self._step = None
        self._agent_states = None

    def step(self, action):
        result = self.env.step(action)
        if not self._active:
            return result
        self._step += 1
        for agent_idx in range(self.unwrapped.get_num_agents()):
            agent_state = self._get_agent_state(self.unwrapped.grid, agent_idx)
            if agent_state != self._agent_states[agent_idx][-1]:
                self._agent_states[agent_idx].append(agent_state)

        return result

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if not self._active:
            return result

        self._step = 0

        self._agent_states = []
        for agent_idx in range(self.unwrapped.get_num_agents()):
            self._agent_states.append([self._get_agent_state(self.unwrapped.grid, agent_idx)])

        return result

    def _get_agent_state(self, grid, agent_idx):
        x, y = grid.positions_xy[agent_idx]
        tx, ty = grid.finishes_xy[agent_idx]
        active = grid.is_active[agent_idx]
        return AgentState(x, y, tx, ty, self._step, active)

    def enable_animation(self, animation_config=None):
        self._active = True
        if animation_config is not None:
            self._animation_config = animation_config

    def disable_animation(self):
        self._active = False

    @property
    def animation_is_active(self):
        return self._active

    def _build_svg_string(self, animation_config=None):
        if not self._active:
            raise RuntimeError(
                "Animation is not active. Call env.enable_animation() and then env.reset() before saving."
            )
        if self._agent_states is None:
            raise RuntimeError(
                "No history recorded. Call env.reset() after enable_animation() before saving."
            )

        from pogema.svg_animation.animation_drawer import AnimationConfig, AnimationDrawer, GridHolder, SvgSettings

        if animation_config is None:
            animation_config = self._animation_config
        if animation_config is None:
            animation_config = AnimationConfig()

        working_radius = self.unwrapped.grid_config.obs_radius - 1
        if working_radius > 0:
            obstacles = self.unwrapped.get_obstacles(ignore_borders=False)[working_radius:-working_radius,
                        working_radius:-working_radius]
        else:
            obstacles = self.unwrapped.get_obstacles(ignore_borders=False)

        # Apply offset at render time to shift positions into trimmed coordinate space
        offset = -working_radius
        raw_history = self._agent_states
        shifted_history = []
        for agent_states in raw_history:
            shifted = []
            for s in agent_states:
                if offset != 0:
                    shifted.append(AgentState(s.x + offset, s.y + offset, s.tx + offset, s.ty + offset, s.step, s.active))
                else:
                    shifted.append(s)
            shifted_history.append(shifted)

        history = decompress_history(shifted_history)

        svg_settings = SvgSettings()
        colors_cycle = cycle(svg_settings.colors)
        agents_colors = {index: next(colors_cycle) for index in range(self.unwrapped.grid_config.num_agents)}

        for agent_idx in range(self.unwrapped.grid_config.num_agents):
            history[agent_idx].append(history[agent_idx][-1])

        episode_length = len(history[0])
        if animation_config.egocentric_idx is not None and self.unwrapped.grid_config.on_target == 'finish':
            episode_length = history[animation_config.egocentric_idx][-1].step + 1
            for agent_idx in range(self.unwrapped.grid_config.num_agents):
                history[agent_idx] = history[agent_idx][:episode_length]

        grid_holder = GridHolder(
            width=len(obstacles), height=len(obstacles[0]),
            obstacles=obstacles,
            episode_length=episode_length,
            history=history,
            obs_radius=self.unwrapped.grid_config.obs_radius,
            on_target=self.unwrapped.grid_config.on_target,
            colors=agents_colors,
            config=animation_config,
            svg_settings=svg_settings,
        )

        animation = AnimationDrawer().create_animation(grid_holder)
        return animation.render()

    def render_animation(self, animation_config=None):
        return SvgAnimation(self._build_svg_string(animation_config=animation_config))

    def save_animation(self, name='render.svg', animation_config=None):
        self.render_animation(animation_config=animation_config).save(name)
