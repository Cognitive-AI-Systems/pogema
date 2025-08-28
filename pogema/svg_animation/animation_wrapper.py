import os
from itertools import cycle
from gymnasium import logger, Wrapper

from pogema import GridConfig
from pogema.svg_animation.animation_drawer import AnimationConfig, SvgSettings, GridHolder, AnimationDrawer
from pogema.wrappers.persistence import PersistentWrapper, AgentState


class AnimationMonitor(Wrapper):
    """
    Defines the animation, which saves the episode as SVG.
    """

    def __init__(self, env, animation_config=AnimationConfig()):
        self._working_radius = env.grid_config.obs_radius - 1
        env = PersistentWrapper(env, xy_offset=-self._working_radius)

        super().__init__(env)

        self.history = env.get_history()

        self.svg_settings: SvgSettings = SvgSettings()
        self.animation_config: AnimationConfig = animation_config

        self._episode_idx = 0

    def step(self, action):
        """
        Saves information about the episode.
        :param action: current actions
        :return: obs, reward, done, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        multi_agent_terminated = isinstance(terminated, (list, tuple)) and all(terminated)
        single_agent_terminated = isinstance(terminated, (bool, int)) and terminated
        multi_agent_truncated = isinstance(truncated, (list, tuple)) and all(truncated)
        single_agent_truncated = isinstance(truncated, (bool, int)) and truncated

        if multi_agent_terminated or single_agent_terminated or multi_agent_truncated or single_agent_truncated:
            save_tau = self.animation_config.save_every_idx_episode
            if save_tau:
                if (self._episode_idx + 1) % save_tau or save_tau == 1:
                    if not os.path.exists(self.animation_config.directory):
                        logger.info(f"Creating pogema monitor directory {self.animation_config.directory}", )
                        os.makedirs(self.animation_config.directory, exist_ok=True)

                    path = os.path.join(self.animation_config.directory,
                                        self.pick_name(self.grid_config, self._episode_idx))
                    self.save_animation(path)

        return obs, reward, terminated, truncated, info

    @staticmethod
    def pick_name(grid_config: GridConfig, episode_idx=None, zfill_ep=5):
        """
        Picks a name for the SVG file.
        :param grid_config: configuration of the grid
        :param episode_idx: idx of the episode
        :param zfill_ep: zfill for the episode number
        :return:
        """
        gc = grid_config
        name = 'pogema'
        if episode_idx is not None:
            name += f'-ep{str(episode_idx).zfill(zfill_ep)}'
        if gc:
            if gc.map_name:
                name += f'-{gc.map_name}'
            if gc.seed is not None:
                name += f'-seed{gc.seed}'
        else:
            name += '-render'
        return name + '.svg'

    def reset(self, **kwargs):
        """
        Resets the environment and resets the current positions of agents and targets
        :param kwargs:
        :return: obs: observation
        """
        obs = self.env.reset(**kwargs)

        self._episode_idx += 1
        self.history = self.env.get_history()

        return obs

    def save_animation(self, name='render.svg', animation_config: AnimationConfig = AnimationConfig()):
        """
        Saves the animation.
        :param name: name of the file
        :param animation_config: animation configuration
        :return: None
        """
        wr = self._working_radius
        if wr > 0:
            obstacles = self.env.get_obstacles(ignore_borders=False)[wr:-wr, wr:-wr]
        else:
            obstacles = self.env.get_obstacles(ignore_borders=False)
        history: list[list[AgentState]] = self.env.decompress_history(self.history)

        svg_settings = SvgSettings()
        colors_cycle = cycle(svg_settings.colors)
        agents_colors = {index: next(colors_cycle) for index in range(self.grid_config.num_agents)}

        for agent_idx in range(self.grid_config.num_agents):
            history[agent_idx].append(history[agent_idx][-1])

        episode_length = len(history[0])
        # Change episode length for egocentric environment
        if animation_config.egocentric_idx is not None and self.grid_config.on_target == 'finish':
            episode_length = history[animation_config.egocentric_idx][-1].step + 1
            for agent_idx in range(self.grid_config.num_agents):
                history[agent_idx] = history[agent_idx][:episode_length]

        grid_holder = GridHolder(
            width=len(obstacles), height=len(obstacles[0]),
            obstacles=obstacles,
            episode_length=episode_length,
            history=history,
            obs_radius=self.grid_config.obs_radius,
            on_target=self.grid_config.on_target,
            colors=agents_colors,
            config=animation_config,
            svg_settings=svg_settings
        )

        animation = AnimationDrawer().create_animation(grid_holder)
        with open(name, "w") as f:
            f.write(animation.render())


def main():
    from pogema import GridConfig, pogema_v0, AnimationMonitor, BatchAStarAgent, AnimationConfig

    for egocentric_idx in [0, 1]:
        for on_target in ['nothing', 'restart', 'finish']:
            grid = """
            ....#..
            ..#....
            .......
            .......
            #.#.#..
            #.#.#..
            """
            grid_config = GridConfig(size=32, num_agents=2, obs_radius=2, seed=8, on_target=on_target,
                                     max_episode_steps=16,
                                     density=0.1, map=grid, observation_type="POMAPF")
            env = pogema_v0(grid_config=grid_config)
            env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))

            obs, _ = env.reset()
            truncated = terminated = [False]

            agent = BatchAStarAgent()
            while not all(terminated) and not all(truncated):
                obs, _, terminated, truncated, _ = env.step(agent.act(obs))

            anim_folder = 'renders'
            if not os.path.exists(anim_folder):
                os.makedirs(anim_folder)

            env.save_animation(f'{anim_folder}/anim-{on_target}.svg')
            env.save_animation(f'{anim_folder}/anim-{on_target}-ego-{egocentric_idx}.svg',
                               AnimationConfig(egocentric_idx=egocentric_idx))
            env.save_animation(f'{anim_folder}/anim-static.svg', AnimationConfig(static=True))
            env.save_animation(f'{anim_folder}/anim-static-ego.svg', AnimationConfig(egocentric_idx=0, static=True))
            env.save_animation(f'{anim_folder}/anim-static-no-agents.svg',
                               AnimationConfig(show_agents=False, static=True))


if __name__ == '__main__':
    main()
