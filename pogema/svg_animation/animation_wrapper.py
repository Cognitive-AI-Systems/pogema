import os
import warnings

from pogema import GridConfig
from pogema.svg_animation.animation_drawer import AnimationConfig
from pogema.wrappers.base import PogemaWrapper
from pogema.wrappers.animation import AnimationWrapper


def _find_animation_wrapper(env):
    wrapper = env
    while wrapper is not None:
        if isinstance(wrapper, AnimationWrapper):
            return wrapper
        wrapper = getattr(wrapper, 'env', None)
    return None


class AnimationMonitor(PogemaWrapper):
    """
    Deprecated: Use env.enable_animation() instead.
    """

    def __init__(self, env, animation_config=AnimationConfig()):
        warnings.warn(
            "AnimationMonitor is deprecated. Use env.enable_animation(animation_config) "
            "and env.save_animation(name) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(env)
        self.animation_config: AnimationConfig = animation_config
        self._episode_idx = 0

        aw = _find_animation_wrapper(self.env)
        if aw is not None:
            aw.enable_animation(animation_config)
        else:
            raise RuntimeError("No AnimationWrapper found in the wrapper chain.")

    def step(self, action):
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
                        os.makedirs(self.animation_config.directory, exist_ok=True)

                    path = os.path.join(self.animation_config.directory,
                                        self.pick_name(self.unwrapped.grid_config, self._episode_idx))
                    self.save_animation(path)

        return obs, reward, terminated, truncated, info

    @staticmethod
    def pick_name(grid_config: GridConfig, episode_idx=None, zfill_ep=5):
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
        obs = self.env.reset(**kwargs)
        self._episode_idx += 1
        return obs

    def save_animation(self, name='render.svg', animation_config=None):
        if animation_config is None:
            animation_config = self.animation_config
        aw = _find_animation_wrapper(self.env)
        aw.save_animation(name, animation_config=animation_config)


def main():
    from pogema import GridConfig, pogema_v0, BatchAStarAgent

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
            env.enable_animation()

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
            env.save_animation(f'{anim_folder}/anim-static.svg', AnimationConfig(static_frame_idx=0))
            env.save_animation(f'{anim_folder}/anim-static-ego.svg',
                               AnimationConfig(egocentric_idx=0, static_frame_idx=0))
            env.save_animation(f'{anim_folder}/anim-static-no-agents.svg',
                               AnimationConfig(show_agents=False, static_frame_idx=0))
            env.disable_animation()


if __name__ == '__main__':
    main()
