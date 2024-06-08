try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print(
        """To use the integration it's necessary to intall `jax`:
        https://jax.readthedocs.io/en/latest/installation.html.
        Terminating.
    """
    )
    exit(-1)

import functools
from typing import NamedTuple

from pogema import GridConfig
from pogema.envs import _make_pogema


class EnvState(NamedTuple):
    t: jax.Array


def make_step_callback(env):
    def callback_step(action, env_state):
        observation, reward, terminated, truncated, info = env.step(
            action=action.tolist()
        )
        return (
            jnp.array(observation),
            (env_state + 1).astype(jnp.int32),
            jnp.array(reward).astype(jnp.float32),
            jnp.array(terminated).astype(jnp.bool),
            jnp.array(truncated).astype(jnp.bool),
            None,  # operating without info
        )

    return callback_step


def make_reset_callback(env):
    def callback_reset(seed):
        int_seed = jax.random.randint(seed, (), 0, 1000)
        observation, _ = env.reset(seed=int_seed)
        return jnp.array(observation)

    return jax.jit(callback_reset)


class PogemaForJax:
    def __init__(self, grid_config: GridConfig) -> None:
        self._env = _make_pogema(grid_config)

        self.callback_reset = make_reset_callback(self._env)
        self.callback_step = make_step_callback(self._env)
        self.num_agents = self._env.get_num_agents()

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        def jit_reset(*args):
            reset_shape = jax.ShapeDtypeStruct(
                (self.num_agents, *self._env.observation_space.shape), jnp.float32
            )
            return jax.pure_callback(self.callback_reset, reset_shape, *args), EnvState(
                jnp.array(0)
            )

        return jit_reset(key)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, action, env_state):

        def jit_step(*args):

            step_shape = (
                jax.ShapeDtypeStruct(
                    (self.num_agents, *self._env.observation_space.shape), jnp.float32
                ),
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((self.num_agents,), jnp.float32),
                jax.ShapeDtypeStruct((self.num_agents,), jnp.bool),
                jax.ShapeDtypeStruct((self.num_agents,), jnp.bool),
                None,
            )
            return jax.experimental.io_callback(self.callback_step, step_shape, *args)

        observation, t, reward, terminated, truncated, info = jit_step(
            action, env_state.t
        )

        return observation, EnvState(t), reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self._env

    def close(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        return self._env.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self):
        return self._env.action_space
