import os
import time
import warnings

import numpy as np
import pytest
from tabulate import tabulate

from pogema import pogema_v0, AnimationMonitor, AnimationConfig

from pogema.envs import ActionsSampler
from pogema.grid import GridConfig
from pogema.wrappers.persistence import PersistentWrapper


class ActionMapping:
    noop: int = 0
    up: int = 1
    down: int = 2
    left: int = 3
    right: int = 4


def test_moving():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42))
    ac = ActionMapping()
    env.reset()

    env.step([ac.right, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.left, ac.noop])
    env.step([ac.down, ac.noop])
    env.step([ac.down, ac.noop])
    env.step([ac.left, ac.noop])
    env.step([ac.left, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.up, ac.noop])

    env.step([ac.right, ac.noop])
    env.step([ac.up, ac.noop])
    env.step([ac.right, ac.noop])
    env.step([ac.down, ac.noop])
    obs, reward, terminated, truncated, infos = env.step([ac.right, ac.noop])

    assert np.isclose([1.0, 0.0], reward).all()
    assert np.isclose([True, False], terminated).all()


def test_types():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42))
    obs, info = env.reset()
    assert obs[0].dtype == np.float32


def run_episode(grid_config=None, env=None):
    if env is None:
        env = pogema_v0(grid_config)
    env.reset()

    obs, rewards, terminated, truncated, infos = env.reset(), [None], [False], [False], [None]

    results = [[obs, rewards, terminated, truncated, infos]]
    while True:
        results.append(env.step(env.unwrapped.sample_actions()))
        terminated, truncated = results[-1][2], results[-1][3]
        if all(terminated) or all(truncated):
            break
    return results


def test_metrics():
    *_, infos = run_episode(GridConfig(num_agents=2, seed=5, size=5, max_episode_steps=64))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 0.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 0.5)

    *_, infos = run_episode(GridConfig(num_agents=2, seed=5, size=5, max_episode_steps=512))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 1.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 1.0)

    *_, infos = run_episode(GridConfig(num_agents=5, seed=5, size=5, max_episode_steps=64))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 0.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 0.2)


def test_standard_pogema():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish'))
    env.reset()
    run_episode(env=env)


def test_pomapf_observation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish',
                               observation_type='POMAPF'))
    obs, info = env.reset()
    assert 'agents' in obs[0]
    assert 'obstacles' in obs[0]
    assert 'xy' in obs[0]
    assert 'target_xy' in obs[0]
    run_episode(env=env)


def test_mapf_observation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish',
                               observation_type='MAPF'))
    obs, info = env.reset()
    assert 'global_obstacles' in obs[0]
    assert 'global_xy' in obs[0]
    assert 'global_target_xy' in obs[0]
    run_episode(env=env)


def test_standard_pogema_animation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish'))
    env = AnimationMonitor(env)
    env.reset()
    run_episode(env=env)


def test_gym_pogema_animation():
    import gymnasium
    env = gymnasium.make('Pogema-v0',
                         grid_config=GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42,
                                                on_target='finish'))
    env = AnimationMonitor(env)
    env.reset()

    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break


def test_non_disappearing_pogema():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='nothing'))
    env.reset()
    run_episode(env=env)


def test_non_disappearing_pogema_no_seed():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=None, on_target='nothing'))
    env.reset()
    run_episode(env=env)


def test_non_disappearing_pogema_animation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='nothing'))
    env = AnimationMonitor(env)
    env.reset()
    run_episode(env=env)


def test_life_long_pogema():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='restart'))
    env.reset()
    run_episode(env=env)


def test_life_long_pogema_empty_seed():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=None, on_target='restart'))
    env.reset()
    run_episode(env=env)


def test_life_long_pogema_animation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='restart'))
    env = AnimationMonitor(env)
    env.reset()
    run_episode(env=env)


def test_custom_positions_and_num_agents():
    grid = """
    ....
    ....
    """
    gc = GridConfig(
        map=grid,
        agents_xy=[[0, 0], [0, 1], [0, 2], [0, 3]],
        targets_xy=[[1, 0], [1, 1], [1, 2], [1, 3]],
    )

    for num_agents in range(1, 5):
        gc.num_agents = num_agents
        env = pogema_v0(grid_config=gc)
        env.reset()
        assert num_agents == len(env.unwrapped.get_agents_xy())
        assert num_agents == len(env.unwrapped.get_targets_xy())


def test_custom_positions_and_empty_num_agents():
    grid = """
    ....
    ....
    """
    gc = GridConfig(
        map=grid,
        agents_xy=[[0, 0], [0, 1], [0, 2], [0, 3]],
        targets_xy=[[1, 0], [1, 1], [1, 2], [1, 3]],
    )
    env = pogema_v0(grid_config=gc)
    env.reset()
    assert len(gc.agents_xy) == len(env.unwrapped.get_agents_xy())


def test_persistent_env(num_steps=100):
    seed = 42

    env = pogema_v0(
        grid_config=GridConfig(on_target='finish', seed=seed, num_agents=8, density=0.132, size=8, obs_radius=2))
    env = PersistentWrapper(env)

    env.reset()
    action_sampler = ActionsSampler(env.action_space.n, seed=seed)

    first_run_observations = []

    def state_repr(observations, rewards, terminates, truncates, infos):
        return np.concatenate([np.array(observations).flatten(), terminates, truncates, np.array(rewards), ])

    for current_step in range(num_steps):
        actions = action_sampler.sample_actions(dim=env.unwrapped.get_num_agents())
        obs, reward, terminated, truncated, info = env.step(actions)

        first_run_observations.append(state_repr(obs, reward, terminated, truncated, info))
        if all(terminated) or all(truncated):
            break

    # resetting the environment to the initial state using backward steps
    for current_step in range(num_steps):
        if not env.step_back():
            break

    action_sampler = ActionsSampler(env.action_space.n, seed=seed)

    second_run_observations = []
    for current_step in range(num_steps):
        actions = action_sampler.sample_actions(dim=env.unwrapped.get_num_agents())
        obs, reward, terminated, truncated, info = env.step(actions)
        second_run_observations.append(state_repr(obs, reward, terminated, truncated, info))
        assert np.isclose(first_run_observations[current_step], second_run_observations[current_step]).all()
        if all(terminated) or all(truncated):
            break
    assert np.isclose(first_run_observations, second_run_observations).all()


def test_wrapper_attribute_forwarding():
    import pytest
    for on_target in ['finish', 'nothing', 'restart']:
        gc = GridConfig(num_agents=2, size=6, seed=42, on_target=on_target)
        env = pogema_v0(gc)
        env.reset()

        assert env.get_num_agents() == 2
        assert env.grid_config is not None
        assert env.sample_actions() is not None
        assert env.get_obstacles() is not None
        assert env.get_agents_xy() is not None
        assert env.get_targets_xy() is not None

        with pytest.raises(AttributeError):
            env.nonexistent_attribute_xyz


def test_wrapper_forwarding_persistent():
    gc = GridConfig(num_agents=2, size=6, seed=42, on_target='finish')
    env = pogema_v0(gc)
    env = PersistentWrapper(env)
    env.reset()

    assert env.get_num_agents() == 2
    assert env.get_history() is not None
    assert env.grid_config is not None


def test_wrapper_forwarding_animation():
    gc = GridConfig(num_agents=2, size=6, seed=42, on_target='finish')
    env = pogema_v0(gc)
    env = AnimationMonitor(env)
    env.reset()

    assert env.get_num_agents() == 2
    assert env.grid_config is not None


def test_steps_per_second_throughput():
    table = []
    for on_target in ['finish', 'nothing', 'restart']:
        for num_agents in [1, 32, 64]:
            for size in [32, 64]:
                gc = GridConfig(obs_radius=5, seed=42, max_episode_steps=1024,
                              size=size, num_agents=num_agents, on_target=on_target)

                start_time = time.monotonic()
                run_episode(grid_config=gc)
                end_time = time.monotonic()
                steps_per_second = gc.max_episode_steps / (end_time - start_time)
                table.append([on_target, num_agents, size, steps_per_second * gc.num_agents])
    print('\n' + tabulate(table, headers=['on_target', 'num_agents', 'size', 'SPS (individual)'], tablefmt='grid'))


def test_enable_animation_and_save(tmp_path):
    gc = GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish')
    env = pogema_v0(gc)
    env.enable_animation()
    env.reset()
    run_episode(env=env)

    svg_path = str(tmp_path / 'test_anim.svg')
    env.save_animation(svg_path)
    assert os.path.exists(svg_path)
    with open(svg_path) as f:
        content = f.read()
    assert '<svg' in content


def test_no_overhead_without_animation():
    gc = GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish')
    env = pogema_v0(gc)
    assert not env.animation_is_active
    env.reset()
    run_episode(env=env)


def test_disable_animation():
    gc = GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish')
    env = pogema_v0(gc)
    env.enable_animation()
    assert env.animation_is_active
    env.disable_animation()
    assert not env.animation_is_active


def test_save_animation_without_enable_raises():
    gc = GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish')
    env = pogema_v0(gc)
    env.reset()
    with pytest.raises(RuntimeError, match="Animation is not active"):
        env.save_animation('test.svg')


def test_animation_monitor_backward_compat():
    gc = GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish')
    env = pogema_v0(gc)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
    env.reset()
    run_episode(env=env)


def test_metrics_with_animation():
    for on_target in ['finish', 'nothing', 'restart']:
        gc = GridConfig(num_agents=2, seed=5, size=5, max_episode_steps=64, on_target=on_target)
        env = pogema_v0(gc)
        env.enable_animation()
        env.reset()
        *_, infos = run_episode(env=env)[-1]
        assert 'metrics' in infos[0]


def test_enable_animation_for_all_on_target_modes(tmp_path):
    for on_target in ['finish', 'nothing', 'restart']:
        gc = GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42,
                        on_target=on_target, max_episode_steps=16)
        env = pogema_v0(gc)
        env.enable_animation()
        env.reset()
        run_episode(env=env)
        svg_path = str(tmp_path / f'test_{on_target}.svg')
        env.save_animation(svg_path)
        assert os.path.exists(svg_path)
