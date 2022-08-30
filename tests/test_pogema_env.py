import re
import time

import numpy as np
from tabulate import tabulate

from pogema import pogema_v0
from pogema import Easy8x8, Normal8x8, Hard8x8, ExtraHard8x8
from pogema import Easy16x16, Normal16x16, Hard16x16, ExtraHard16x16
from pogema import Easy32x32, Normal32x32, Hard32x32, ExtraHard32x32
from pogema import Easy64x64, Normal64x64, Hard64x64, ExtraHard64x64

from pogema.animation import AnimationMonitor
from pogema.envs import ActionsSampler
from pogema.grid import GridConfig


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
    obs, reward, done, infos = env.step([ac.right, ac.noop])

    assert np.isclose([1.0, 0.0], reward).all()
    assert np.isclose([True, False], done).all()


def test_types():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42))
    obs = env.reset()
    assert obs[0].dtype == np.float32


def run_episode(grid_config=None, env=None):
    if env is None:
        env = pogema_v0(grid_config)
    env.reset()

    obs, rewards, dones, infos = env.reset(), [None], [False], [None]

    results = [[obs, rewards, dones, infos]]
    while not all(dones):
        results.append(env.step(env.sample_actions()))
        dones = results[-1][-2]
    return results


def test_metrics():
    _, _, _, infos = run_episode(GridConfig(num_agents=2, seed=5, size=5, max_episode_steps=64))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 0.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 0.5)

    _, _, _, infos = run_episode(GridConfig(num_agents=2, seed=5, size=5, max_episode_steps=512))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 1.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 1.0)

    _, _, _, infos = run_episode(GridConfig(num_agents=5, seed=5, size=5, max_episode_steps=64))[-1]
    assert np.isclose(infos[0]['metrics']['CSR'], 0.0)
    assert np.isclose(infos[0]['metrics']['ISR'], 0.2)


def test_standard_pogema():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish'))
    env.reset()
    run_episode(env=env)


def test_pomapf_observation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish',
                               observation_type='POMAPF'))
    obs = env.reset()
    assert 'agents' in obs[0]
    assert 'obstacles' in obs[0]
    assert 'xy' in obs[0]
    assert 'target_xy' in obs[0]
    run_episode(env=env)


def test_mapf_observation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish',
                               observation_type='MAPF'))
    obs = env.reset()
    assert 'global_obstacles' in obs[0]
    assert 'global_xy' in obs[0]
    assert 'global_target_xy' in obs[0]
    run_episode(env=env)


def test_standard_pogema_animation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='finish'))
    env = AnimationMonitor(env)
    env.reset()
    run_episode(env=env)


def test_non_disappearing_pogema():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='nothing'))
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


def test_life_long_pogema_animation():
    env = pogema_v0(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, on_target='restart'))
    env = AnimationMonitor(env)
    env.reset()
    run_episode(env=env)


def test_predefined_configurations():
    def get_num_agents_by_target_density(size, agent_density, obstacle_density):
        return round(agent_density * (size * size * (1.0 - obstacle_density)))

    def get_target_density_by_num_agents(size, num_agents, obstacle_density):
        return num_agents / (size * size * (1.0 - obstacle_density))

    predefined_grids = [
        Easy8x8, Normal8x8, Hard8x8, ExtraHard8x8,
        Easy16x16, Normal16x16, Hard16x16, ExtraHard16x16,
        Easy32x32, Normal32x32, Hard32x32, ExtraHard32x32,
        Easy64x64, Normal64x64, Hard64x64, ExtraHard64x64,
    ]

    # checking that the number of agents (agent density) is correct
    for make_grid_config_func in predefined_grids:
        gc = make_grid_config_func(seed=42)
        for difficulty, agent_density in zip(['Easy', 'Normal', 'Hard', 'ExtraHard'],
                                             [0.02232142, 0.04464285, 0.08928571, 0.17857142]):
            if re.match(f'^{difficulty}\d+x\d+', make_grid_config_func.__name__):
                assert np.isclose(get_target_density_by_num_agents(gc.size, gc.num_agents, gc.density), agent_density)

    # checking creation
    for make_grid_config_func in predefined_grids:
        gc = make_grid_config_func(seed=42)
        env = pogema_v0(gc)
        env.reset()

    # checking map_name
    for make_grid_config_func in predefined_grids:
        gc = make_grid_config_func(seed=42)
        assert gc.map_name == make_grid_config_func.__name__



def test_persistent_env(num_steps=100):
    seed = 42

    env = pogema_v0(
        grid_config=GridConfig(on_target='finish', seed=seed, num_agents=8, density=0.132, size=8, obs_radius=2,
                               persistent=True))

    env.reset()
    action_sampler = ActionsSampler(env.action_space.n, seed=seed)

    first_run_observations = []

    def state_repr(observations, rewards, dones, infos):
        return np.concatenate([np.array(observations).flatten(), dones, np.array(rewards), ])

    for current_step in range(num_steps):
        actions = action_sampler.sample_actions(dim=env.get_num_agents())
        obs, reward, done, info = env.step(actions)

        first_run_observations.append(state_repr(obs, reward, done, info))
        if all(done):
            break

    # resetting the environment to the initial state using backward steps
    for current_step in range(num_steps):
        if not env.step_back():
            break

    action_sampler = ActionsSampler(env.action_space.n, seed=seed)

    second_run_observations = []
    for current_step in range(num_steps):
        actions = action_sampler.sample_actions(dim=env.get_num_agents())
        obs, reward, done, info = env.step(actions)
        second_run_observations.append(state_repr(obs, reward, done, info))
        assert np.isclose(first_run_observations[current_step], second_run_observations[current_step]).all()
        if all(done):
            break
    assert np.isclose(first_run_observations, second_run_observations).all()


def test_steps_per_second_throughput():
    table = []
    for on_target in ['finish', 'nothing', 'restart']:
        for num_agents in [1, 32, 64]:
            for size in [32, 64]:
                gc = GridConfig(obs_radius=5, seed=42, max_episode_steps=1024, )
                gc.size = size
                gc.num_agents = num_agents
                gc.on_target = on_target

                start_time = time.monotonic()
                run_episode(grid_config=gc)
                end_time = time.monotonic()
                steps_per_second = gc.max_episode_steps / (end_time - start_time)
                table.append([on_target, num_agents, size, steps_per_second * gc.num_agents])
    print('\n' + tabulate(table, headers=['on_target', 'num_agents', 'size', 'SPS (individual)'], tablefmt='grid'))
