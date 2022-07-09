from functools import reduce

import numpy as np
import pettingzoo
from pogema import GridConfig
from pogema.integrations.make_pogema import make_pogema


def test_gym_creation():
    import gym

    for integration in ['SampleFactory', 'PyMARL', 'gym', "PettingZoo", None]:
        env = gym.make("Pogema-v0", integration=integration)
        env.reset()

    for integration in ['SampleFactory', 'PyMARL', 'gym', "PettingZoo", None]:
        env = gym.make("Pogema-8x8-easy-v0", integration=integration)
        env.reset()

    for integration in ['SampleFactory', 'PyMARL', "PettingZoo", None]:
        env = gym.make("Pogema-16x16-hard-v0", integration=integration)
        env.reset()


def test_sample_factory_integration():
    env = make_pogema(GridConfig(seed=7, num_agents=4, size=10, integration='SampleFactory'))
    env.reset()

    assert env.num_agents == 4
    assert env.is_multiagent is True

    # testing auto-reset wrapper
    for _ in range(2):
        dones = [False]
        infos = None
        while not all(dones):
            _, _, dones, infos = env.step(env.sample_actions())

        assert np.isclose(infos[0]['episode_extra_stats']['ISR'], 0.0)
        assert np.isclose(infos[1]['episode_extra_stats']['ISR'], 0.0)
        assert np.isclose(infos[0]['episode_extra_stats']['CSR'], 0.0)


def test_pymarl_integration():
    gc = GridConfig(seed=7, num_agents=4, obs_radius=3, max_episode_steps=16, integration='PyMARL')
    env = make_pogema(gc)

    _state = [0.14285714285714285, 1.0, 1.0, 0.5714285714285714, 0.42857142857142855, 0.7142857142857143,
              0.8571428571428571, 0.2857142857142857, 0.8571428571428571, 0.42857142857142855, 0.42857142857142855, 1.0,
              0.5714285714285714, 0.7142857142857143, 0.14285714285714285, 0.42857142857142855, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
              1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
              0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
              0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0]
    assert np.isclose(_state, env.get_state()).all()

    assert env.episode_limit == 16
    assert env.get_env_info()['state_shape'] == 212
    assert env.get_env_info()['obs_shape'] == 147
    assert env.get_env_info()['n_agents'] == 4
    assert env.get_env_info()['episode_limit'] == 16

    num_agents, dimension = env.get_obs().shape
    assert num_agents == gc.num_agents
    assert dimension == reduce(lambda a, b: a * b, env.env.observation_space.shape)
    assert dimension == env.get_obs_size()
    assert env.get_state_size() == env.get_state().shape[0]

    done = False
    cnt = 0
    while not done:
        assert cnt < gc.max_episode_steps
        _, done, _ = env.step(env.sample_actions())
        cnt += 1


def test_single_agent_gym_integration():
    gc = GridConfig(seed=7, num_agents=1, integration='gym')
    env = make_pogema(gc)

    obs = env.reset()

    assert obs.shape == env.observation_space.shape
    done = False

    cnt = 0
    while not done:
        assert cnt < gc.max_episode_steps
        obs, reward, done, info = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        cnt += 1


def test_petting_zoo():
    from pettingzoo.test import api_test, parallel_api_test, render_test

    gc = GridConfig(num_agents=16, size=16, integration='PettingZoo')

    parallel_api_test(make_pogema(gc), num_cycles=1000)

    try:
        from pettingzoo.utils import parallel_to_aec

        def env(grid_config: GridConfig = GridConfig(num_agents=20, size=16)):
            return parallel_to_aec(make_pogema(grid_config))

        api_test(env(gc), num_cycles=1000, verbose_progress=True)
        render_test(env)
    except ImportError:
        pass
