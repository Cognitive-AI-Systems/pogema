import numpy as np

from pogema import GridConfig
from pogema.integrations.make_pogema import pogema_v0


def test_gym_creation():
    import gymnasium

    env = gymnasium.make("Pogema-v0", grid_config=GridConfig(integration='gymnasium'))
    env.reset()


def test_integrations():
    for integration in ['SampleFactory', 'gymnasium', "PettingZoo", None]:
        env = pogema_v0(grid_config=GridConfig(integration=integration))
        env.reset()


def test_sample_factory_integration():
    env = pogema_v0(GridConfig(seed=7, num_agents=4, size=12, integration='SampleFactory'))
    env.reset()

    assert env.unwrapped.get_num_agents() == 4
    assert env.env.is_multiagent is True

    # testing auto-reset wrapper
    for _ in range(2):

        while True:
            _, _, terminated, truncated, infos = env.step(env.unwrapped.sample_actions())
            if all(terminated) or all(truncated):
                break

        assert np.isclose(infos[0]['episode_extra_stats']['ISR'], 0.0)
        assert np.isclose(infos[0]['episode_extra_stats']['CSR'], 0.0)


def test_single_agent_gym_integration():
    gc = GridConfig(seed=7, num_agents=1, integration='gymnasium')
    env = pogema_v0(gc)

    obs, info = env.reset()

    assert obs.shape == env.observation_space.shape
    done = False

    cnt = 0
    while True:
        assert cnt < gc.max_episode_steps
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        cnt += 1
        if terminated or truncated:
            break


def test_petting_zoo():
    from pettingzoo.test import api_test, parallel_api_test

    gc = GridConfig(num_agents=16, size=16, integration='PettingZoo')

    parallel_api_test(pogema_v0(gc), num_cycles=1000)

    try:
        from pettingzoo.utils import parallel_to_aec

        def env(grid_config: GridConfig = GridConfig(num_agents=20, size=16)):
            return parallel_to_aec(pogema_v0(grid_config))

        api_test(env(gc), num_cycles=1000, verbose_progress=True)
    except ImportError:
        pass
