import numpy as np
from pogema.grid import GridConfig
from pogema.integrations.make_pogema import make_pogema
import gym


class ActionMapping:
    noop: int = 0
    up: int = 1
    down: int = 2
    left: int = 3
    right: int = 4


def test_non_dis():
    grid = """
    ......#....
    ...####....
    ...........
    ...........
    ...........
    ...........
    ...........
    ...........
    ...........
    ...........
    """
    gc = GridConfig(map=grid, num_agents=2, agents_xy=[(0,0), (0,1)], targets_xy=[(0,5), (0,4)], disappear_on_goal=False)
    env = gym.make("Pogema-v0", grid_config=gc)
    ac = ActionMapping()
    obs = env.reset()
    env.step([ac.right, ac.right])
    env.step([ac.right, ac.right])
    env.step([ac.right, ac.right])
    env.step([ac.right, ac.noop])
    obs, reward, done, infos = env.step([ac.right, ac.noop])
    print(reward, done)
    assert np.isclose([0.0, 0.0], reward).all()
    assert np.isclose([False, False], done).all()


def test_moving_non_disapeaer():
    env = make_pogema(GridConfig(num_agents=2, size=6, obs_radius=2, density=0.3, seed=42, disappear_on_goal=False))
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
    assert np.isclose([0.0, 0.0], reward).all()
    assert np.isclose([False, False], done).all()

def run_episode(grid_config):
    env = make_pogema(grid_config)
    env.reset()

    obs, rewards, dones, infos = env.reset(), [None], [False], [None]

    results = [[obs, rewards, dones, infos]]
    while not all(dones):
        results.append(env.step(env.sample_actions()))
        dones = results[-1][-2]
    return results

def test_CoopRewardWrapper():
    env = env = make_pogema(GridConfig(num_agents=2, size=2, disappear_on_goal=False, max_episode_steps=3, density=0.0, agents_xy=[(0,0), (0,1)], targets_xy=[(1,0), (1,1)]))
    ac = ActionMapping()
    env.reset()

    obs, reward, done, infos = env.step([ac.down, ac.down])
    assert np.isclose([0.0, 0.0], reward).all()
    assert np.isclose([False, False], done).all() 

    obs, reward, done, infos = env.step([ac.down, ac.down])
    assert np.isclose([0.0, 0.0], reward).all()
    assert np.isclose([False, False], done).all()
    
    obs, reward, done, infos = env.step([ac.up, ac.up])
    assert np.isclose([1.0, 1.0], reward).all()
    assert np.isclose([True, True], done).all()
