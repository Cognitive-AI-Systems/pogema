import numpy as np
import pytest

from pogema import GridConfig, pogema_v0


def make_env(positions, *, walls=(), on_target='nothing', targets=None):
    grid_map = np.zeros((7, 7), dtype=int)
    for cell in walls:
        grid_map[cell] = 1
    if targets is None:
        targets = [(0, agent_id) for agent_id in range(len(positions))]
    env = pogema_v0(GridConfig(
        map=grid_map.tolist(),
        agents_xy=positions,
        targets_xy=targets,
        obs_radius=2,
        collision_system='soft',
        observation_type='POMAPF',
        on_target=on_target,
        seed=42,
    ))
    env.reset()
    return env


def assert_occupancy(env, observations):
    grid = env.unwrapped.grid
    expected = np.full_like(grid.positions, grid.config.FREE)
    for agent_id, cell in enumerate(grid.positions_xy):
        if grid.is_active[agent_id]:
            expected[cell] = grid.config.OBSTACLE
    np.testing.assert_array_equal(grid.positions, expected)
    radius = grid.config.obs_radius
    for agent_id, (x, y) in enumerate(grid.positions_xy):
        np.testing.assert_array_equal(
            observations[agent_id]['agents'],
            expected[x - radius:x + radius + 1, y - radius:y + radius + 1],
        )


@pytest.mark.parametrize('reverse_order', [False, True], ids=['follower-first', 'leader-first'])
def test_soft_following_occupancy(reverse_order):
    positions = [(3, 1), (3, 2), (3, 3)]
    if reverse_order:
        positions.reverse()
    env = make_env(positions)
    for step in range(1, 3):
        observations, *_ = env.step([4, 4, 4])
        np.testing.assert_array_equal(
            env.unwrapped.get_agents_xy(ignore_borders=True), [(x, y + step) for x, y in positions],
        )
        assert_occupancy(env, observations)
    env.close()


def test_soft_rotation_occupancy():
    env = make_env([(2, 2), (2, 3), (3, 3), (3, 2)])
    observations, *_ = env.step([4, 2, 3, 1])
    np.testing.assert_array_equal(
        env.unwrapped.get_agents_xy(ignore_borders=True), [(2, 3), (3, 3), (3, 2), (2, 2)],
    )
    assert_occupancy(env, observations)
    env.close()


@pytest.mark.parametrize(
    'positions,actions,walls,expected',
    [
        ([(3, 1), (3, 3)], [4, 3], [], [(3, 2), (3, 3)]),
        ([(3, 1), (3, 2)], [4, 3], [], [(3, 1), (3, 2)]),
        ([(3, 1), (3, 2), (3, 3)], [4, 4, 4], [(3, 4)], [(3, 1), (3, 2), (3, 3)]),
        ([(3, 1), (3, 2), (3, 3)], [4, 4, 0], [], [(3, 1), (3, 2), (3, 3)]),
    ],
    ids=['vertex-conflict', 'swap', 'following-into-wall', 'following-staying-agent'],
)
def test_soft_collision_outcomes(positions, actions, walls, expected):
    env = make_env(positions, walls=walls)
    observations, *_ = env.step(actions.copy())
    np.testing.assert_array_equal(env.unwrapped.get_agents_xy(ignore_borders=True), expected)
    assert_occupancy(env, observations)
    env.close()


def test_soft_occupancy_excludes_inactive_agents():
    env = make_env([(1, 1), (3, 1), (3, 2), (3, 3)])
    env.unwrapped.grid.hide_agent(0)
    observations, *_ = env.step([0, 4, 4, 4])
    np.testing.assert_array_equal(
        env.unwrapped.get_agents_xy(ignore_borders=True), [(1, 1), (3, 2), (3, 3), (3, 4)],
    )
    assert_occupancy(env, observations)
    env.close()


def test_soft_occupancy_after_agent_finishes():
    env = make_env([(3, 1), (3, 2)], on_target='finish', targets=[(0, 0), (3, 3)])
    observations, _, terminated, *_ = env.step([4, 4])
    assert terminated == [False, True]
    assert_occupancy(env, observations)
    observations, *_ = env.step([4, 0])
    np.testing.assert_array_equal(env.unwrapped.get_agents_xy(ignore_borders=True), [(3, 3), (3, 3)])
    assert_occupancy(env, observations)
    env.close()


@pytest.mark.parametrize('collision_system', ['soft', 'priority', 'block_both'])
@pytest.mark.parametrize('seed', [0, 42, 123])
def test_occupancy_during_random_moves(collision_system, seed):
    env = pogema_v0(GridConfig(
        size=10, num_agents=12, density=0.15, obs_radius=2, seed=seed,
        collision_system=collision_system, observation_type='POMAPF', on_target='nothing',
        max_episode_steps=64,
    ))
    observations, _ = env.reset()
    assert_occupancy(env, observations)
    rng = np.random.default_rng(seed)
    for _ in range(64):
        observations, *_ = env.step(rng.integers(0, 5, size=12).tolist())
        assert_occupancy(env, observations)
    env.close()
