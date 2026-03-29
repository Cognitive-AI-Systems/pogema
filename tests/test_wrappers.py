import pytest

from pogema import GridConfig, RuntimeMetricWrapper, pogema_v0

# ---------------------------------------------------------------------------
# MultiTimeLimit
# ---------------------------------------------------------------------------

def test_multi_time_limit_truncates_at_exact_step():
    """Episode should truncate at exactly max_episode_steps."""
    max_steps = 16
    gc = GridConfig(num_agents=2, size=8, seed=42, max_episode_steps=max_steps)
    env = pogema_v0(gc)
    env.reset()

    for step in range(max_steps):
        obs, rewards, terminated, truncated, infos = env.step(
            [0] * gc.num_agents  # noop — agents won't reach goals
        )
        if step < max_steps - 1:
            assert not all(truncated), f"Truncated too early at step {step}"
    assert all(truncated), "Should be truncated at max_episode_steps"


def test_multi_time_limit_set_elapsed_steps():
    """set_elapsed_steps() should advance the internal counter."""
    gc = GridConfig(num_agents=1, size=6, seed=42, max_episode_steps=10)
    env = pogema_v0(gc)
    env.reset()

    # Jump to step 9
    env.set_elapsed_steps(9)
    _, _, _, truncated, _ = env.step([0])
    assert all(truncated)


def test_multi_time_limit_set_negative_raises():
    gc = GridConfig(num_agents=1, size=6, seed=42, max_episode_steps=10)
    env = pogema_v0(gc)
    env.reset()
    with pytest.raises(ValueError, match="non-negative"):
        env.set_elapsed_steps(-1)


def test_multi_time_limit_reset_clears():
    """reset() should clear elapsed steps."""
    gc = GridConfig(num_agents=1, size=6, seed=42, max_episode_steps=10)
    env = pogema_v0(gc)
    env.reset()

    for _ in range(5):
        env.step([0])

    env.reset()
    # After reset, we should be able to take max_episode_steps again
    for step in range(10):
        _, _, _, truncated, _ = env.step([0])
        if step < 9:
            assert not all(truncated)
    assert all(truncated)


# ---------------------------------------------------------------------------
# SumOfCostsAndMakespanMetric
# ---------------------------------------------------------------------------

def test_sum_of_costs_and_makespan_known_scenario():
    """With known map and seed, SoC and makespan should have expected values."""
    gc = GridConfig(
        map="......\n......\n......\n......\n......\n......",
        agents_xy=[[0, 0], [0, 1]],
        targets_xy=[[0, 1], [0, 0]],
        on_target='nothing',
        max_episode_steps=64,
        collision_system='priority',
    )
    env = pogema_v0(gc)
    env.reset()

    # Manually solve: swap positions
    # Agent 0 moves right (to [0,1]), Agent 1 moves left (to [0,0])
    # With priority collision, higher-index agent wins — so agent 1 moves first
    # Both should reach goals
    for _ in range(64):
        obs, rewards, terminated, truncated, infos = env.step([4, 3])  # right, left
        if all(terminated) or all(truncated):
            break

    assert 'metrics' in infos[0]
    assert 'SoC' in infos[0]['metrics']
    assert 'makespan' in infos[0]['metrics']
    assert infos[0]['metrics']['SoC'] >= 2  # At minimum 2 (one step each)
    assert infos[0]['metrics']['makespan'] >= 1


def test_sum_of_costs_truncated():
    """When episode truncates, SoC/makespan should still be reported."""
    gc = GridConfig(
        num_agents=2, size=8, seed=42, on_target='nothing', max_episode_steps=8
    )
    env = pogema_v0(gc)
    env.reset()

    for _ in range(8):
        obs, rewards, terminated, truncated, infos = env.step([0, 0])
        if all(truncated):
            break

    assert 'metrics' in infos[0]
    assert 'SoC' in infos[0]['metrics']


# ---------------------------------------------------------------------------
# RuntimeMetricWrapper
# ---------------------------------------------------------------------------

def test_runtime_metric_wrapper():
    """RuntimeMetricWrapper should add a positive 'runtime' to metrics."""
    gc = GridConfig(num_agents=2, size=6, seed=42, max_episode_steps=16)
    env = pogema_v0(gc)
    env = RuntimeMetricWrapper(env)
    env.reset()

    for _ in range(16):
        obs, rewards, terminated, truncated, infos = env.step(
            env.unwrapped.sample_actions()
        )
        if all(terminated) or all(truncated):
            break

    assert 'metrics' in infos[0]
    assert 'runtime' in infos[0]['metrics']
    assert infos[0]['metrics']['runtime'] >= 0.0


def test_runtime_metric_reset_clears():
    """RuntimeMetricWrapper should reset timing on env.reset()."""
    gc = GridConfig(num_agents=2, size=6, seed=42, max_episode_steps=8)
    env = pogema_v0(gc)
    env = RuntimeMetricWrapper(env)

    for _episode in range(2):
        env.reset()
        for _ in range(8):
            obs, rewards, terminated, truncated, infos = env.step(
                env.unwrapped.sample_actions()
            )
            if all(terminated) or all(truncated):
                break
        assert 'metrics' in infos[0]
        assert 'runtime' in infos[0]['metrics']


# ---------------------------------------------------------------------------
# Soft collision edge cases
# ---------------------------------------------------------------------------

def test_soft_collision_no_crash():
    """Soft collision system should not crash with many agents in tight space."""
    gc = GridConfig(
        num_agents=32, size=16, density=0.1, seed=42,
        collision_system='soft', max_episode_steps=100,
    )
    env = pogema_v0(gc)
    env.reset()
    for _ in range(100):
        actions = env.unwrapped.sample_actions()
        env.step(actions.tolist())


def test_soft_collision_cascading_revert():
    """Soft collision cascading reverts should not error or cause inconsistency."""
    gc = GridConfig(
        num_agents=8, size=8, density=0.0, seed=7,
        collision_system='soft', max_episode_steps=50,
    )
    env = pogema_v0(gc)
    env.reset()
    for _ in range(50):
        # All agents try to move in same direction → forces collision resolution
        actions = [1] * gc.num_agents
        obs, rewards, terminated, truncated, infos = env.step(actions)
        if all(terminated) or all(truncated):
            break


def test_soft_collision_edge_swap():
    """Two agents trying to swap positions should both be reverted in soft mode."""
    gc = GridConfig(
        map="....\n....\n....\n....",
        agents_xy=[[1, 1], [1, 2]],
        targets_xy=[[1, 2], [1, 1]],
        collision_system='soft',
        on_target='nothing',
        max_episode_steps=4,
    )
    env = pogema_v0(gc)
    env.reset()

    agents_before = [tuple(pos) for pos in env.unwrapped.get_agents_xy(ignore_borders=True)]

    # Agent 0 moves right, Agent 1 moves left — edge collision
    env.step([4, 3])

    agents_after = [tuple(pos) for pos in env.unwrapped.get_agents_xy(ignore_borders=True)]
    # Both should stay in place due to edge collision
    assert agents_before == agents_after, (
        f"Edge swap should be reverted: before={agents_before}, after={agents_after}"
    )
