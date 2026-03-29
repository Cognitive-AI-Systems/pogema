import numpy as np
import pytest

from pogema import AStarAgent, BatchAStarAgent, GridConfig, pogema_v0


def test_astar_agent_basic_pathfinding():
    """AStarAgent should reach the target on a simple open map."""
    gc = GridConfig(
        map="....\n....\n....\n....",
        agents_xy=[[0, 0]],
        targets_xy=[[3, 3]],
        observation_type='POMAPF',
        on_target='finish',
        max_episode_steps=64,
    )
    env = pogema_v0(gc)
    obs, info = env.reset()

    agent = AStarAgent()
    for _ in range(64):
        action = agent.act(obs[0])
        obs, rewards, terminated, truncated, infos = env.step([action])
        if all(terminated) or all(truncated):
            break

    assert rewards[0] == 1.0


def test_astar_agent_clear_state():
    """clear_state() should allow reuse across episodes without error."""
    gc = GridConfig(
        map="....\n....\n....\n....",
        agents_xy=[[0, 0]],
        targets_xy=[[1, 1]],
        observation_type='POMAPF',
        on_target='finish',
        max_episode_steps=64,
    )
    env = pogema_v0(gc)

    agent = AStarAgent()

    for _episode in range(3):
        obs, info = env.reset()
        agent.clear_state()
        for _ in range(64):
            action = agent.act(obs[0])
            obs, rewards, terminated, truncated, infos = env.step([action])
            if all(terminated) or all(truncated):
                break


def test_astar_agent_detects_teleport():
    """AStarAgent should raise if the agent teleports between calls."""
    gc = GridConfig(
        map="......\n......\n......\n......\n......\n......",
        agents_xy=[[0, 0], [5, 5]],
        targets_xy=[[5, 5], [0, 0]],
        observation_type='POMAPF',
        on_target='finish',
        max_episode_steps=64,
    )
    env = pogema_v0(gc)
    obs, _ = env.reset()

    agent = AStarAgent()
    # Move agent 0 a few steps to build up distance from origin
    for _ in range(3):
        action = agent.act(obs[0])
        obs, _, terminated, truncated, _ = env.step([action, 0])
        if all(terminated) or all(truncated):
            break

    # Feed agent 1's observation (at relative (0,0)) — agent 0 has moved ~3 steps away
    with pytest.raises(IndexError, match="moved more than 1 step"):
        agent.act(obs[1])


def test_batch_astar_agent():
    """BatchAStarAgent should handle multiple agents and reach targets."""
    gc = GridConfig(
        map="......\n......\n......\n......\n......\n......",
        agents_xy=[[0, 0], [0, 5]],
        targets_xy=[[5, 5], [5, 0]],
        observation_type='POMAPF',
        on_target='finish',
        max_episode_steps=128,
    )
    env = pogema_v0(gc)
    obs, _ = env.reset()

    batch_agent = BatchAStarAgent()
    solved = False
    for _ in range(128):
        actions = batch_agent.act(obs)
        obs, rewards, terminated, truncated, infos = env.step(actions)
        if all(terminated) or all(truncated):
            solved = any(r > 0 for r in rewards)
            break

    assert solved


def test_batch_astar_agent_reset_states():
    """reset_states() should clear all agent state."""
    batch_agent = BatchAStarAgent()

    gc = GridConfig(
        map="....\n....\n....\n....",
        agents_xy=[[0, 0]],
        targets_xy=[[1, 1]],
        observation_type='POMAPF',
        on_target='finish',
        max_episode_steps=64,
    )
    env = pogema_v0(gc)

    for _episode in range(2):
        obs, _ = env.reset()
        batch_agent.reset_states()
        assert len(batch_agent.astar_agents) == 0
        for _ in range(64):
            actions = batch_agent.act(obs)
            obs, rewards, terminated, truncated, infos = env.step(actions)
            if all(terminated) or all(truncated):
                break


def test_grid_memory_expansion():
    """GridMemory should expand when updating with distant coordinates."""
    from pogema.a_star_policy import GridMemory

    gm = GridMemory(start_r=2)
    obstacles = np.zeros((5, 5), dtype=bool)
    # Update at the origin — should fit in initial memory
    gm.update(0, 0, obstacles)
    assert not gm.is_obstacle(0, 0)

    # Update far from center — should trigger expansion
    gm.update(100, 100, obstacles)
    assert not gm.is_obstacle(100, 100)
