import numpy as np
import gym
from gym.error import ResetNeeded

from pogema.grid import Grid, CooperativeGrid
from pogema.grid_config import GridConfig
from pogema.wrappers.metrics import MetricsWrapper
from pogema.wrappers.multi_time_limit import MultiTimeLimit, CoopRewardWrapper


class ActionsSampler:
    def __init__(self, num_actions, seed=42):
        self._num_actions = num_actions
        self._rnd = None
        self.update_seed(seed)

    def update_seed(self, seed=None):
        self._rnd = np.random.default_rng(seed)

    def sample_actions(self, dim=1):
        return self._rnd.integers(self._num_actions, size=dim)


class PogemaBase(gym.Env):

    def step(self, action):
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError

    def __init__(self, config: GridConfig = GridConfig()):
        # noinspection PyTypeChecker
        self.grid: Grid = None
        self.config = config

        full_size = self.config.obs_radius * 2 + 1
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3, full_size, full_size))
        self.action_space = gym.spaces.Discrete(len(self.config.MOVES))
        self._multi_action_sampler = ActionsSampler(self.action_space.n, seed=self.config.seed)

    def _get_agents_obs(self, agent_id=0):
        return np.concatenate([
            self.grid.get_obstacles_for_agent(agent_id)[None],
            self.grid.get_positions(agent_id)[None],
            self.grid.get_square_target(agent_id)[None]
        ])

    def check_reset(self):
        if self.grid is None:
            raise ResetNeeded("Please reset environment first!")

    def render(self, mode='human'):
        self.check_reset()
        return self.grid.render(mode=mode)

    def sample_actions(self):
        return self._multi_action_sampler.sample_actions(dim=self.config.num_agents)

    def get_num_agents(self):
        return self.config.num_agents


class PogemaCoopFinish(PogemaBase):
    def __init__(self, config=GridConfig(num_agents=2)):
        super().__init__(config)
        self.num_agents = self.config.num_agents
        self.is_multiagent = True
        self.active = None

    def _obs(self):
        return [self._get_agents_obs(index) for index in range(self.config.num_agents)]

    def step(self, action: list):
        assert len(action) == self.config.num_agents
        rewards = []

        infos = [dict() for _ in range(self.config.num_agents)]

        dones = [False for _ in range(self.config.num_agents)]
        
        for agent_idx in range(self.config.num_agents):
            #A way to refactor:
            agent_done = self.grid.move(agent_idx, action[agent_idx])

            if agent_done:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        obs = self._obs()
        return obs, rewards, dones, infos

    def reset(self):
        self.grid: CooperativeGrid = CooperativeGrid(grid_config=self.config)
        self.active = {agent_idx: True for agent_idx in range(self.config.num_agents)}
        return self._obs()

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_agents_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_targets_xy(only_active=only_active, ignore_borders=ignore_borders)


class Pogema(PogemaBase):
    def __init__(self, config=GridConfig(num_agents=2)):
        super().__init__(config)
        self.active = None

    def _obs(self):
        return [self._get_agents_obs(index) for index in range(self.config.num_agents)]

    def step(self, action: list):
        assert len(action) == self.config.num_agents
        rewards = []

        infos = [dict() for _ in range(self.config.num_agents)]

        dones = []
        for agent_idx in range(self.config.num_agents):
            if self.active[agent_idx]:
                self.grid.move(agent_idx, action[agent_idx])

            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            dones.append(on_goal)

        for agent_idx in range(self.config.num_agents):
            if self.grid.on_goal(agent_idx):
                self.grid.hide_agent(agent_idx)
                self.active[agent_idx] = False

            infos[agent_idx]['is_active'] = self.active[agent_idx]

        obs = self._obs()
        return obs, rewards, dones, infos

    def reset(self, **kwargs):
        self.grid: Grid = Grid(grid_config=self.config)
        self.active = {agent_idx: True for agent_idx in range(self.config.num_agents)}
        return self._obs()

    def get_agents_xy_relative(self):
        return self.grid.get_agents_xy_relative()

    def get_targets_xy_relative(self):
        return self.grid.get_targets_xy_relative()

    def get_obstacles(self, ignore_borders=False):
        return self.grid.get_obstacles(ignore_borders=ignore_borders)

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_agents_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_targets_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_state(self, ignore_borders=False, as_dict=False):
        return self.grid.get_state(ignore_borders=ignore_borders, as_dict=as_dict)


def _make_pogema(grid_config):
    if grid_config.disappear_on_goal:
        env = Pogema(config=grid_config)
        env = MultiTimeLimit(env, grid_config.max_episode_steps)
    else:
        env = PogemaCoopFinish(config=grid_config)
        env = MultiTimeLimit(env, grid_config.max_episode_steps)
        env = CoopRewardWrapper(env)
    env = MetricsWrapper(env)

    return env
