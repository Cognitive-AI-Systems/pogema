import numpy as np
import gym
from gym.error import ResetNeeded

from pogema.grid import Grid, GridLifeLong, CooperativeGrid
from pogema.grid_config import GridConfig
from pogema.wrappers.metrics import MetricsWrapper, MetricsWrapperLifeLong
from pogema.wrappers.multi_time_limit import MultiTimeLimit, CoopRewardWrapper
from pogema.generator import generate_new_target


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

        dones = []
        
        for agent_idx in range(self.config.num_agents):
            #A way to refactor:
            agent_done = self.grid.move(agent_idx, action[agent_idx])
            on_goal = self.grid.on_goal(agent_idx)
            if agent_done:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            dones.append(on_goal)
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


class PogemaLifeLong(PogemaBase):
    def __init__(self, config=GridConfig(num_agents=2)):
        super().__init__(config)
        self.active = None
        self.random_generators: list = [np.random.default_rng(config.seed + i) for i in range(config.num_agents)]
        self._steps_after_new_target = [0] * self.config.num_agents
        if self.config.steps_before_renew_target is None:
            self.config.steps_before_renew_target = self.config.max_episode_steps

    def _obs(self):
        return [self._get_agents_obs(index) for index in range(self.config.num_agents)]

    def step(self, action: list):
        assert len(action) == self.config.num_agents
        rewards = []

        infos = [dict() for _ in range(self.config.num_agents)]

        dones = [False] * self.config.num_agents
        for agent_idx in range(self.config.num_agents):
            self._steps_after_new_target[agent_idx] += 1

            if self.active[agent_idx]:
                self.grid.move(agent_idx, action[agent_idx])

            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.active[agent_idx]:
                dones[agent_idx] = True
                rewards.append(1.0)
            else:
                rewards.append(0.0)

            if self._steps_after_new_target[agent_idx] == self.config.steps_before_renew_target or \
                    self.grid.on_goal(agent_idx):
                self.grid.finishes_xy[agent_idx] = generate_new_target(self.random_generators[agent_idx],
                                                                       self.grid.point_to_component,
                                                                       self.grid.component_to_points,
                                                                       self.grid.positions_xy[agent_idx])
                self._steps_after_new_target[agent_idx] = 0

            infos[agent_idx]['is_active'] = self.active[agent_idx]

        obs = self._obs()
        return obs, rewards, dones, infos

    def reset(self, **kwargs):
        self.grid: Grid = GridLifeLong(grid_config=self.config)
        self.active = {agent_idx: True for agent_idx in range(self.config.num_agents)}
        self._steps_after_new_target = [0] * self.config.num_agents
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
    if grid_config.pogema_type == 'life_long':
        env = PogemaLifeLong(config=grid_config)
    elif grid_config.pogema_type == 'non_disappearing':
        grid_config.disappear_on_goal = False
        env = PogemaCoopFinish(config=grid_config)
    else:
        env = Pogema(config=grid_config)
    env = MultiTimeLimit(env, grid_config.max_episode_steps)
    if grid_config.pogema_type == 'life_long':
        env = MetricsWrapperLifeLong(env)
    elif grid_config.pogema_type == 'non_disappearing':
        env = CoopRewardWrapper(env, grid_config.max_episode_steps)
        env = MetricsWrapper(env)
    else:
        env = MetricsWrapper(env)

    return env
