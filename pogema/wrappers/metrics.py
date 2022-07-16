import gym


class MetricsWrapper(gym.Wrapper):
    def __init__(self, env, group_name='metrics'):
        super().__init__(env)
        self._ISR = None
        self._group_name = group_name

    def update_group_name(self, group_name):
        self._group_name = group_name

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)

        for agent_idx in range(self.env.get_num_agents()):
            infos[agent_idx][self._group_name] = infos[agent_idx].get(self._group_name, {})

            if done[agent_idx]:
                infos[agent_idx][self._group_name].update(Done=True)
                if agent_idx not in self._ISR:
                    self._ISR[agent_idx] = float('TimeLimit.truncated' not in infos[agent_idx])

        if all(done):
            not_tl_truncated = all(['TimeLimit.truncated' not in info for info in infos])
            infos[0][self._group_name].update(CSR=float(not_tl_truncated))

            for agent_idx in range(self.env.get_num_agents()):
                infos[agent_idx][self._group_name].update(ISR=self._ISR[agent_idx])

        return obs, reward, done, infos

    def reset(self, **kwargs):
        self._ISR = {}
        return self.env.reset(**kwargs)


class MetricsWrapperLifeLong(gym.Wrapper):
    def __init__(self, env, group_name='metrics_life_long'):
        super().__init__(env)
        self._AchievedGoals = None
        self._LastGoal = None
        self._group_name = group_name
        self._num_agents = self.env.get_num_agents()
        self._step: int = 0

    def update_group_name(self, group_name):
        self._group_name = group_name

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)

        self._step += 1
        for agent_idx in range(self._num_agents):
            infos[agent_idx][self._group_name] = infos[agent_idx].get(self._group_name, {})

            if done[agent_idx]:
                infos[agent_idx][self._group_name].update(Done=True)
                self._AchievedGoals[agent_idx] += float('TimeLimit.truncated' not in infos[agent_idx])
                self._LastGoal[agent_idx] = self._step

        if all(done):
            infos[0][self._group_name].update(MinimumAchieved=min(self._AchievedGoals))

            for agent_idx in range(self._num_agents):
                infos[agent_idx][self._group_name].update(AchievedGoals=self._AchievedGoals[agent_idx])
                if self._AchievedGoals[agent_idx]:
                    infos[agent_idx][self._group_name].\
                        update(AverageTimeToGoal=self._LastGoal[agent_idx] / self._AchievedGoals[agent_idx])

        return obs, reward, done, infos

    def reset(self, **kwargs):
        self._AchievedGoals = [0] * self._num_agents
        self._LastGoal = [0] * self._num_agents
        return self.env.reset(**kwargs)
