import gym


class AgentState:
    def __init__(self, x, y, tx, ty, step, active):
        self.x = x
        self.y = y
        self.tx = tx
        self.ty = ty
        self.step = step
        self.active = active

    def __eq__(self, other):
        o = other
        return self.x == o.x and self.y == o.y and self.tx == o.tx and self.ty == o.ty and self.active == o.active


class PersistentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._step = None
        self._agent_states = None

    def step(self, action):
        result = self.env.step(action)
        self._step += 1
        for agent_idx in range(self.get_num_agents()):
            agent_state = self._get_agent_state(self.grid, agent_idx)
            if agent_state != self._agent_states[agent_idx][-1]:
                self._agent_states[agent_idx].append(agent_state)

        return result

    def step_back(self):
        if self._step <= 0:
            return False
        self._step -= 1
        self.set_elapsed_steps(self._step)
        for idx in reversed(range(self.get_num_agents())):

            if self._step < self._agent_states[idx][-1].step:
                self._agent_states[idx].pop()
                state = self._agent_states[idx][-1]

                if state.active:
                    self.grid.show_agent(idx)
                else:
                    self.grid.hide_agent(idx)
                self.grid.move_agent_to_cell(idx, state.x, state.y)
                self.grid.finishes_xy[idx] = state.tx, state.ty

        return True
        # self.env.grid.set_activity(agent_idx, a_s[agent_idx][-1].active)

    def _get_agent_state(self, grid, agent_idx):
        x, y = grid.positions_xy[agent_idx]
        tx, ty = grid.finishes_xy[agent_idx]
        active = grid.is_active[agent_idx]
        return AgentState(x, y, tx, ty, self._step, active)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)

        self._step = 0

        self._agent_states = []
        for agent_idx in range(self.get_num_agents()):
            self._agent_states.append([self._get_agent_state(self.grid, agent_idx)])

        return result
