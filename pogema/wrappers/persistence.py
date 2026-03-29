from pogema.wrappers.base import PogemaWrapper


class AgentState:
    def __init__(self, x, y, tx, ty, step, active):
        self.x = x
        self.y = y
        self.tx = tx
        self.ty = ty
        self.step = step
        self.active = active

    def get_xy(self):
        return self.x, self.y

    def get_target_xy(self):
        return self.tx, self.ty

    def is_active(self):
        return self.active

    def get_step(self):
        return self.step

    def __eq__(self, other):
        o = other
        return self.x == o.x and self.y == o.y and self.tx == o.tx and self.ty == o.ty and self.active == o.active

    def __str__(self):
        return str([self.x, self.y, self.tx, self.ty, self.step, self.active])


def agent_state_to_full_list(agent_states, num_steps):
    result = []
    current_state_id = 0
    # going over num_steps + 1, to handle last step
    for episode_step in range(num_steps + 1):
        if current_state_id < len(agent_states) - 1 and agent_states[current_state_id + 1].step == episode_step:
            current_state_id += 1
        result.append(agent_states[current_state_id])
    return result


def decompress_history(history):
    max_steps = max([agent_states[-1].step for agent_states in history])
    result = [agent_state_to_full_list(agent_states, max_steps) for agent_states in history]
    return result


class PersistentWrapper(PogemaWrapper):
    def __init__(self, env, xy_offset=None):
        super().__init__(env)
        self._step = None
        self._agent_states = None
        self._xy_offset = xy_offset

    def step(self, action):
        result = self.env.step(action)
        self._step += 1
        for agent_idx in range(self.unwrapped.get_num_agents()):
            agent_state = self._get_agent_state(self.unwrapped.grid, agent_idx)
            if agent_state != self._agent_states[agent_idx][-1]:
                self._agent_states[agent_idx].append(agent_state)

        return result

    def step_back(self):
        if self._step <= 0:
            return False
        self._step -= 1
        self.env.set_elapsed_steps(self._step)
        for idx in reversed(range(self.unwrapped.get_num_agents())):

            if self._step < self._agent_states[idx][-1].step:
                self._agent_states[idx].pop()
                state = self._agent_states[idx][-1]

                if state.active:
                    self.unwrapped.grid.show_agent(idx)
                else:
                    self.unwrapped.grid.hide_agent(idx)
                self.unwrapped.grid.move_agent_to_cell(idx, state.x, state.y)
                self.unwrapped.grid.finishes_xy[idx] = state.tx, state.ty

        return True

    def _get_agent_state(self, grid, agent_idx):
        x, y = grid.positions_xy[agent_idx]
        tx, ty = grid.finishes_xy[agent_idx]
        active = grid.is_active[agent_idx]
        if self._xy_offset:
            x += self._xy_offset
            y += self._xy_offset
            tx += self._xy_offset
            ty += self._xy_offset
        return AgentState(x, y, tx, ty, self._step, active)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)

        self._step = 0

        self._agent_states = []
        for agent_idx in range(self.unwrapped.get_num_agents()):
            self._agent_states.append([self._get_agent_state(self.unwrapped.grid, agent_idx)])

        return result

    def get_full_history(self):
        return [agent_state_to_full_list(agent_states, self._step) for agent_states in self._agent_states]

    def get_history(self):
        return self._agent_states
