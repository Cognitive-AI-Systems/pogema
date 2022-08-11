import numpy as np
from heapq import heappop, heappush
from pogema import GridConfig, pogema_v0, grid_config
from pogema.animation import AnimationMonitor

INF = 1000000007


class Node:
    def __init__(self, coord=(INF, INF), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f or ((self.f == other.f) and (self.g < other.g))


class AStar:
    def __init__(self):
        self.start = (0, 0)
        self.goal = (0, 0)
        self.max_steps = 500
        self.OPEN = list()
        self.CLOSED = dict()
        self.obstacles = set()
        self.other_agents = set()
        self.best_node = Node(self.start, 0, self.h(self.start))

    def h(self, node):
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def get_neighbours(self, u):
        neighbors = []
        for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (u[0] + d[0], u[1] + d[1]) not in self.obstacles:
                neighbors.append((u[0] + d[0], u[1] + d[1]))
        return neighbors

    def compute_shortest_path(self):
        u = Node()
        steps = 0
        while len(self.OPEN) > 0 and steps < self.max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            if self.best_node.h > u.h:
                self.best_node = u
            steps += 1
            for n in self.get_neighbours((u.i, u.j)):
                if n not in self.CLOSED and n not in self.other_agents:
                    heappush(self.OPEN, Node(n, u.g + 1, self.h(n)))
                    self.CLOSED[n] = (u.i, u.j)

    def get_next_node(self):
        next_node = self.start
        if self.goal in self.CLOSED:
            next_node = self.goal
            while self.CLOSED[next_node] != self.start:
                next_node = self.CLOSED[next_node]
        return next_node

    def update_obstacles(self, obs, other_agents):
        obstacles = np.nonzero(obs)
        self.obstacles.clear()
        for k in range(len(obstacles[0])):
            self.obstacles.add((obstacles[0][k], obstacles[1][k]))
        self.other_agents.clear()
        agents = np.nonzero(other_agents)
        for k in range(len(agents[0])):
            self.other_agents.add((agents[0][k], agents[1][k]))

    def reset(self):
        self.CLOSED = dict()
        self.OPEN = list()
        heappush(self.OPEN, Node(self.start, 0, self.h(self.start)))
        self.best_node = Node(self.start, 0, self.h(self.start))

    def update_path(self, start, goal):
        self.start = start
        self.goal = goal
        self.reset()
        self.compute_shortest_path()


class DeterministicPolicy:
    def __init__(self, random_seed=42, random_rate=0.2):
        self.agents = None
        self.actions = {tuple(GridConfig().MOVES[i]): i for i in
                        range(len(GridConfig().MOVES))}  # make a dictionary to translate coordinates of action into id
        self.obs_radius = GridConfig().obs_radius
        self._rnd = np.random.RandomState(random_seed)
        self._random_rate = random_rate

    def get_goal(self, obs):
        goal = np.nonzero(obs[2])
        goal_i = goal[0][0]
        goal_j = goal[1][0]
        if obs[0][goal_i][goal_j]:
            goal_i -= 1 if goal_i == 0 else 0
            goal_i += 1 if goal_i == self.obs_radius * 2 else 0
            goal_j -= 1 if goal_j == 0 else 0
            goal_j += 1 if goal_j == self.obs_radius * 2 else 0
        return goal_i, goal_j

    def act(self, obs, dones) -> list:
        if self.agents is None:
            self.obs_radius = len(obs[0][0]) // 2
            self.agents = [AStar() for _ in range(len(obs))]  # create a planner for each of the agents
        actions = []
        for k in range(len(obs)):
            start = (self.obs_radius, self.obs_radius)
            goal = self.get_goal(obs[k])
            if start == goal:  # don't waste time on the agents that have already reached their goals
                actions.append(0)  # just add useless action to save the order and length of the actions
                continue
            self.agents[k].update_obstacles(obs[k][0], obs[k][1])
            self.agents[k].update_path(start, goal)
            next_node = self.agents[k].get_next_node()
            actions.append(self.actions[(next_node[0] - start[0], next_node[1] - start[1])])
        for idx, action in enumerate(actions):
            if self._rnd.random() < self._random_rate:
                actions[idx] = self._rnd.randint(1, 4)
        return actions


def run_policy(gc: GridConfig, save_animation=False):
    policy = DeterministicPolicy()
    env = pogema_v0(grid_config=gc)
    if save_animation:
        env = AnimationMonitor(env)

    while True:
        obs = env.reset()
        done = [False, ...]
        info = None
        while not all(done):
            obs, reward, done, info = env.step(policy.act(obs, done))
            if all(done):
                break

        yield info[0]['metrics']


def test_life_long():
    gc = GridConfig(num_agents=20, size=8, obs_radius=4, seed=42, max_episode_steps=64, on_target='restart')
    results_generator = run_policy(gc, save_animation=False)

    for _ in range(3):
        metrics = results_generator.__next__()
        assert np.isclose(metrics['avg_throughput'], 1.625)
    metrics = results_generator.__next__()
    assert np.isclose(metrics['avg_throughput'], 1.546875)

    gc = GridConfig(num_agents=24, size=8, obs_radius=4, seed=43, max_episode_steps=64, on_target='restart')
    results_generator = run_policy(gc, save_animation=False)

    metrics = results_generator.__next__()
    assert np.isclose(metrics['avg_throughput'], 0.375)


def test_disappearing():
    gc = GridConfig(num_agents=20, size=8, obs_radius=2, seed=42, density=0.2, max_episode_steps=32, on_target='finish')
    results_generator = run_policy(gc, save_animation=False)

    metrics = results_generator.__next__()
    assert np.isclose(metrics['ep_length'], 22.95)
    assert np.isclose(metrics['ISR'], 0.5)
    assert np.isclose(metrics['CSR'], 0.0)

    metrics = results_generator.__next__()
    assert np.isclose(metrics['ep_length'], 15.55)
    assert np.isclose(metrics['ISR'], 0.9)
    assert np.isclose(metrics['CSR'], 0.0)


def test_non_disappearing():
    gc = GridConfig(num_agents=4, size=5, obs_radius=2, seed=3, density=0.2, max_episode_steps=32, on_target='nothing')
    results_generator = run_policy(gc, save_animation=False)

    metrics = results_generator.__next__()
    assert np.isclose(metrics['ep_length'], 20)
    assert np.isclose(metrics['CSR'], 1.0)
    assert np.isclose(metrics['ISR'], 1.0)

    metrics = results_generator.__next__()
    assert np.isclose(metrics['ep_length'], 13)
    assert np.isclose(metrics['CSR'], 1.0)
    assert np.isclose(metrics['ISR'], 1.0)

    gc = GridConfig(num_agents=7, size=5, obs_radius=2, seed=3, density=0.2, max_episode_steps=32, on_target='nothing')
    results_generator = run_policy(gc, save_animation=False)

    metrics = results_generator.__next__()
    assert np.isclose(metrics['ep_length'], 31)
    assert np.isclose(metrics['CSR'], 0.0)
    assert np.isclose(metrics['ISR'], 0.71428571428)
