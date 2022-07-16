import time
from collections import defaultdict

import numpy as np

from pogema import GridConfig


def generate_obstacles(grid_config: GridConfig, rnd=None):
    if rnd is None:
        rnd = np.random.default_rng(grid_config.seed)
    return rnd.binomial(1, grid_config.density, (grid_config.size, grid_config.size))


def generate_positions_and_targets(obstacles, grid_config: GridConfig):
    c = grid_config
    grid = obstacles.copy()
    q = []
    # pick free id
    current_id = max(c.FREE, c.OBSTACLE) + 1

    for x in range(c.size):
        for y in range(c.size):
            if grid[x, y] != c.FREE:
                continue

            grid[x, y] = current_id
            q.append((x, y))

            while len(q):
                cx, cy = q.pop(0)

                for dx, dy in c.MOVES:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < c.size and 0 <= ny < c.size:
                        if grid[nx, ny] == c.FREE:
                            grid[nx, ny] = current_id
                            q.append((nx, ny))

            current_id += 1
    xy_to_id = dict()
    id_to_xy = defaultdict(set)
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            if grid[x, y] != c.OBSTACLE:
                xy_to_id[x, y] = grid[x, y]
                id_to_xy[grid[x, y]].add((x, y))
    order = list(xy_to_id.keys())
    np.random.default_rng(c.seed).shuffle(order)

    requests = defaultdict(set)
    done_requests = 0
    positions_xy = []
    finishes_xy = [(-1, -1) for _ in range(c.num_agents)]
    for x, y in order:
        if (x, y) not in xy_to_id:
            continue

        # remove cell
        id_ = xy_to_id.pop((x, y))
        id_to_xy[id_].remove((x, y))

        # deal with requests first
        if requests[id_]:
            finishes_xy[requests[id_].pop()] = x, y
            done_requests += 1
            continue

        # no empty space so skip
        if not len(id_to_xy[id_]):
            continue

        if len(positions_xy) >= c.num_agents:
            if done_requests >= c.num_agents:
                break
            continue

        # add start position and request finish for it
        requests[id_].add(len(positions_xy))
        positions_xy.append((x, y))
    return positions_xy, finishes_xy


def bfs(grid, moves, size, start_id, free_cell):
    q = []
    current_id = start_id

    components = [0 for _ in range(start_id)]

    size_x = len(grid)
    size_y = len(grid[0])

    for x in range(size_x):
        for y in range(size_y):
            if grid[x, y] != free_cell:
                continue

            grid[x, y] = current_id
            components.append(1)
            q.append((x, y))

            while len(q):
                cx, cy = q.pop(0)

                for dx, dy in moves:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < size_x and 0 <= ny < size_y:
                        if grid[nx, ny] == free_cell:
                            grid[nx, ny] = current_id
                            components[current_id] += 1
                            q.append((nx, ny))

            current_id += 1
    return components


def placing_fast(order, components, grid, start_id, num_agents):
    link_to_next = [-1 for _ in range(len(order))]
    colors = [-1 for _ in range(len(components))]
    size = len(order)
    for index in range(size):
        reversed_index = len(order) - index - 1
        color = grid[order[reversed_index]]
        link_to_next[reversed_index] = colors[color]
        colors[color] = reversed_index

    positions_xy = []
    finishes_xy = []

    for index in range(len(order)):
        next_index = link_to_next[index]
        if next_index == -1:
            continue

        positions_xy.append(order[index])
        finishes_xy.append(order[next_index])

        link_to_next[next_index] = -1
        if len(finishes_xy) >= num_agents:
            break
    return positions_xy, finishes_xy


def placing(order, components, grid, start_id, num_agents):
    requests = [[] for _ in range(len(components))]

    done_requests = 0
    positions_xy = []
    finishes_xy = [(-1, -1) for _ in range(num_agents)]
    for x, y in order:
        if grid[x, y] < start_id:
            continue

        id_ = grid[x, y]
        grid[x, y] = 0

        if requests[id_]:
            tt = requests[id_].pop()
            finishes_xy[tt] = x, y
            done_requests += 1
            continue

        if len(positions_xy) >= num_agents:
            if done_requests >= num_agents:
                break
            continue

        if components[id_] >= 2:
            components[id_] -= 2
            requests[id_].append(len(positions_xy))
            positions_xy.append((x, y))

    return positions_xy, finishes_xy


def generate_positions_and_targets_fast(obstacles, grid_config):
    c = grid_config
    grid = obstacles.copy()

    start_id = max(c.FREE, c.OBSTACLE) + 1

    components = bfs(grid, tuple(c.MOVES), c.size, start_id, free_cell=c.FREE)
    height, width = obstacles.shape
    order = [(x, y) for x in range(height) for y in range(width) if grid[x, y] >= start_id]
    np.random.default_rng(c.seed).shuffle(order)

    return placing(order=order, components=components, grid=grid, start_id=start_id, num_agents=c.num_agents)


def generate_new_target(rnd_generator, point_to_component, component_to_points, position):

    component_id = point_to_component[position]
    component = component_to_points[component_id]
    new_target = tuple(*rnd_generator.choice(component, 1))

    return new_target


def get_components(grid_config, obstacles, positions_xy, target_xy):
    c = grid_config
    grid = obstacles.copy()

    start_id = max(c.FREE, c.OBSTACLE) + 1
    components = bfs(grid, tuple(c.MOVES), c.size, start_id, free_cell=c.FREE)
    height, width = obstacles.shape

    comp_to_points = defaultdict(list)
    point_to_comp = {}
    for x in range(height):
        for y in range(width):
            comp_to_points[grid[x, y]].append((x, y))
            point_to_comp[(x, y)] = grid[x, y]
    return comp_to_points, point_to_comp


def time_it(func, num_iterations):
    start = time.monotonic()
    for index in range(num_iterations):
        grid_config = GridConfig(num_agents=64, size=64, seed=index)
        obstacles = generate_obstacles(grid_config)
        result = func(obstacles, grid_config, )
        if index == 0 and num_iterations > 1:
            print(result)
    finish = time.monotonic()

    return finish - start


def main():
    num_iterations = 1000
    time_it(generate_positions_and_targets, num_iterations=1)
    time_it(generate_positions_and_targets_fast, num_iterations=1)
    print('fast:', time_it(generate_positions_and_targets_fast, num_iterations=num_iterations))
    print('slow:', time_it(generate_positions_and_targets, num_iterations=num_iterations))


if __name__ == '__main__':
    main()
