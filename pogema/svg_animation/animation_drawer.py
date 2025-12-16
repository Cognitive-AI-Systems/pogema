import math
import typing
from dataclasses import dataclass

from pogema import GridConfig
from pogema.svg_animation.svg_objects import Line, RectangleHref, Animation, Circle, Rectangle


@dataclass
class AnimationConfig:
    directory: str = 'renders/'
    static: bool = False
    show_agents: bool = True
    egocentric_idx: typing.Optional[int] = None
    frame_idx: typing.Optional[int] = None
    uid: typing.Optional[str] = None
    save_every_idx_episode: typing.Optional[int] = 1
    show_grid_lines: bool = True


@dataclass
class SvgSettings:
    r: int = 35
    stroke_width: int = 10
    scale_size: int = 100
    time_scale: float = 0.25
    draw_start: int = 100
    rx: int = 15

    obstacle_color: str = '#84A1AE'
    ego_color: str = '#c1433c'
    ego_other_color: str = '#6e81af'
    shaded_opacity: float = 0.2
    egocentric_shaded: bool = True
    stroke_dasharray: int = 25

    colors: tuple = (
        '#c1433c',
        '#2e6f9e',
        '#6e81af',
        '#00b9c8',
        '#72D5C8',
        '#0ea08c',
        '#8F7B66',
    )


@dataclass
class GridHolder:
    obstacles: typing.Any = None
    episode_length: int = None
    height: int = None
    width: int = None
    colors: dict = None
    history: list = None
    obs_radius: int = None
    grid_config: GridConfig = None
    on_target: str = None
    config: AnimationConfig = None
    svg_settings: SvgSettings = None


class Drawing:

    def __init__(self, height, width, svg_settings):
        self.height = height
        self.width = width
        self.origin = (0, 0)
        self.elements = []
        self.svg_settings = svg_settings

    def add_element(self, element):
        self.elements.append(element)

    def render(self):
        scale = max(self.height, self.width) / 512
        scaled_width = math.ceil(self.width / scale)
        scaled_height = math.ceil(self.height / scale)

        dx, dy = self.origin
        view_box = (dx, dy - self.height, self.width, self.height)

        svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
             width="{scaled_width}" height="{scaled_height}" viewBox="{" ".join(map(str, view_box))}">'''

        definitions = f'''
        <rect id="obstacle" width="{self.svg_settings.r * 2}" height="{self.svg_settings.r * 2}" fill="{self.svg_settings.obstacle_color}" rx="{self.svg_settings.rx}"/>
        <style>
        .line {{stroke: {self.svg_settings.obstacle_color}; stroke-width: {self.svg_settings.stroke_width};}}
        .agent {{r: {self.svg_settings.r};}}
        .target {{fill: none; stroke-width: {self.svg_settings.stroke_width}; r: {self.svg_settings.r};}}
        </style>
        '''

        elements_svg = [svg_header, '<defs>', definitions, '</defs>\n']
        elements_svg.extend(element.render() for element in self.elements)
        elements_svg.append('</svg>')
        return "\n".join(elements_svg)


class AnimationDrawer:

    def __init__(self):
        pass

    def create_animation(self, grid_holder: GridHolder):
        gh = grid_holder
        render_width = gh.height * gh.svg_settings.scale_size + gh.svg_settings.scale_size
        render_height = gh.width * gh.svg_settings.scale_size + gh.svg_settings.scale_size
        drawing = Drawing(width=render_width, height=render_height, svg_settings=SvgSettings())
        obstacles = self.create_obstacles(gh)

        agents = []
        targets = []

        if gh.config.static and gh.config.frame_idx is not None:
            self.create_frame_view(gh)

        if gh.config.show_agents:
            agents = self.create_agents(gh)
            targets = self.create_targets(gh)

            if not gh.config.static:
                self.animate_agents(agents, gh)
                self.animate_targets(targets, gh)
        if gh.config.show_grid_lines:
            grid_lines = self.create_grid_lines(gh, render_width, render_height)
            for line in grid_lines:
                drawing.add_element(line)
        for obj in [*obstacles, *agents, *targets]:
            drawing.add_element(obj)
        if gh.config.egocentric_idx is not None:
            field_of_view = self.create_field_of_view(grid_holder=gh)
            if not gh.config.static:
                self.animate_obstacles(obstacles=obstacles, grid_holder=gh)
                self.animate_field_of_view(field_of_view, gh)
            drawing.add_element(field_of_view)

        return drawing

    @staticmethod
    def fix_point(x, y, length):
        return length - y - 1, x

    @staticmethod
    def check_in_radius(x1, y1, x2, y2, r) -> bool:
        return x2 - r <= x1 <= x2 + r and y2 - r <= y1 <= y2 + r
    
    @staticmethod
    def create_grid_lines(grid_holder: GridHolder, render_width, render_height):
        gh = grid_holder
        offset = 0
        stroke_settings = {'class': 'line'}
        grid_lines = []
        for i in range(-1, grid_holder.height + 1):
            x = i * gh.svg_settings.scale_size + gh.svg_settings.scale_size / 2
            grid_lines.append(Line(x1=x, y1=offset, x2=x, y2=render_height - offset, **stroke_settings))

        for i in range(-1, grid_holder.width + 1):
            y = i * gh.svg_settings.scale_size + gh.svg_settings.scale_size / 2
            grid_lines.append(Line(x1=offset, y1=y, x2=render_width - offset, y2=y, **stroke_settings))

        return grid_lines

    @staticmethod
    def create_field_of_view(grid_holder):
        gh: GridHolder = grid_holder
        ego_idx = gh.config.egocentric_idx
        x, y = gh.history[ego_idx][0].get_xy()
        cx = gh.svg_settings.draw_start + y * gh.svg_settings.scale_size
        cy = gh.svg_settings.draw_start + (gh.width - x - 1) * gh.svg_settings.scale_size

        dr = (grid_holder.obs_radius + 1) * gh.svg_settings.scale_size - gh.svg_settings.stroke_width * 2
        result = Rectangle(
            x=cx - dr + gh.svg_settings.r, y=cy - dr + gh.svg_settings.r,
            width=2 * dr - 2 * gh.svg_settings.r, height=2 * dr - 2 * gh.svg_settings.r,
            stroke=gh.svg_settings.ego_color, stroke_width=gh.svg_settings.stroke_width,
            fill='none', rx=gh.svg_settings.rx, stroke_dasharray=gh.svg_settings.stroke_dasharray
        )

        return result
        
    def create_frame_view(self, grid_holder):
        gh: GridHolder = grid_holder
        frame_idx = gh.config.frame_idx
        gh.history = [[agent_states[gh.config.frame_idx]] for agent_states in gh.history]

    def animate_field_of_view(self, view, grid_holder):
        gh: GridHolder = grid_holder
        x_path = []
        y_path = []
        ego_idx = grid_holder.config.egocentric_idx
        for state in gh.history[ego_idx]:
            x, y = state.get_xy()
            dr = (grid_holder.obs_radius + 1) * gh.svg_settings.scale_size - gh.svg_settings.stroke_width * 2
            cx = gh.svg_settings.draw_start + y * gh.svg_settings.scale_size
            cy = -gh.svg_settings.draw_start + -(gh.width - x - 1) * gh.svg_settings.scale_size
            x_path.append(str(cx - dr + gh.svg_settings.r))
            y_path.append(str(cy - dr + gh.svg_settings.r))

        visibility = ['visible' if state.is_active() else 'hidden' for state in gh.history[ego_idx]]

        view.add_animation(self.compressed_anim('x', x_path, gh.svg_settings.time_scale))
        view.add_animation(self.compressed_anim('y', y_path, gh.svg_settings.time_scale))
        view.add_animation(self.compressed_anim('visibility', visibility, gh.svg_settings.time_scale))

    def animate_agents(self, agents, grid_holder):
        gh: GridHolder = grid_holder
        ego_idx = gh.config.egocentric_idx

        for agent_idx, agent in enumerate(agents):
            x_path = []
            y_path = []
            opacity = []
            for idx, agent_state in enumerate(gh.history[agent_idx]):
                x, y = agent_state.get_xy()

                x_path.append(str(gh.svg_settings.draw_start + y * gh.svg_settings.scale_size))
                y_path.append(str(-gh.svg_settings.draw_start + -(gh.width - x - 1) * gh.svg_settings.scale_size))

                if ego_idx is not None:
                    ego_x, ego_y = gh.history[ego_idx][idx].get_xy()
                    if self.check_in_radius(x, y, ego_x, ego_y, grid_holder.obs_radius):
                        opacity.append('1.0')
                    else:
                        opacity.append(str(gh.svg_settings.shaded_opacity))

            visibility = ['visible' if state.is_active() else 'hidden' for state in gh.history[agent_idx]]

            agent.add_animation(self.compressed_anim('cy', y_path, gh.svg_settings.time_scale))
            agent.add_animation(self.compressed_anim('cx', x_path, gh.svg_settings.time_scale))
            agent.add_animation(self.compressed_anim('visibility', visibility, gh.svg_settings.time_scale))
            if opacity:
                agent.add_animation(self.compressed_anim('opacity', opacity, gh.svg_settings.time_scale))

    @classmethod
    def compressed_anim(cls, attr_name, tokens, time_scale, rep_cnt='indefinite'):
        tokens, times = cls.compress_tokens(tokens)
        cumulative = [0, ]
        for t in times:
            cumulative.append(cumulative[-1] + t)
        times = [str(round(value / cumulative[-1], 10)) for value in cumulative]
        tokens = [tokens[0]] + tokens

        times = times
        tokens = tokens
        return Animation(
            attributeName=attr_name, dur=f'{time_scale * (-1 + cumulative[-1])}s',
            values=";".join(tokens), repeatCount=rep_cnt, keyTimes=";".join(times)
        )

    @staticmethod
    def wisely_add(token, cnt, tokens, times):
        if cnt > 1:
            tokens += [token, token]
            times += [1, cnt - 1]
        else:
            tokens.append(token)
            times.append(cnt)

    @classmethod
    def compress_tokens(cls, input_tokens: list):
        tokens = []
        times = []
        if input_tokens:
            cur_idx = 0
            cnt = 1
            for idx in range(1, len(input_tokens)):
                if input_tokens[idx] == input_tokens[cur_idx]:
                    cnt += 1
                else:
                    cls.wisely_add(input_tokens[cur_idx], cnt, tokens, times)
                    cnt = 1
                    cur_idx = idx
            cls.wisely_add(input_tokens[cur_idx], cnt, tokens, times)
        return tokens, times

    def animate_targets(self, targets, grid_holder):
        gh: GridHolder = grid_holder
        ego_idx = gh.config.egocentric_idx

        for agent_idx, target in enumerate(targets):
            target_idx = ego_idx if ego_idx is not None else agent_idx

            x_path = []
            y_path = []

            for step_idx, state in enumerate(gh.history[target_idx]):
                x, y = state.get_target_xy()
                x_path.append(str(gh.svg_settings.draw_start + y * gh.svg_settings.scale_size))
                y_path.append(str(-gh.svg_settings.draw_start + -(gh.width - x - 1) * gh.svg_settings.scale_size))

            visibility = ['visible' if state.is_active() else 'hidden' for state in gh.history[agent_idx]]

            if gh.on_target == 'restart' or gh.on_target == 'wait':
                target.add_animation(self.compressed_anim('cy', y_path, gh.svg_settings.time_scale))
                target.add_animation(self.compressed_anim('cx', x_path, gh.svg_settings.time_scale))
            target.add_animation(self.compressed_anim("visibility", visibility, gh.svg_settings.time_scale))

    def create_obstacles(self, grid_holder):
        gh = grid_holder
        result = []

        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)

                if gh.obstacles[x][y]:
                    obs_settings = {}
                    obs_settings.update(
                        x=gh.svg_settings.draw_start + i * gh.svg_settings.scale_size - gh.svg_settings.r,
                        y=gh.svg_settings.draw_start + j * gh.svg_settings.scale_size - gh.svg_settings.r,
                        height=gh.svg_settings.r * 2,
                    )

                    if gh.config.egocentric_idx is not None and gh.svg_settings.egocentric_shaded:
                        initial_positions = [agent_states[0].get_xy() for agent_states in gh.history]
                        ego_x, ego_y = initial_positions[gh.config.egocentric_idx]
                        if not self.check_in_radius(x, y, ego_x, ego_y, grid_holder.obs_radius):
                            obs_settings.update(opacity=gh.svg_settings.shaded_opacity)

                    result.append(RectangleHref(**obs_settings))

        return result

    def animate_obstacles(self, obstacles, grid_holder):
        gh: GridHolder = grid_holder
        obstacle_idx = 0

        for i in range(gh.height):
            for j in range(gh.width):
                x, y = self.fix_point(i, j, gh.width)
                if not gh.obstacles[x][y]:
                    continue
                opacity = []
                seen = set()
                for step_idx, agent_state in enumerate(gh.history[gh.config.egocentric_idx]):
                    ego_x, ego_y = agent_state.get_xy()
                    if self.check_in_radius(x, y, ego_x, ego_y, grid_holder.obs_radius):
                        seen.add((x, y))
                    if (x, y) in seen:
                        opacity.append(str(1.0))
                    else:
                        opacity.append(str(gh.svg_settings.shaded_opacity))

                obstacle = obstacles[obstacle_idx]
                obstacle.add_animation(self.compressed_anim('opacity', opacity, gh.svg_settings.time_scale))

                obstacle_idx += 1

    def create_agents(self, grid_holder):
        initial_positions = [state[0].get_xy() for state in grid_holder.history if state[0].is_active()]
        agents = []
        gh: GridHolder = grid_holder
        ego_idx = grid_holder.config.egocentric_idx

        for idx, (x, y) in enumerate(initial_positions):
            circle_settings = {
                'cx': gh.svg_settings.draw_start + y * gh.svg_settings.scale_size,
                'cy': gh.svg_settings.draw_start + (grid_holder.width - x - 1) * gh.svg_settings.scale_size,
                'r': gh.svg_settings.r, 'fill': grid_holder.colors[idx], 'class': 'agent',
            }

            if ego_idx is not None:
                ego_x, ego_y = initial_positions[ego_idx]
                is_out_of_radius = not self.check_in_radius(x, y, ego_x, ego_y, grid_holder.obs_radius)
                circle_settings['fill'] = gh.svg_settings.ego_other_color
                if idx == ego_idx:
                    circle_settings['fill'] = gh.svg_settings.ego_color
                elif is_out_of_radius and gh.svg_settings.egocentric_shaded:
                    circle_settings['opacity'] = gh.svg_settings.shaded_opacity

            agents.append(Circle(**circle_settings))

        return agents

    @staticmethod
    def create_targets(grid_holder):
        gh: GridHolder = grid_holder
        targets = []
        for agent_idx, agent_states in enumerate(gh.history):

            tx, ty = agent_states[0].get_target_xy()
            x, y = ty, gh.width - tx - 1

            if not any([agent_state.is_active() for agent_state in gh.history[agent_idx]]):
                continue

            circle_settings = {"class": 'target'}
            circle_settings.update(
                cx=gh.svg_settings.draw_start + x * gh.svg_settings.scale_size, r=gh.svg_settings.r,
                cy=gh.svg_settings.draw_start + y * gh.svg_settings.scale_size, stroke=gh.colors[agent_idx],
            )

            if gh.config.egocentric_idx is not None:
                if gh.config.egocentric_idx != agent_idx:
                    continue

                circle_settings.update(stroke=gh.svg_settings.ego_color)
            target = Circle(**circle_settings)
            targets.append(target)
        return targets
