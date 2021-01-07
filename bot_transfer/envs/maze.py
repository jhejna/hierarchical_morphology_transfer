import tempfile
import xml.etree.ElementTree as ET
import bot_transfer
import os
from bot_transfer.envs.hierarchical import HierarchicalMJ
import numpy as np
from gym import spaces
import os.path as osp
import math

def line_intersect(pt1, pt2, ptA, ptB):
    """
    Taken from https://www.cs.hmc.edu/ACM/lectures/intersections.html
    this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
    returns a tuple: (xi, yi, valid, r, s), where
    (xi, yi) is the intersection
    r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
    s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
    valid == 0 if there are 0 or inf. intersections (invalid)
    valid == 1 if it has a unique intersection ON the segment
    """
    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0, 0, 0, 0, 0)

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (xi, yi, 1, r, s)


def ray_segment_intersect(ray, segment):
    """
    Check if the ray originated from (x, y) with direction theta intersects the line segment (x1, y1) -- (x2, y2),
    and return the intersection point if there is one
    """
    (x, y), theta = ray
    # (x1, y1), (x2, y2) = segment
    pt1 = (x, y)
    len = 1
    pt2 = (x + len * math.cos(theta), y + len * math.sin(theta))
    xo, yo, valid, r, s = line_intersect(pt1, pt2, *segment)
    if valid and r >= 0 and 0 <= s <= 1:
        return (xo, yo)
    return None


def point_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def construct_maze(maze_id=0, length=1):
    # define the maze to use
    if maze_id == 0:
        if length != 1:
            raise NotImplementedError("Maze_id 0 only has length 1!")
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 'g', 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == 1:  # donuts maze: can reach the single goal by 2 equal paths
        c = length + 4
        M = np.ones((c, c))
        M[1:c - 1, (1, c - 2)] = 0
        M[(1, c - 2), 1:c - 1] = 0
        M = M.astype(int).tolist()
        M[1][c // 2] = 'r'
        M[c - 2][c // 2] = 'g'
        structure = M

    elif maze_id == 2:  # spiral maze: need to use all the keys (only makes sense for length >=3)
        c = length + 4
        M = np.ones((c, c))
        M[1:c - 1, (1, c - 2)] = 0
        M[(1, c - 2), 1:c - 1] = 0
        M = M.astype(int).tolist()
        M[1][c // 2] = 'r'
        # now block one of the ways and put the goal on the other side
        M[1][c // 2 - 1] = 1
        M[1][c // 2 - 2] = 'g'
        structure = M

    elif maze_id == 3:  # corridor with goals at the 2 extremes
        structure = [
            [1] * (2 * length + 5),
            [1, 'g'] + [0] * length + ['r'] + [0] * length + ['g', 1],
            [1] * (2 * length + 5),
            ]

    elif 4 <= maze_id <= 7:  # cross corridor, goal in
        c = 2 * length + 5
        M = np.ones((c, c))
        M = M - np.diag(np.ones(c))
        M = M - np.diag(np.ones(c - 1), 1) - np.diag(np.ones(c - 1), -1)
        i = np.arange(c)
        j = i[::-1]
        M[i, j] = 0
        M[i[:-1], j[1:]] = 0
        M[i[1:], j[:-1]] = 0
        M[np.array([0, c - 1]), :] = 1
        M[:, np.array([0, c - 1])] = 1
        M = M.astype(int).tolist()
        M[c // 2][c // 2] = 'r'
        if maze_id == 4:
            M[1][1] = 'g'
        if maze_id == 5:
            M[1][c - 2] = 'g'
        if maze_id == 6:
            M[c - 2][1] = 'g'
        if maze_id == 7:
            M[c - 2][c - 2] = 'g'
        structure = M

    elif maze_id == 8:  # reflexion of benchmark maze
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'g', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]

    elif maze_id == 9:  # sym benchmark maze
        structure = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 'r', 1],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 'g', 1],
            [1, 1, 1, 1, 1],
        ]

    elif maze_id == 10:  # reflexion of sym of benchmark maze
        structure = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 'g', 1],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 'r', 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == 11:
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 0.5, 0.25, 0, 1],
            [1, 'g', 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == 12:
        structure = np.random.randint(2, size=(21,21)).tolist()
        structure[0][0] = 'g'
        structure[10][10] = 'r'
    if structure:
        return structure
    else:
        raise NotImplementedError("The provided MazeId is not recognized")


class MazeMJ(HierarchicalMJ):

    STRUCTURE = [
        [1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 'g', 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    RANDOM_GOALS = False
    SCALING = 8.0
    HEIGHT = 2
    FILE = None
    MAZE_ID = 0
    SKIP = 3
    SENSORS = 0
    RANGE = 20
    DIST_REWARD = 0.0

    def __init__(self, epsilon_low=0.01,
                       epsilon_high=0.01, 
                       action_penalty=0.05,
                       skill_penalty=0.05,
                       early_low_termination=False,
                       k=10,
                       delta_max=1,
                       sparse_reward=1000,
                       agent_size=None,
                       gear=None,
                       ant_density=False,
                       ant_mass=False,
                       reset_prob=1.0,
                       sample_goals=False,
                       dist_reward=None
                       ):
        
        self.RANDOM_GOALS = sample_goals
        self.sparse_reward = sparse_reward
        if dist_reward:
            self.dist_reward = dist_reward
        else:
            self.dist_reward = 0.0
        print("########################################")
        print("DIST REWARD", self.dist_reward)
        print("########################################")
        self.n_bins = self.SENSORS
        self.sensor_range = self.RANGE

        xml_path = os.path.dirname(bot_transfer.__file__) + self.FILE
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        self.STRUCTURE = construct_maze(maze_id=self.MAZE_ID, length=1)

        torso_x, torso_y = self.get_agent_start()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

        if agent_size:
            pm_geom_elem = tree.find(".//worldbody/body/geom")
            pm_geom_elem.set("size", str(agent_size))
            pm_geom_elem.set("pos", "0 0 " + str(agent_size))
        
        if ant_density:
            pm_geom_elem = tree.find(".//worldbody/body/geom")
            pm_geom_elem.set("density", "1.6781894576")
        if ant_mass:
            pm_geom_elem = tree.find(".//worldbody/body/geom")
            pm_geom_elem.set("mass", "0.87871")

        if gear:
            for motor in tree.find(".//actuator").findall('motor'):
                motor.set("gear", str(gear))

        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 'g':
                    self.current_goal_pos = (i,j)
                if (isinstance(self.STRUCTURE[i][j], int) or isinstance(self.STRUCTURE[i][j], float)) \
                    and self.STRUCTURE[i][j] > 0:
                    height = float(self.STRUCTURE[i][j])
                    # offset all coordinates so that robot starts at the origin
                    if height == 0.25:
                        ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * self.SCALING - torso_x,
                                        i * self.SCALING - torso_y,
                                        self.HEIGHT / 2 * height),
                        size="%f %f %f" % (0.5 * self.SCALING,
                                        0.5 * self.SCALING,
                                        self.HEIGHT / 2 * height),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.45 0.75 0.95 1"
                        )
                    elif height == 0.5:
                        ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * self.SCALING - torso_x,
                                        i * self.SCALING - torso_y,
                                        self.HEIGHT / 2 * height),
                        size="%f %f %f" % (0.5 * self.SCALING,
                                        0.5 * self.SCALING,
                                        self.HEIGHT / 2 * height),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.05 0.3 0.7 1"
                        )
                    else:
                        ET.SubElement(
                            worldbody, "geom",
                            name="block_%d_%d" % (i, j),
                            pos="%f %f %f" % (j * self.SCALING - torso_x,
                                            i * self.SCALING - torso_y,
                                            self.HEIGHT / 2 * height),
                            size="%f %f %f" % (0.5 * self.SCALING,
                                            0.5 * self.SCALING,
                                            self.HEIGHT / 2 * height),
                            type="box",
                            material="",
                            contype="1",
                            conaffinity="1",
                            rgba="%f %f 0.3 1" % (height * 0.3, height * 0.3)
                        )
                    print("#####################", "Height", height)

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)  # here we write a temporal file with the robot specifications. Why not the original one??
        
        self.possible_goal_positions = list()
        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 0 or self.STRUCTURE[i][j] == 'g':
                    self.possible_goal_positions.append((i,j))
        self.goal_range = self.get_goal_range()
        self.center_goal = np.array([(self.goal_range[0] + self.goal_range[1]) / 2, 
                                     (self.goal_range[2] + self.goal_range[3]) / 2])

        # Call the super constructor last, with the path to the temp file that describes the maze
        super(MazeMJ, self).__init__(file_path, self.SKIP,
                       epsilon_low=epsilon_low,
                       epsilon_high=epsilon_high, 
                       action_penalty=action_penalty,
                       skill_penalty=skill_penalty,
                       early_low_termination=early_low_termination,
                       k=k,
                       delta_max=delta_max,
                       reset_prob=reset_prob)
    
    def sample_goal_pos(self):
        if not self.RANDOM_GOALS:
            return
        cur_x, cur_y = self.current_goal_pos
        self.STRUCTURE[cur_x][cur_y] = 0
        new_x, new_y = self.possible_goal_positions[self.np_random.randint(low=0, high=len(self.possible_goal_positions))]
        self.STRUCTURE[new_x][new_y] = 'g'
        self.current_goal_pos = (new_x, new_y)
        self.goal_range = self.get_goal_range()
        self.center_goal = np.array([(self.goal_range[0] + self.goal_range[1]) / 2, 
                                     (self.goal_range[2] + self.goal_range[3]) / 2])

    def high_level_reward(self):
        minx, maxx, miny, maxy = self.goal_range
        x, y = self.get_body_com("torso")[:2]
        reward = 0
        if minx <= x <= maxx and miny <= y <= maxy:
            reward += self.sparse_reward
        if self.dist_reward > 0:
            # adds L2 reward
            reward += -self.dist_reward * np.linalg.norm(self.get_body_com("torso")[:2] - self.center_goal)
        return reward

    def high_level_is_done(self):
        minx, maxx, miny, maxy = self.goal_range
        x, y = self.get_body_com("torso")[:2]
        if minx <= x <= maxx and miny <= y <= maxy:
            return True
        else:
            return False

    def viewer_setup(self):
        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0
        self.viewer.cam.lookat[0] += 8.0
        self.viewer.cam.lookat[1] += 4.0
        self.viewer.cam.distance *= 1.1

    def get_agent_start(self):
        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 'r':
                    return j * self.SCALING, i * self.SCALING
        assert False

    def get_goal_range(self):
        for i in range(len(self.STRUCTURE)):
            for j in range(len(self.STRUCTURE[0])):
                if self.STRUCTURE[i][j] == 'g':
                    minx = j * self.SCALING - self.SCALING * 0.5 - self._init_torso_x
                    maxx = j * self.SCALING + self.SCALING * 0.5 - self._init_torso_x
                    miny = i * self.SCALING - self.SCALING * 0.5 - self._init_torso_y
                    maxy = i * self.SCALING + self.SCALING * 0.5 - self._init_torso_y
                    return minx, maxx, miny, maxy

    def get_current_maze_obs(self):
        # The observation would include both information about the robot itself as well as the sensors around its
        # environment
        robot_x, robot_y = self.get_body_com("torso")[:2]

        structure = self.STRUCTURE
        size_scaling = self.SCALING

        segments = []
        # compute the distance of all segments

        # Get all line segments of the goal and the obstacles
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1 or structure[i][j] == 'g':
                    cx = j * size_scaling - self._init_torso_x
                    cy = i * size_scaling - self._init_torso_y
                    x1 = cx - 0.5 * size_scaling
                    x2 = cx + 0.5 * size_scaling
                    y1 = cy - 0.5 * size_scaling
                    y2 = cy + 0.5 * size_scaling
                    struct_segments = [
                        ((x1, y1), (x2, y1)),
                        ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)),
                        ((x1, y2), (x1, y1)),
                    ]
                    for seg in struct_segments:
                        segments.append(dict(
                            segment=seg,
                            type=structure[i][j],
                        ))

        wall_readings = np.zeros(self.n_bins)
        goal_readings = np.zeros(self.n_bins)

        for ray_idx in range(self.n_bins):
            ray_ori = ray_idx * 2*np.pi/self.n_bins
            ray_segments = []
            for seg in segments:
                p = ray_segment_intersect(ray=((robot_x, robot_y), ray_ori), segment=seg["segment"])
                if p is not None:
                    ray_segments.append(dict(
                        segment=seg["segment"],
                        type=seg["type"],
                        ray_ori=ray_ori,
                        distance=point_distance(p, (robot_x, robot_y)),
                    ))
            if len(ray_segments) > 0:
                first_seg = sorted(ray_segments, key=lambda x: x["distance"])[0]
                # print first_seg
                if first_seg["type"] == 1:
                    # Wall -> add to wall readings
                    if first_seg["distance"] <= self.sensor_range:
                        wall_readings[ray_idx] = (self.sensor_range - first_seg["distance"]) / self.sensor_range
                elif first_seg["type"] == 'g':
                    # Goal -> add to goal readings
                    if first_seg["distance"] <= self.sensor_range:
                        goal_readings[ray_idx] = (self.sensor_range - first_seg["distance"]) / self.sensor_range
                else:
                    assert False

        obs = np.concatenate([
            wall_readings,
            goal_readings
        ])
        return obs

    def state(self):
        return NotImplemented

    def agent_state_func(self, state):
        return NotImplemented

    def obs_func(self, state):
        return NotImplemented

    def skill_func(self, obs):
        return NotImplemented

    def low_level_is_done(self):
        return NotImplemented

    def low_level_reward(self):
        return NotImplemented

    def apply_action(self):
        return NotImplemented

    def reset_func(self, low=False):
        return NotImplemented

class PointMaze(MazeMJ):
    FILE = '/envs/assets/point_gather.xml'
    SKIP = 3

    def __init__(self, k=15, early_low_termination=False, delta_max=4, skill_penalty=0.005, 
                        action_penalty=0.5, agent_size=0.5, ant_density=False, ant_mass=False, gear=75, reset_prob=None, sample_goals=False, dist_reward=None):

        super(PointMaze, self).__init__(
                        epsilon_low=0.27,
                        epsilon_high=0.5,
                        action_penalty=action_penalty,
                        skill_penalty=skill_penalty,
                        early_low_termination=early_low_termination,
                        k=k,
                        delta_max=delta_max,
                        agent_size=agent_size,
                        ant_density=ant_density,
                        ant_mass=ant_mass,
                        gear=gear,
                        sample_goals=sample_goals,
                        dist_reward=dist_reward)

    def state(self):
        if self.n_bins > 0:
            return np.concatenate([
                    self.sim.data.qvel.flat[:],
                    self.get_current_maze_obs(),
                    self.get_body_com("torso")[:2],
                    self.center_goal
                ])
        else:
            return np.concatenate([
                    self.sim.data.qvel.flat[:],
                    self.get_body_com("torso")[:2],
                    self.center_goal
                ])

    def agent_state_func(self, state):
        return np.concatenate([
            state[:-(4+2*self.n_bins)],
            self.low_level_goal - state[-4:-2]
        ])

    def obs_func(self, state):
        return state[-(4+2*self.n_bins):]

    def skill_func(self, obs):
        return obs[-4:-2]

    def low_level_is_done(self):
        return False

    def low_level_reward(self, prev_state):
        # dist_to_low_level_goal = np.linalg.norm(self.skill_func(self.obs_func(self.state())) - self.low_level_goal)
        # reward = -1 * dist_to_low_level_goal
        # if dist_to_low_level_goal < self.epsilon_low:
        #     reward += 50
        # return reward
        prev_delta = prev_state[-2:]
        curr_delta = self.agent_state_func(self.state())[-2:]
        forward_reward = np.dot(prev_delta - curr_delta, prev_delta) / (np.linalg.norm(prev_delta) + 1e-6)
        forward_reward /= self.dt
        return forward_reward

    def reset_func(self, low=False):
        self.sim.reset()
        # Setup the goal for the maze
        self.sample_goal_pos()

        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        self.model.body_pos[2][:2] = self.center_goal        
        qvel = self.init_qvel + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv)
        self.set_mj_state(qpos, qvel)

    def apply_action(self, action):
        self.model.body_pos[3][:2] = self.low_level_goal
        self.do_simulation(action, self.frame_skip)

class PointMazeVelocity(PointMaze):

    def apply_action(self, action):
        self.model.body_pos[3][:2] = self.low_level_goal

        clipped_action = np.clip(action, self.action_space().low, self.action_space().high)
        clipped_action *= self.SCALING
        for _ in range(self.frame_skip):
            qpos = self.sim.data.qpos.flat[:]
            qvel = self.sim.data.qvel.flat[:]
            qvel[:2] = clipped_action
            self.set_mj_state(qpos, qvel)
            self.do_simulation(np.zeros(2), 1)

class AntMaze(MazeMJ):
    FILE = '/envs/assets/ant_gather.xml'
    SKIP = 5

    def __init__(self, k=15, early_low_termination=False, delta_max=4, skill_penalty=0.005,
                             action_penalty=0.5, gear=None, reset_prob=1.0, include_contacts=True, sample_goals=False, dist_reward=None):
        self.include_contacts = include_contacts
        super(AntMaze, self).__init__(
                        epsilon_low=0.27,
                        epsilon_high=0.5,
                        action_penalty=action_penalty,
                        skill_penalty=skill_penalty,
                        early_low_termination=early_low_termination,
                        k=k,
                        delta_max=delta_max,
                        reset_prob=reset_prob,
                        sample_goals=sample_goals,
                        dist_reward=dist_reward)

    def state(self):
        rf_readings = self.sim.data.sensordata
        state_data = [ self.sim.data.qpos.flat[2:], self.sim.data.qvel.flat[:]]
        if self.include_contacts:
            state_data.append(np.clip(self.sim.data.cfrc_ext, -1, 1).flat)
        if not (rf_readings is None):
            state_data.append(np.where(rf_readings == -1.0, 1.0, np.tanh(rf_readings)))
        if self.n_bins > 0:
            state_data.append(self.get_current_maze_obs())
        state_data.append(self.get_body_com("torso")[:2])
        state_data.append(self.center_goal)
        return np.concatenate(state_data)

    def agent_state_func(self, state):
        return np.concatenate([
            state[:-(4+2*self.n_bins)],
            self.low_level_goal - state[-4:-2]
        ])

    def obs_func(self, state):
        # Return just the current position and the maze obs
        return state[-(4+2*self.n_bins):]

    def skill_func(self, observation):
        # return just the current position
        return observation[-4:-2]

    def apply_action(self, action):
        self.model.body_pos[-1][:2] = self.low_level_goal
        self.do_simulation(action, self.frame_skip)

    def low_level_is_done(self):
        return False

    def low_level_reward(self, prev_state):
        prev_delta = prev_state[-2:]
        curr_delta = self.agent_state_func(self.state())[-2:]
        # Previous delta is given by prev delta.
        # The vector of our movement is given by prev_delta - cur_delta
        # We then get the projection of our movement onto the previous delta.
        forward_reward = np.dot(prev_delta - curr_delta, prev_delta) / (np.linalg.norm(prev_delta) + 1e-6)
        forward_reward /= self.dt
        survive_reward = 0.7
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        contact_cost = 0
        return forward_reward - contact_cost + survive_reward

    def reset_func(self, low=False):
        self.sim.reset()
        self.sample_goal_pos()
        qpos = self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) + self.init_qpos
        self.model.body_pos[-2][:2] = self.center_goal
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_mj_state(qpos, qvel)

class QuadrupedMaze(AntMaze):
    FILE = '/envs/assets/quadruped_gather.xml'

class AntDisabledMaze(AntMaze):
    FILE = '/envs/assets/ant_disabled_gather.xml'

class Ant2LegMaze(AntMaze):
    FILE = '/envs/assets/ant_2leg_gather.xml'

class PointSteps(PointMaze):
    MAZE_ID = 11
    SCALING = 4
    HEIGHT = 0.625
    SENSORS = 0
    RANDOM_GOALS = False

class AntSteps(AntMaze):
    MAZE_ID = 11
    SCALING = 4
    HEIGHT = 0.625 # 0.75
    SENSORS = 0
    RANDOM_GOALS = False
    FILE = '/envs/assets/ant_step.xml'

class AntStepTraining(AntMaze):
    MAZE_ID = 12
    SCALING = 3.5
    HEIGHT = 0.15625 # 0.1875
    SENSORS = 0
    RANDOM_GOALS = False
    FILE = '/envs/assets/ant_step.xml'

    def reset_func(self, low=False):
        self.sim.reset()
        self.sample_goal_pos()
        qpos = self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) + self.init_qpos
        qpos[:2] = self.np_random.uniform(size=2, low=-26, high=26)
        self.model.body_pos[14][:2] = self.center_goal
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_mj_state(qpos, qvel)

if __name__ == "__main__":
    from bot_transfer.envs.hierarchical import LowLevelEnv, FullEnv
    from gym.wrappers import TimeLimit
    env = LowLevelEnv(AntMaze(gear=100, sample_goals=False))
    env = TimeLimit(env, 100)
    env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, done, _ = env.step(env.action_space.sample())
        if done:
            print("Reset called")
            env.reset()
