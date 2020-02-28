from bot_transfer.envs.hierarchical import Hierarchical, HierarchicalMJ
import numpy as np
from gym import spaces
import bot_transfer
import os
import tempfile
import xml.etree.ElementTree as ET

class Ant(HierarchicalMJ):

    FILE = '/envs/assets/ant_gather.xml'
    def __init__(self, k=20, early_low_termination=False, delta_max=4.0, 
                             gear=None,
                             reset_prob=0.25,
                             include_contacts=True,
                             skill_penalty=0.25):
        
        self.reached_goal = False
        self.arena_width = 8
        self.high_level_sparse_reward = 100
        self.include_contacts = include_contacts

        file_path = os.path.dirname(bot_transfer.__file__) + self.FILE
        if gear:
            xml_path = file_path
            tree = ET.parse(xml_path)
            for motor in tree.find(".//actuator").findall('motor'):
                motor.set("gear", str(gear))

            _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
            tree.write(file_path)

        super(Ant, self).__init__(file_path, 5,
                                  epsilon_low=0.27,
                                  epsilon_high=0.5,
                                  action_penalty=0.5,
                                  skill_penalty=skill_penalty,
                                  early_low_termination=early_low_termination,
                                  k=k,
                                  delta_max=delta_max,
                                  reset_prob=reset_prob,
                                  max_sequential_low=10)       

    def state(self):
        if self.include_contacts:
            return np.concatenate([
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat[:],
                    np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                    self.get_body_com("torso")[:2],
                    self.get_body_com("target")[:2]
                ])
        else:
            return np.concatenate([
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat[:],
                    self.get_body_com("torso")[:2],
                    self.get_body_com("target")[:2]
                ])

    def agent_state_func(self, state):
        # override the goal position and remove location.
        return np.concatenate([
            state[:-4],
            self.low_level_goal - state[-4:-2]
        ])

    def obs_func(self, state):
        # Return just the current position and the goal
        return state[-4:]

    def skill_func(self, observation):
        # return just the current position
        return observation[:2]

    def apply_action(self, action):
        self.model.body_pos[15][:2] = self.low_level_goal
        self.do_simulation(action, self.frame_skip)

    def post_action(self):
        # resets goal if it was reached.
        dist_to_target = np.linalg.norm(self.get_body_com("torso") - self.get_body_com("target"))
        if dist_to_target < self.epsilon_high:
            self.reached_goal = True
            while True:
                goal = self.np_random.uniform(low=-self.arena_width, high=self.arena_width, size=2)
                if np.linalg.norm(goal) < self.arena_width:
                    break
            self.model.body_pos[14][:2] = goal    
    
    def low_level_is_done(self):
        state_vec = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
        notdone = np.isfinite(state_vec).all() \
            and state_vec[2] >= 0.2 and state_vec[2] <= 1.0
        done = not notdone
        return done

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

    def high_level_is_done(self):
        return False

    def high_level_reward(self):
        if self.reached_goal:
            self.reached_goal = False
            return self.high_level_sparse_reward
        else:
            dist_to_target = np.linalg.norm(self.get_body_com("torso") - self.get_body_com("target"))
            return -0.1 * dist_to_target
    
    def reset_func(self, low=False):
        self.sim.reset()
        qpos = self.np_random.uniform(size=self.model.nq, low=-.13, high=.13) + self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .13
        self.set_mj_state(qpos, qvel)
        while True:
            goal = self.np_random.uniform(low=-self.arena_width, high=self.arena_width, size=2)
            if np.linalg.norm(goal) < self.arena_width:
                break
        self.model.body_pos[14][:2] = goal

    def viewer_setup(self):
        self.viewer.cam.distance = 20

class AntDiscrete(Ant):

    def skill_map(self):
        return {
            0: np.array([self.delta_max, 0]),
            1: np.array([0, self.delta_max]),
            2: np.array([-self.delta_max, 0]),
            3: np.array([0, -self.delta_max])
        }

    # def agent_state_func(self, state):
    #     one_hot_skill = np.zeros(len(self.skill_map()))
    #     one_hot_skill[self.current_skill] = 1
    #     return np.concatenate([
    #         state[:-2],
    #         one_hot_skill
    #     ])

class AntDiscreteX(Ant):

    def skill_map(self):
        return {
            0: np.array([self.delta_max, 0]),
            1: np.array([-self.delta_max, 0]),
        }

    # def agent_state_func(self, state):
    #     one_hot_skill = np.zeros(len(self.skill_map()))
    #     one_hot_skill[self.current_skill] = 1
    #     return np.concatenate([
    #         state[:-2],
    #         one_hot_skill
    #     ])

class AntDisabled(Ant):
    FILE = '/envs/assets/ant_disabled_gather.xml'

class Ant2Leg(Ant):
    FILE = '/envs/assets/ant_2leg_gather.xml'

class AntTerrain(Ant):
    FILE = '/envs/assets/ant_terrain.xml'
    
    _HEIGHTFIELD_ID = 0
    _TERRAIN_SMOOTHNESS = .15  # 0.0: maximally bumpy; 1.0: completely smooth.
    _TERRAIN_BUMP_SCALE = 4

    def __init__(self, k=20, early_low_termination=False, delta_max=5.0, gear=None):
        from scipy import ndimage

        self.reached_goal = False
        self.arena_width = 8
        self.high_level_sparse_reward = 100
        super(AntTerrain, self).__init__(k=k, early_low_termination=early_low_termination, delta_max=delta_max, gear=gear)
        
        # Get heightfield resolution, assert that it is square.
        res = self.model.hfield_nrow[self._HEIGHTFIELD_ID]
        assert res == self.model.hfield_ncol[self._HEIGHTFIELD_ID]
        
        terrain = 8 * np.random.randint(2, size=(8,8)) - 7.9375
        terrain = np.repeat(np.repeat(terrain, 5, axis=0), 5, axis=1)
        start_idx = self.model.hfield_adr[self._HEIGHTFIELD_ID]
        self.model.hfield_data[start_idx:start_idx+res**2] = terrain.ravel()

        # Sinusoidal bowl shape.
        # row_grid, col_grid = np.ogrid[-1:1:res*1j, -1:1:res*1j]
        # radius = np.clip(np.sqrt(col_grid**2 + row_grid**2), .04, 1)
        # bowl_shape = .5 - np.cos(2*np.pi*radius)/2
        # # Random smooth bumps.
        # terrain_size = 2 * self.model.hfield_size[self._HEIGHTFIELD_ID, 0]
        # bump_res = int(terrain_size / self._TERRAIN_BUMP_SCALE)
        # bumps = np.random.uniform(self._TERRAIN_SMOOTHNESS, 1, (bump_res, bump_res))
        # print(bumps)
        # smooth_bumps = ndimage.zoom(bumps, res / float(bump_res), order=0)
        # # Terrain is elementwise product.
        # terrain = bowl_shape * smooth_bumps
        # start_idx = self.model.hfield_adr[self._HEIGHTFIELD_ID]
        # self.model.hfield_data[start_idx:start_idx+res**2] = terrain.ravel()

    def reset_func(self, low=False):
        
        res = self.model.hfield_nrow[self._HEIGHTFIELD_ID]
        terrain = 8 * np.random.randint(2, size=(8,8)) - 7.9375
        terrain = np.repeat(np.repeat(terrain, 5, axis=0), 5, axis=1)
        start_idx = self.model.hfield_adr[self._HEIGHTFIELD_ID]
        self.model.hfield_data[start_idx:start_idx+res**2] = terrain.ravel()
        self.sim.reset()

        qpos = self.np_random.uniform(size=self.model.nq, low=-.13, high=.13) + self.init_qpos
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .13
        self.set_mj_state(qpos, qvel)
        while True:
            goal = self.np_random.uniform(low=-self.arena_width, high=self.arena_width, size=2)
            if np.linalg.norm(goal) < self.arena_width:
                break
        
        self.model.body_pos[14][:2] = goal

class AntPush(Ant):
    FILE = '/envs/assets/ant_push.xml'

    def state(self):
        return np.concatenate([
                np.array([self.sim.data.qpos.flat[2] - 4.0]),
                self.sim.data.qpos.flat[3:-2],
                self.sim.data.qvel.flat[:-2],
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat[:96],
                self.sim.data.qpos.flat[-2:],
                self.sim.data.qvel.flat[-2:],
                self.get_body_com("torso")[:2],
            ])

    def agent_state_func(self, state):
        # override the goal position and remove location.
        return np.concatenate([
            state[:-6],
            self.low_level_goal - state[-2:]
        ])

    def obs_func(self, state):
        # Return just the current position and the goal
        return state[-6:]

    def skill_func(self, observation):
        # return just the current position
        return observation[-2:]

    def low_level_is_done(self):
        state_vec = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
        notdone = np.isfinite(state_vec).all() \
            and state_vec[2] >= 4.2 and state_vec[2] <= 5.0
        done = not notdone
        return done
    
    def high_level_is_done(self):
        x_pos, y_pos, z_pos = self.get_body_com("torso")
        dist_to_end = np.linalg.norm(np.array([x_pos, y_pos-27.0]))
        reached_end = True if dist_to_end < 5.0 else False
        fell = True if z_pos < 3.75 else False
        return reached_end or fell

    def high_level_reward(self):
        # self.render()
        x_pos, y_pos, z_pos = self.get_body_com("torso")
        dist_to_end = np.linalg.norm(np.array([x_pos, y_pos - 27.0]))
        reached_end = True if dist_to_end < 5.0 else False
        fell = True if z_pos < 3.75 else False
        if reached_end:
            return 1000
        elif fell:
            return -10
        else:
            return -0.05 * dist_to_end

    def reset_func(self, low=False):
        self.sim.reset()
        qpos = self.np_random.uniform(size=self.model.nq, low=-.13, high=.13) + self.init_qpos
        qpos[-2:] = self.init_qpos[-2:]
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .13
        qvel[-2:] = self.init_qvel[-2:]
        self.set_mj_state(qpos, qvel)

class AntPushLow(Ant):
    FILE = '/envs/assets/ant_push_low.xml'
    def state(self):
        return np.concatenate([
                np.array([self.sim.data.qpos.flat[2]- 4.0]),
                self.sim.data.qpos.flat[3:],
                self.sim.data.qvel.flat[:],
                # np.clip(self.sim.data.cfrc_ext, -1, 1).flat[:96],
                self.get_body_com("torso")[:2],
                self.get_body_com("target")[:2]
            ])

    def low_level_is_done(self):
        state_vec = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
        notdone = np.isfinite(state_vec).all() \
            and state_vec[2] >= 4.2 and state_vec[2] <= 5.0
        done = not notdone
        return done

if __name__ == "__main__":
    from bot_transfer.envs.hierarchical import LowLevelEnv, HighLevelEnv, FullEnv
    from gym.wrappers import TimeLimit

    env = AntPush(gear=120)
    env = FullEnv(env)
    env = TimeLimit(env, 2000)

    env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()

    