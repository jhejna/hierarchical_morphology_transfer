from bot_transfer.envs.hierarchical import Hierarchical, HierarchicalMJ
import numpy as np
from gym import spaces
import bot_transfer
import os
import xml.etree.ElementTree as ET
import tempfile

class PointMassMJ(HierarchicalMJ):

    FILE = None
    
    def __init__(self, epsilon_low=0.01,
                       epsilon_high=0.01, 
                       action_penalty=0.05,
                       skill_penalty=0.05,
                       early_low_termination=False,
                       k=10,
                       delta_max=1,
                       high_level_sparse_reward=100,
                       arena_width=0.3,
                       velocity_bound=0.01,
                       agent_size=None,
                       ant_density=False,
                       ant_mass=False,
                       gear=None,
                       reset_prob=0.25):

        self.reached_goal = False
        self.arena_width = arena_width
        self.high_level_sparse_reward = high_level_sparse_reward
        self.velocity_bound = velocity_bound

        file_path = os.path.dirname(bot_transfer.__file__) + self.FILE
        if agent_size or ant_density or ant_mass or gear:
            xml_path = file_path
            tree = ET.parse(xml_path)
            pm_geom_elem = tree.find(".//worldbody/body/geom")
            if agent_size:
                pm_geom_elem.set("size", str(agent_size))
                pm_geom_elem.set("pos", "0 0 " + str(agent_size))
            if ant_density:
                # pm_geom_elem.set("density", "1.6781894576")
                pm_geom_elem.set("density", "2")
            if ant_mass:
                pm_geom_elem.set("mass", "0.87871")
            if gear:
                for motor in tree.find(".//actuator").findall('motor'):
                    motor.set("gear", str(gear))
            _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
            tree.write(file_path)

        super(PointMassMJ, self).__init__(file_path, 3,
                       epsilon_low=epsilon_low,
                       epsilon_high=epsilon_high, 
                       action_penalty=action_penalty,
                       skill_penalty=skill_penalty,
                       early_low_termination=early_low_termination,
                       k=k,
                       delta_max=delta_max,
                       reset_prob=reset_prob)
       
    def state(self):
        return np.concatenate([
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:],
                self.get_body_com("torso")[:2],
                self.get_body_com("target")[:2]
            ])

    def agent_state_func(self, state):
        # override the goal position
        # Agent state func is q1, q1, dx, dy
        return np.concatenate([
            state[:-4],
            self.low_level_goal - state[-4:-2]
        ])

    def obs_func(self, state):
        # remove the velocity component thats at the start
        return state[-4:]

    def skill_func(self, observation):
        # return just the current position
        return observation[:2]

    def apply_action(self, action):
        self.model.body_pos[3][:2] = self.low_level_goal
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
            self.model.body_pos[2][:2] = goal
    
    def low_level_is_done(self):
        return False

    def low_level_reward(self, prev_state):
        # OLD L2 Reward + Sparse Bonus
        # dist_to_low_level_goal = np.linalg.norm(self.skill_func(self.obs_func(self.state())) - self.low_level_goal)
        # reward = -1 * dist_to_low_level_goal
        # if dist_to_low_level_goal < self.epsilon_low:

        # NEW weighted cos Reward (Like ant)
        prev_delta = prev_state[-2:]
        curr_delta = self.agent_state_func(self.state())[-2:]
        forward_reward = np.dot(prev_delta - curr_delta, prev_delta) / (np.linalg.norm(prev_delta) + 1e-6)
        forward_reward /= self.dt
        return forward_reward

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
        qpos = self.np_random.uniform(low=-0.05*self.arena_width, high=0.05*self.arena_width, size=self.model.nq) + self.init_qpos
        while True:
            goal = self.np_random.uniform(low=-self.arena_width, high=self.arena_width, size=2)
            if np.linalg.norm(goal) < self.arena_width:
                break
        self.model.body_pos[2][:2] = goal
        qvel = self.init_qvel + self.np_random.uniform(low=-self.velocity_bound, high=self.velocity_bound, size=self.model.nv)
        self.set_mj_state(qpos, qvel)

    def viewer_setup(self):
        self.viewer.cam.distance *= 2.5

class PointMassDiscreteMJ(PointMassMJ):

    def skill_map(self):
        return {
            0: np.array([self.delta_max, 0]),
            1: np.array([0, self.delta_max]),
            2: np.array([-self.delta_max, 0]),
            3: np.array([0, -self.delta_max])
        }

    # def agent_state_func(self, state):
    #     # Option 2: Give agent the One-Hot Skill instead of state delta.
    #     one_hot_skill = np.zeros(len(self.skill_map()))
    #     one_hot_skill[self.current_skill] = 1
    #     return np.concatenate([
    #         state[:4],
    #         one_hot_skill
    #     ])

class PointMassVelocityMJ(PointMassMJ):

    def apply_action(self, action):
        self.model.body_pos[3][:2] = self.low_level_goal

        clipped_action = np.clip(action, 0.35*self.action_space().low, 0.35*self.action_space().high)
        for _ in range(self.frame_skip):
            qpos = self.sim.data.qpos.flat[:]
            qvel = self.sim.data.qvel.flat[:]
            qvel[:2] = clipped_action
            self.set_mj_state(qpos, qvel)
            self.do_simulation(np.zeros(2), 1)

class PointMassSmallMJ(PointMassMJ):
    
    FILE = '/envs/assets/point_mass_small.xml'

    def __init__(self, k=10, early_low_termination=False, delta_max=0.12, reset_prob=0.25):

        super(PointMassSmallMJ, self).__init__(
                        epsilon_low=0.005,
                        epsilon_high=0.006,
                        action_penalty=0.1,
                        skill_penalty=0.05,
                        early_low_termination=early_low_termination,
                        k=k,
                        delta_max=delta_max,
                        high_level_sparse_reward=100,
                        arena_width=0.29,
                        velocity_bound=0.01,
                        reset_prob=reset_prob)

class PointMassDiscreteSmallMJ(PointMassSmallMJ, PointMassDiscreteMJ):
    pass

class PointMassSmallVelocityMJ(PointMassSmallMJ, PointMassVelocityMJ):
    pass

class PointMassSmallRotMJ(PointMassSmallMJ):
    FILE = '/envs/assets/point_mass_small_rot.xml'
    
class PointMassSmallRotVelocityMJ(PointMassSmallMJ, PointMassVelocityMJ):
    FILE = '/envs/assets/point_mass_small_rot.xml'
 
class PointMassLargeMJ(PointMassMJ):
    
    FILE = '/envs/assets/point_gather.xml'
    
    def __init__(self, k=15, early_low_termination=False, delta_max=4, agent_size=0.4, ant_density=False, ant_mass=False, gear=75, reset_prob=0.25,
                        skill_penalty=0.25):

        super(PointMassLargeMJ, self).__init__(
                        epsilon_low=0.27,
                        epsilon_high=0.5,
                        action_penalty=0.5,
                        skill_penalty=skill_penalty,
                        early_low_termination=early_low_termination,
                        k=k,
                        delta_max=delta_max,
                        high_level_sparse_reward=100,
                        arena_width=8,
                        velocity_bound=0.1,
                        agent_size=agent_size,
                        ant_density=ant_density,
                        ant_mass=ant_mass,
                        gear=gear,
                        reset_prob=reset_prob)

        def viewer_setup(self):
            self.viewer.cam.distance = 20

class PointMassLargeZMJ(PointMassLargeMJ):
    FILE = '/envs/assets/point_gather_z.xml'

class PointMassDiscreteLargeMJ(PointMassLargeMJ, PointMassDiscreteMJ):
    pass

class PointMassLargeVelocityMJ(PointMassLargeMJ, PointMassVelocityMJ):
    pass

class PointMassLargeRotMJ(PointMassLargeMJ):
    FILE = '/envs/assets/point_gather_rot.xml'

class PointMassLargeRotVelocityMJ(PointMassLargeMJ, PointMassVelocityMJ):
    FILE = '/envs/assets/point_gather_rot.xml'


class PointMassPush(PointMassLargeMJ):
    FILE = '/envs/assets/point_push.xml'

    def state(self):
        return np.concatenate([
                self.sim.data.qpos.flat[2:-2],
                self.sim.data.qvel.flat[:-2],
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
            and state_vec[2] >= 8.2 and state_vec[2] <= 9.0
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
        
if __name__ == "__main__":
    from bot_transfer.envs.hierarchical import LowLevelEnv, HighLevelEnv, FullEnv
    
    pm_env = PointMassPush(agent_size=0.5, ant_density=True) #PointMassLargeMJ()
    # env = LowLevelEnv(pm_env)
    # env = HighLevelEnv(pm_env, pm_env.skill_space().sample)
    env = FullEnv(pm_env)
    # env = JointEnv(pm_env)

    env.reset()

    for _ in range(1000):
        env.render()
        obs, reward, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()
    