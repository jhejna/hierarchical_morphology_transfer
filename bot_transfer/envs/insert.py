from bot_transfer.envs.hierarchical import Hierarchical, HierarchicalMJ
import numpy as np
from gym import spaces
import bot_transfer
import os
import xml.etree.ElementTree as ET
import tempfile

class Insert(HierarchicalMJ):
    FILE = '/envs/assets/reacher_insert.xml'
    RAND_LIMITS = 3.0
    INSERT_DEST = np.array([0.0, -0.19])

    def __init__(self, k=20, early_low_termination=False, delta_max=0.07, reset_prob=0.25):
        
        super(Insert, self).__init__(os.path.dirname(bot_transfer.__file__) + self.FILE, 3,
                                  epsilon_low=0.015,
                                  epsilon_high=0.01,
                                  action_penalty=0.05,
                                  skill_penalty=0.05,
                                  early_low_termination=early_low_termination,
                                  k=k,
                                  delta_max=delta_max,
                                  reset_prob=reset_prob)       

    def state(self):
        theta = self.sim.data.qpos.flat[:] # remove position of target
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:],
            self.get_body_com("fingertip")[:2],
            self.get_body_com("peg")[:2]
        ])

    def agent_state_func(self, state):
        # return entire state but make peg and finger tip positions delta
        return np.concatenate([
            state[:-4],
            self.low_level_goal - state[-4:-2],
            self.low_level_goal - state[-2:]
        ])

    def obs_func(self, state):
        # Get the position of the fingertip and peg.
        return state[-4:]

    def skill_func(self, obs):
        # Return the position of the peg.
        return obs[-2:]

    def apply_action(self, action):
        self.model.body_pos[-1][:2] = self.low_level_goal
        self.do_simulation(action, self.frame_skip)

    def post_action(self):
        return

    def low_level_is_done(self):
        return False

    def low_level_reward(self, prev_state):
        dist_to_low_level_goal = np.linalg.norm(self.skill_func(self.obs_func(self.state())) - self.low_level_goal)
        reward = -1 * dist_to_low_level_goal
        if dist_to_low_level_goal < self.epsilon_low:
            reward += 4
        return reward

    def high_level_is_done(self):
        peg_pos_x, peg_pos_y = self.get_body_com("peg")[:2]
        inserted = abs(peg_pos_x - self.INSERT_DEST[0]) < 0.05 and peg_pos_y < self.INSERT_DEST[1]
        return True if inserted else False

    def high_level_reward(self):
        peg_pos = self.get_body_com("peg")[:2]
        norm_end = np.linalg.norm(peg_pos - self.INSERT_DEST)
        alpha = 1e-5
        # reward =  -(np.square(norm_end) + np.log(np.square(norm_end) + alpha))
        reward = -20*np.square(norm_end)
        if abs(peg_pos[0] - self.INSERT_DEST[0]) < 0.05 and peg_pos[1] < self.INSERT_DEST[1]:
            reward += 50
        return reward
    
    def reset_func(self, low=False):
        self.sim.reset()
        if low:
            rot_rand = self.np_random.uniform(low=-np.pi, high=np.pi, size=1)
            arm_rand = self.np_random.uniform(low=-self.RAND_LIMITS, high=self.RAND_LIMITS, size=self.model.nq-1)
            qpos = np.concatenate((rot_rand, arm_rand)) + self.init_qpos
        else:
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_mj_state(qpos, qvel)

class Insert3Link(Insert):
    FILE = '/envs/assets/reacher_insert_3_link.xml'
    RAND_LIMITS = 2.7

class Insert4Link(Insert):
    FILE = '/envs/assets/reacher_insert_4_link.xml'
    RAND_LIMITS = 2.0

class Insert3LinkLow(Insert):
    FILE = '/envs/assets/reacher_insert_3_link_low.xml'
    RAND_LIMITS = 2.7

class Insert4LinkLow(Insert):
    FILE = '/envs/assets/reacher_insert_4_link_low.xml'
    RAND_LIMITS = 2.0

class InsertPM(Insert):
    FILE = '/envs/assets/reacher_insert_pm.xml'

    def state(self):
        return np.concatenate([
            self.sim.data.qvel.flat[:],
            self.get_body_com("fingertip")[:2],
            self.get_body_com("peg")[:2]
        ])

    def reset(self, low=False):
        self.sim.reset()
        if low:
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            qpos[0] -= 0.2 
            qpos += self.init_qpos
        else:
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
            qpos[-2:] = self.init_qpos[-2:]
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_mj_state(qpos, qvel)
        self.set_skill(self.skill_space().sample())

class InsertPMLow(InsertPM):
    FILE = '/envs/assets/reacher_insert_pm_low.xml'
    
if __name__ == "__main__":
    from bot_transfer.envs.hierarchical import LowLevelEnv, HighLevelEnv, FullEnv
    
    env = InsertPMLow()
    env = LowLevelEnv(env)

    env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, done, _ = env.step(env.action_space.sample())
        if done:
            print("done")
            env.reset()
