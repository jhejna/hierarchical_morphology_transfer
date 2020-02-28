from bot_transfer.envs.hierarchical import HierarchicalMJ
import numpy as np
from gym import spaces
import bot_transfer
import os
import xml.etree.ElementTree as ET
import tempfile

class Reacher(HierarchicalMJ):

    FILE = '/envs/assets/reacher_push.xml'
    PUSH_DEST = np.array([-0.10, -0.08])
    RAND_LIMITS = 3.0

    def __init__(self, k=20, early_low_termination=False, delta_max=0.07, reset_prob=0.25):
        
        self.push_destination = self.PUSH_DEST

        xml_path = os.path.dirname(bot_transfer.__file__) + self.FILE
        tree = ET.parse(xml_path)
        dest_elem = tree.find('.//worldbody/body[@name="dest"]')
        dest_elem.set("pos", str(self.PUSH_DEST[0]) + " " + str(self.PUSH_DEST[1]) + " 0.02")
        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)
        
        super(Reacher, self).__init__(file_path, 3,
                                  epsilon_low=0.01,
                                  epsilon_high=0.025,
                                  action_penalty=0.05,
                                  skill_penalty=0.05,
                                  early_low_termination=early_low_termination,
                                  k=k,
                                  delta_max=delta_max,
                                  reset_prob=reset_prob)       

    def state(self):
        theta = self.sim.data.qpos.flat[:-2] # remove position of target
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qvel.flat[:],
            self.get_body_com("fingertip")[:2],
            self.get_body_com("target")[:2]
        ])

    def agent_state_func(self, state):
        # override the goal position and remove location. Remember target velocity
        return np.concatenate([
            state[:-6],
            self.low_level_goal - state[-4:-2]
        ])

    def obs_func(self, state):
        # Return just the current position and the target plus its velocity
        return state[-6:]

    def skill_func(self, observation):
        # return just the current position
        return observation[2:4]

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
        target_pos = self.get_body_com("target")[:2]
        dist_to_target = np.linalg.norm(target_pos - self.push_destination)
        if dist_to_target < self.epsilon_high:
            return True
        else:
            return False

    def high_level_reward(self):
        target_pos = self.get_body_com("target")[:2]
        dist_to_target = np.linalg.norm(target_pos - self.push_destination)
        if dist_to_target < self.epsilon_high:
            return 200
        else:
            return -1*dist_to_target
    
    def reset_func(self, low=False):
        self.sim.reset()
        if low:
            rot_rand = self.np_random.uniform(low=-np.pi, high=np.pi, size=1)
            arm_rand = self.np_random.uniform(low=-self.RAND_LIMITS, high=self.RAND_LIMITS, size=self.model.nq-1)
            qpos = np.concatenate((rot_rand, arm_rand)) + self.init_qpos
            qpos[-2:] = np.array([0.26, 0.26])
        else:
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
            qpos[-2:] = self.init_qpos[-2:]
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_mj_state(qpos, qvel)

class Reacher3Link(Reacher):
    FILE = '/envs/assets/reacher_push_3_link.xml'
    RAND_LIMITS = 2.7

class Reacher4Link(Reacher):
    FILE = '/envs/assets/reacher_push_4_link.xml'
    RAND_LIMITS = 2.0

class ReacherPM(Reacher):
    FILE = '/envs/assets/reacher_pm.xml'

    def state(self):
        return np.concatenate([
            self.sim.data.qvel.flat[:],
            self.get_body_com("fingertip")[:2],
            self.get_body_com("target")[:2]
        ])

    def reset(self, low=False):
        self.sim.reset()
        if low:
            qpos = self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nq)
            qpos[0] -= 0.2 
            qpos += self.init_qpos
            qpos[-2:] = np.array([0.26, 0.26])
        else:
            qpos = self.np_random.uniform(low=-0.05, high=0.05, size=self.model.nq) + self.init_qpos
            qpos[-2:] = self.init_qpos[-2:]
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_mj_state(qpos, qvel)
        self.set_skill(self.skill_space().sample())

class ReacherHard(Reacher):
    PUSH_DEST = np.array([-0.12, -0.13])

class Reacher3LinkHard(Reacher3Link):
    PUSH_DEST = np.array([-0.12, -0.13])

class Reacher4LinkHard(Reacher4Link):
    PUSH_DEST = np.array([-0.12, -0.13])

class ReacherPMHard(ReacherPM):
    PUSH_DEST = np.array([-0.12, -0.13])

if __name__ == "__main__":
    from bot_transfer.envs.hierarchical import LowLevelEnv, HighLevelEnv, FullEnv
    
    env = Reacher3LinkHard()
    env = FullEnv(env)

    env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, done, _ = env.step(env.action_space.sample())
        if done:
            print("done")
            env.reset()