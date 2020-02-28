from bot_transfer.envs.hierarchical import Hierarchical, HierarchicalMJ
import numpy as np
from gym import spaces
import bot_transfer
import os

class Swimmer(HierarchicalMJ):
    FILE = '/envs/assets/swimmer_gather.xml'

    def __init__(self, k=20, early_low_termination=False, delta_max=3.0):

        super(Swimmer, self).__init__(os.path.dirname(bot_transfer.__file__) + self.FILE, 5,
                                      epsilon_low=0.3,
                                      epsilon_high=0.5,
                                      action_penalty=0.0001,
                                      skill_penalty=0.25,
                                      early_low_termination=early_low_termination,
                                      k=k,
                                      delta_max=delta_max)

        self.reached_goal = False
        self.arena_width = 8
        self.low_level_sparse_reward = 100
        self.high_level_sparse_reward = 100
        
    def state(self):
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
        self.model.body_pos[5][:2] = self.low_level_goal
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
            self.model.body_pos[4][:2] = goal   

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
        return forward_reward
        
    def high_level_is_done(self):
        return False

    def high_level_reward(self):
        if self.reached_goal:
            self.reached_goal = False
            return self.high_level_sparse_reward
        else:
            dist_to_target = np.linalg.norm(self.get_body_com("torso") - self.get_body_com("target"))
            return -1 * dist_to_target
    
    def reset_func(self, low=False):
        self.sim.reset()
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv)
        self.set_mj_state(qpos, qvel)
        while True:
            goal = self.np_random.uniform(low=-self.arena_width, high=self.arena_width, size=2)
            if np.linalg.norm(goal) < self.arena_width:
                break
        self.model.body_pos[4][:2] = goal

    def viewer_setup(self):
        self.viewer.cam.distance = 20

class SwimmerDiscreteX(Swimmer):

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

if __name__ == "__main__":
    from bot_transfer.envs.hierarchical import LowLevelEnv, HighLevelEnv, FullEnv
    
    pm_env = SwimmerDiscreteX()
    # env = LowLevelEnv(pm_env)
    # env = HighLevelEnv(pm_env, pm_env.skill_space().sample)
    env = FullEnv(pm_env)
    # env = JointEnv(pm_env)

    env.reset()

    for _ in range(3000):
        env.render()
        obs, reward, done, _ = env.step(env.action_space.sample())
        