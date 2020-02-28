from bot_transfer.envs.hierarchical import Hierarchical, HierarchicalMJ
import numpy as np
from gym import spaces
import bot_transfer
import os

class Quadruped(HierarchicalMJ):

    FILE = '/envs/assets/quadruped_gather.xml'
    def __init__(self, k=20, early_low_termination=False, delta_max=4.0, low_epsilon=0.27, gear=None, reset_prob=0.25, include_contacts=True):
        
        self.reached_goal = False
        self.arena_width = 8
        self.high_level_sparse_reward = 100
        self.low_level_sparse_reward = 50
        self.include_contacts = include_contacts
        super(Quadruped, self).__init__(os.path.dirname(bot_transfer.__file__) + self.FILE, 3,
                                  epsilon_low=low_epsilon,
                                  epsilon_high=0.5,
                                  action_penalty=0.5,
                                  skill_penalty=0.25,
                                  early_low_termination=early_low_termination,
                                  k=k,
                                  delta_max=delta_max,
                                  reset_prob=reset_prob) 

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
        self.model.body_pos[19][:2] = self.low_level_goal
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
            self.model.body_pos[18][:2] = goal    

    def low_level_is_done(self):
        state_vec = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
        notdone = np.isfinite(state_vec).all() \
            and state_vec[2] >= 0.15 and state_vec[2] <= 0.9
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
        # Note: Lower contact reward than the Ant by a factor of 100
        # contact_cost = 0.5 * 1e-5 * np.sum(
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
        qpos = self.np_random.uniform(size=self.model.nq, low=-.1, high=.1) + self.init_qpos
        while True:
            goal = self.np_random.uniform(low=-self.arena_width, high=self.arena_width, size=2)
            if np.linalg.norm(goal) < self.arena_width:
                break
        self.model.body_pos[18][:2] = goal
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_mj_state(qpos, qvel)

    def viewer_setup(self):
        self.viewer.cam.distance = 20

if __name__ == "__main__":
    from bot_transfer.envs.hierarchical import LowLevelEnv, HighLevelEnv, FullEnv
    
    env = Quadruped()
    env = FullEnv(env)

    env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, done, _ = env.step(env.action_space.sample())
        print(reward)

        
