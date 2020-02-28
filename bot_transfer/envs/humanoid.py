from bot_transfer.envs.hierarchical import Hierarchical, HierarchicalMJ
import numpy as np
from gym import spaces
import bot_transfer
import os

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

class Humanoid(HierarchicalMJ):

    FILE = '/envs/assets/humanoid_gather.xml'
    def __init__(self, k=20, early_low_termination=False, delta_max=2.5, use_mass_center=False):
        
        self.reached_goal = False
        self.arena_width = 7
        self.high_level_sparse_reward = 100
        self.low_level_sparse_reward = 50
        self.use_mass_center = use_mass_center
        super(Humanoid, self).__init__(os.path.dirname(bot_transfer.__file__) + self.FILE, 5,
                                  epsilon_low=0.27,
                                  epsilon_high=0.5,
                                  action_penalty=0.1,
                                  skill_penalty=0.25,
                                  early_low_termination=early_low_termination,
                                  k=k,
                                  delta_max=delta_max)   

    def state(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()
        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()
        # Exclude current position from qpos
        position = position[2:]
        # Get the position measurement
        if self.use_mass_center:
            location = mass_center(self.model, self.sim)
        else:
            location = self.get_body_com("torso")[:2]
        return np.concatenate((
            position,
            velocity,
            com_inertia,
            com_velocity,
            actuator_forces,
            external_contact_forces,
            location,
            self.get_body_com("target")[:2]
        ))
        
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
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return done

    def low_level_reward(self, prev_state):
        prev_delta = prev_state[-2:]
        curr_delta = self.agent_state_func(self.state())[-2:]
        # Previous delta is given by prev delta.
        # The vector of our movement is given by prev_delta - cur_delta
        # We then get the projection of our movement onto the previous delta.
        forward_reward = np.dot(prev_delta - curr_delta, prev_delta) / (np.linalg.norm(prev_delta) + 1e-6)
        forward_reward /= self.dt
        survive_reward = 5.0
        quad_impact_cost = .5e-6 * np.square(self.sim.data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)

        return 1.6*forward_reward - quad_impact_cost + survive_reward
        
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
        c = 0.01
        self.set_mj_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        while True:
            goal = self.np_random.uniform(low=-self.arena_width, high=self.arena_width, size=2)
            if np.linalg.norm(goal) < self.arena_width:
                break
        self.model.body_pos[14][:2] = goal

    def viewer_setup(self):
        self.viewer.cam.distance = 20

if __name__ == "__main__":
    from bot_transfer.envs.hierarchical import LowLevelEnv, HighLevelEnv, FullEnv
    
    env = Humanoid()
    env = LowLevelEnv(env)

    env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, done, _ = env.step(env.action_space.sample())
        # obs, reward, done, _ = env.step(np.zeros(len(env.action_space.sample())))
        # print(reward)
        