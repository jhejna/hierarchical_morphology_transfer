from os import path
import gym
from gym import utils, spaces
from abc import ABC, abstractmethod
import numpy as np

try:
    import mujoco_py
except ImportError as e:
    raise gym.error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class Hierarchical(ABC):

    def __init__(self, epsilon_low=0.01,
                       epsilon_high=0.01, 
                       action_penalty=0.05,
                       skill_penalty=0.05,
                       early_low_termination=False,
                       k=10,
                       delta_max=1,
                       absolute_skill=False,
                       reset_prob=1.0,
                       max_sequential_low=10,
                       ):
        self.seed()
        self.early_low_termination = early_low_termination
        self.low_level_goal = None
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.action_penalty = action_penalty
        self.skill_penalty = skill_penalty
        self.delta_max = delta_max
        self.k = k
        self.absolute_skill = absolute_skill
        self.reset_prob = reset_prob
        self.max_sequential_low = max_sequential_low
        self.num_sequential_low = 0
        self.current_skill = None

    @abstractmethod
    def obs_func(self, state):
        '''
        Takes in environment state and returns the agent agnostic observation
        '''
        return NotImplemented

    @abstractmethod
    def agent_state_func(self, state):
        '''
        Takes in environment state and returns the agent specific state
        '''
        return state

    @abstractmethod
    def skill_func(self, observation):
        '''
        Takes in an agent agnostic state and returns the associated skill to reach that state
        '''
        return NotImplemented

    @property
    @abstractmethod
    def state(self):
        return NotImplemented

    @property
    @abstractmethod
    def action_space(self):
        return NotImplemented

    @abstractmethod
    def high_level_is_done(self):
        return NotImplemented

    @abstractmethod
    def high_level_reward(self):
        return NotImplemented

    @abstractmethod
    def apply_action(self, action):
        return NotImplemented

    @abstractmethod
    def low_level_reward(self, prev_state):
        return NotImplemented

    @abstractmethod
    def low_level_is_done(self):
        return NotImplemented    
    
    def post_action(self):
        pass

    def skill_map(self):
        return NotImplemented

    def state_space(self):
        state = self.state()
        low = np.full(state.shape, -float('inf'))
        high = np.full(state.shape, float('inf'))
        return spaces.Box(low, high, dtype=state.dtype)

    def agent_state_space(self):
        low_level_state = self.agent_state_func(self.state())
        low = np.full(low_level_state.shape, -float('inf'))
        high = np.full(low_level_state.shape, float('inf'))
        return spaces.Box(low, high, dtype=low_level_state.dtype)

    def observation_space(self):
        ob = self.obs_func(self.state())
        low = np.full(ob.shape, -float('inf'))
        high = np.full(ob.shape, float('inf'))
        return spaces.Box(low, high, dtype=ob.dtype)

    def skill_space(self):
        skill_map = self.skill_map()
        if skill_map == NotImplemented:
            skill_ob = self.skill_func(self.obs_func(self.state()))
            low = np.full(skill_ob.shape, -self.delta_max)
            high = np.full(skill_ob.shape, self.delta_max)
            return spaces.Box(low, high, dtype=skill_ob.dtype)
        else:
            return spaces.Discrete(len(skill_map))

    def low_level_step(self, action):
        # print("Contacts:")
        # for arr, obj in zip(self.sim.data.cfrc_ext, self.sim.data.contact):
        #     print(obj.geom1, obj.geom2, arr)

        prev_state = self.agent_state_func(self.state())
        self.apply_action(action)
        reward = self.low_level_reward(prev_state)
        
        # Control Penalty
        # print("Action Penalty", -1 * self.action_penalty * np.sum(np.square(action)))
        # print("Regular Reward", reward)
        reward += -1 * self.action_penalty * np.sum(np.square(action))
        # Check Done condition
        done = self.low_level_is_done()
        if self.early_low_termination:
            dist_to_goal = np.linalg.norm(self.skill_func(self.obs_func(self.state()))- self.low_level_goal)
            if dist_to_goal < self.epsilon_low:
                done = True
        
        self.post_action()
        return self.agent_state_func(self.state()), reward, done, dict()

    def set_skill(self, skill):
        self.current_skill = skill
        if isinstance(skill, int):
            skill = self.skill_map()[skill]
        if self.absolute_skill:
            self.low_level_goal = skill
        else:
            self.low_level_goal = self.skill_func(self.obs_func(self.state())) + skill

    def high_level_step(self, skill, predict_func, intermediate_steps=False):
        # High level action chooses the goal
        self.set_skill(skill)
        if isinstance(self.k, int):
            num_low_level_steps = self.k
        elif len(self.k) == 2:
            num_low_level_steps = self.np_random.uniform(low=self.k[0], high=self.k[1])
        else:
            raise Exception("specifics of k were not specified correctly")

        init_state = self.state()
        frames = []
        if intermediate_steps:
            high_obs = [self.obs_func(init_state)]
            high_actions = [skill]
            high_rewards = []
            high_dones = []
        low_state = self.agent_state_func(init_state)
        for _ in range(num_low_level_steps):
            low_action = predict_func(low_state)
            if len(low_action) > 0:
                low_action = low_action[0]
            low_state, _, low_done, _ = self.low_level_step(low_action)
            
            if intermediate_steps:
                high_ob = self.obs_func(self.state())
                high_obs.append(high_ob)
                cur_skill = self.low_level_goal - self.skill_func(high_ob)
                high_actions.append(cur_skill)
                high_reward = self.high_level_reward()
                high_reward -= -1 * self.skill_penalty * np.linalg.norm(cur_skill)
                high_rewards.append(high_reward)
                high_dones.append(self.high_level_is_done())

            # catch calculation with short circuit
            if low_done or np.linalg.norm(self.skill_func(self.obs_func(self.state()))
                                             - self.low_level_goal) < self.epsilon_low:
                break
            
            #### TO ENABLE DEBUG RENDERING ###
            # frames.append(self.render(mode='rgb_array'))
            # self.render()

        if intermediate_steps:
            return high_obs[-1], high_rewards[-1], high_dones[-1], dict(obs=high_obs, rewards=high_rewards, dones=high_dones, actions=high_actions, frames=frames)
        else:
            reward = 0
            reward += -1 * self.skill_penalty * np.linalg.norm(skill)
            reward += self.high_level_reward()
            done = self.high_level_is_done()
            return self.obs_func(self.state()), reward, done, dict(frames=frames)

    def reset(self, low=False):
        reset = not low or (low and self.low_level_is_done()) \
                        or self.np_random.random() < self.reset_prob \
                        or self.num_sequential_low == self.max_sequential_low
        if reset:
            self.reset_func(low=low)
            self.num_sequential_low = 0
        else:
            self.num_sequential_low += 1

        if hasattr(self, 'is_hcp') and self.is_hcp and self.add_extra_z:
            sampled_xyz = self.skill_space().sample()[:3]
            rand_angle = self.np_random.uniform(low=np.pi/6, high=np.pi/2, size=1)
            top_z = 0.09*np.sin(rand_angle)[0]
            assert top_z > 0.0
            self.set_skill(np.concatenate([sampled_xyz, [sampled_xyz[2] + top_z]]))
        else:
            self.set_skill(self.skill_space().sample())

    @abstractmethod
    def reset_func(self, low=False):
        '''
        For low level, make sure set a low_level_goal!!!
        '''
        return NotImplemented

    @abstractmethod
    def render(self):
        return NotImplemented

    def seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

class HierarchicalMJ(Hierarchical):

    def __init__(self, model_path, frame_skip, **kwargs):
        super(HierarchicalMJ, self).__init__(**kwargs)

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # print("ENV TYPE", type(self))
        if hasattr(self, 'is_hcp') and self.is_hcp:
            print("IN HCP ENV: stepping the sim.")
            self.sim.step()
            bounds = self.sim.model.actuator_ctrlrange[:self.NUM_JOINTS]
            self.control_low = np.copy(bounds[:, 0])
            self.control_high = np.copy(bounds[:, 1])
            if self.add_extra_z:
                sampled_xyz = self.skill_space().sample()[:3]
                rand_angle = self.np_random.uniform(low=np.pi/6, high=np.pi/2, size=1)
                top_z = 0.09*np.sin(rand_angle)[0]
                assert top_z > 0.0
                self.set_skill(np.concatenate([sampled_xyz, [sampled_xyz[2] + top_z]]))
            else:
                self.set_skill(self.skill_space().sample())
        else:
            self.set_skill(self.skill_space().sample())

        action = self.action_space().sample()
        self.low_level_step(action)

    def action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def set_mj_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self,
               mode='human',
               width=800,
               height=800,
               camera_id=-1,
               camera_name=None,
               close=False):
        if mode == 'rgb_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)


class HighLevelEnv(gym.Env):
    '''
    This yields a highl level environment parameterized by a pretrianed low level policy.
    '''
    def __init__(self, env, policy, intermediate_steps=False):
        self.intermediate_steps = intermediate_steps
        utils.EzPickle.__init__(**locals())
        if not isinstance(env, Hierarchical):
            raise ValueError("High Level Env must be created with a Hierarchical Environment")
        self.env = env
        if isinstance(policy, str):
            from bot_transfer.utils.loader import load_from_name
            self.low_level_policy, _ = load_from_name(policy, load_env=False, best=True)
        else:
            self.low_level_policy = policy
        # print("LOADED", self.low_level_policy)
        self.observation_space = self.env.observation_space()
        self.action_space = self.env.skill_space()

    def step(self, skill):
        return self.env.high_level_step(skill, self.low_level_policy.predict, intermediate_steps=self.intermediate_steps)

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        return self.env.obs_func(self.env.state())

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)

class LowLevelEnv(gym.Env, utils.EzPickle):
    '''
    This env trains exclusively on the low level poriton of the environment
    '''
    def __init__(self, env):
        utils.EzPickle.__init__(**locals())
        if not isinstance(env, Hierarchical):
            raise ValueError("Low Level Env must be created with a Hierarchical Environment")
        self.env = env
        self.observation_space = self.env.agent_state_space()
        self.action_space = self.env.action_space()

    def step(self, action):
        return self.env.low_level_step(action)

    def reset(self, *args, **kwargs):
        self.env.reset(*args, low=True, **kwargs)
        return self.env.agent_state_func(self.env.state())

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)


class FullEnv(gym.Env, utils.EzPickle):
    '''
    This env uses low level actions, but the high level reward signal
    '''

    def __init__(self, env):
        utils.EzPickle.__init__(**locals())
        if not isinstance(env, Hierarchical):
                raise ValueError("Must be created with a Hierarchical Environment")
        self.env = env
        self.observation_space = self.env.state_space()
        self.action_space = self.env.action_space()
        self.env.action_penalty = 0.0

    def step(self, action):
        _, _, _, _ = self.env.low_level_step(action)
        reward = self.env.high_level_reward()
        reward += -1 * self.env.action_penalty * np.linalg.norm(action)
        done = self.env.high_level_is_done()

        return self.env.state(), reward, done, dict()
        
    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        return self.env.state()
    
    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)

class DiscriminatorEnv(gym.Env, utils.EzPickle):
    '''
    This env runs another alternate low level environment for collecting hierarchical data
    '''

    def __init__(self, env, policy, params=None, discrim_early_low_term=False, best=True, discrim_time_limit=None, discrim_online=True):
        utils.EzPickle.__init__(**locals())
        if not isinstance(env, Hierarchical):
            raise ValueError("Must be created with a Hierarchical Environment")
        # Get the discriminator params
        if not params:
            from bot_transfer.utils.loader import ModelParams
            discrim_params = ModelParams.load(policy)
        else:
            discrim_params = params
        discrim_params['env'] = discrim_params['env'].split('_')[0]
        # Avoid possible recursion
        if 'discrim_policy' in discrim_params['env_wrapper_args']:
            del discrim_params['env_wrapper_args']['discrim_policy']
        if not discrim_time_limit:
            discrim_time_limit = discrim_params['time_limit']
        discrim_params['time_limit'] = None
        # get the discriminator Environment
        from bot_transfer.utils.loader import get_env
        self.expert_env = get_env(discrim_params)
        if not isinstance(self.expert_env, Hierarchical):
            raise ValueError("Expert Environment must also be Hierarchical")
        # Get the expert policy
        if isinstance(policy, str):
            from bot_transfer.utils.loader import load
            expert_model, _ = load(policy, discrim_params, load_env=False, best=best)
        else:
            expert_model = policy

        self.env = env
        self.observation_space = self.env.agent_state_space()
        self.action_space = self.env.action_space()
        # option for expert env to auto terminate when we reach the goal.
        self.expert_env.early_low_termination = discrim_early_low_term
        self.expert_pred_fn = expert_model.predict
        self.expert_state = self.expert_env.agent_state_func(self.expert_env.state())
        self.expert_time_limit = discrim_time_limit
        self.expert_time_step = 0
        self.prev_agent_obs = None
        self.prev_expert_obs = None
        self.discrim_online = discrim_online

    def step(self, action):
        state, reward, done, _ = self.env.low_level_step(action)
        expert_action = self.expert_pred_fn(self.expert_state)
        # Step the Expert Policy:
        self.expert_state, _, expert_done, _ = self.expert_env.low_level_step(expert_action[0])
        agent_obs = self.env.low_level_goal - self.env.skill_func(self.env.obs_func(self.env.state()))
        expert_obs = self.expert_env.low_level_goal - self.expert_env.skill_func(self.expert_env.obs_func(self.expert_env.state()))
        self.expert_time_step += 1

        info = dict(prev_agent_obs=self.prev_agent_obs, prev_expert_obs=self.prev_expert_obs, 
                    agent_obs=agent_obs, expert_obs=expert_obs, skill=self.env.current_skill)
        self.prev_agent_obs = agent_obs
        self.prev_expert_obs = expert_obs

        if expert_done or self.expert_time_step == self.expert_time_limit:
            self.expert_env.reset()
            self.expert_env.set_skill(self.env.current_skill)
            self.expert_state = self.expert_env.agent_state_func(self.expert_env.state())
            self.expert_time_step = 0
        return state, reward, done, info

    def reset(self, *args, **kwargs):
        self.env.reset(*args, low=True, **kwargs)
        self.expert_env.reset(low=True)
        if self.discrim_online:
            self.expert_env.set_skill(self.env.current_skill)
        
        self.expert_state = self.expert_env.agent_state_func(self.expert_env.state())
        
        self.prev_agent_obs = self.env.low_level_goal - self.env.skill_func(self.env.obs_func(self.env.state()))
        self.prev_expert_obs = self.expert_env.low_level_goal - self.expert_env.skill_func(self.expert_env.obs_func(self.expert_env.state()))

        self.expert_time_step = 0
        return self.env.agent_state_func(self.env.state())
        
    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)


class JointAC1Env(gym.Env):
    '''
    This yields an Environment designed for data collection for AC Agents.
    '''
    def __init__(self, env):
        utils.EzPickle.__init__(**locals())
        if not isinstance(env, Hierarchical):
            raise ValueError("High Level Env must be created with a Hierarchical Environment")
        self.env = env
        self.observation_space = self.env.observation_space()
        self.action_space = self.env.skill_space()
        self.low_level_action_space = self.env.action_space()
        self.low_level_observation_space = self.env.agent_state_space()

    def step(self, skill_predict_func):
        # print("Predict Skill Func", skill_predict_func)
        skill, predict_func = skill_predict_func
        # High level action chooses the goal
        self.env.set_skill(skill)
        if isinstance(self.env.k, int):
            num_low_level_steps = self.env.k
        elif len(self.env.k) == 2:
            num_low_level_steps = self.env.np_random.uniform(low=self.env.k[0], high=self.env.k[1])
        else:
            raise Exception("specifics of k were not specified correctly")

        low_states = []
        low_starts = []
        low_actions = []
        low_rewards = []
        low_dones = []
        low_vpreds = []
        low_start = True

        for _ in range(num_low_level_steps):
            low_ob = self.env.agent_state_func(self.env.state())
            low_action, low_vpred, _, _ = predict_func(low_ob.reshape(-1, *low_ob.shape))

            low_states.append(low_ob)
            low_actions.append(low_action[0])
            low_vpreds.append(low_vpred[0])
            low_starts.append(low_start)
            
            if isinstance(self.low_level_action_space, gym.spaces.Box):
                low_action = np.clip(low_action, self.low_level_action_space.low, self.low_level_action_space.high)

            _, low_reward, low_done, _ = self.env.low_level_step(low_action[0])
            
            low_rewards.append(low_reward)
            low_dones.append(low_done)
            low_start = low_done

            if low_done:
                break

        # Set the last low done to be done as we finished the low level step
        low_dones[-1] = True
    
        reward = 0
        reward += -1*self.env.skill_penalty * np.linalg.norm(skill)
        reward += self.env.high_level_reward()
        done = self.env.high_level_is_done()
        return self.env.obs_func(self.env.state()), reward, done, dict(states=low_states, actions=low_actions, rewards=low_rewards, 
                                                               dones=low_dones, starts=low_starts, vpreds=low_vpreds)

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        return self.env.obs_func(self.env.state())

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)

class JointAC2Env(gym.Env):
    '''
    This yields an Environment designed for data collection for AC Agents.
    '''
    def __init__(self, env):
        utils.EzPickle.__init__(**locals())
        if not isinstance(env, Hierarchical):
            raise ValueError("High Level Env must be created with a Hierarchical Environment")
        self.env = env
        self.observation_space = self.env.observation_space()
        self.action_space = self.env.skill_space()
        self.low_level_action_space = self.env.action_space()
        self.low_level_observation_space = self.env.agent_state_space()

    def step(self, skill_predict_func):
        # print("Predict Skill Func", skill_predict_func)
        skill, predict_func = skill_predict_func
        # High level action chooses the goal
        self.env.set_skill(skill)
        if isinstance(self.env.k, int):
            num_low_level_steps = self.env.k
        elif len(self.env.k) == 2:
            num_low_level_steps = self.env.np_random.uniform(low=self.env.k[0], high=self.env.k[1])
        else:
            raise Exception("specifics of k were not specified correctly")

        low_obs = []
        low_actions = []
        low_values = []
        low_neglogpacs = []
        low_rewards = []
        low_dones = []
        low_done = False
        low_ob = self.env.agent_state_func(self.env.state())
        for _ in range(num_low_level_steps):
            low_action, low_value, _, low_neglogpac = predict_func(low_ob.reshape(-1, *low_ob.shape), None, [low_done])
            low_obs.append(low_ob)
            low_actions.append(low_action)
            low_values.append(low_value)
            low_neglogpacs.append(low_neglogpac)
            low_dones.append(low_done)
            
            if isinstance(self.low_level_action_space, gym.spaces.Box):
                low_action = np.clip(low_action, self.low_level_action_space.low, self.low_level_action_space.high)

            low_ob, low_reward, low_done, _ = self.env.low_level_step(low_action[0])
            low_rewards.append(low_reward)

            if low_done:
                break
            
        # Set the FIRST to be done. Why? We start collecting data right after the last episode ended.
        # This is consistent with the high level data collection and DummyVecEnv
        low_dones[0] = True
        low_ep_info = {'r' : sum(low_rewards), 'l' : len(low_dones)}
        reward = 0
        reward += -1*self.env.skill_penalty * np.linalg.norm(skill)
        reward += self.env.high_level_reward()
        done = self.env.high_level_is_done()
        return self.env.obs_func(self.env.state()), reward, done, dict(obs=low_obs, actions=low_actions, rewards=low_rewards,
                                                                        values=low_values, neglogpacs=low_neglogpacs, dones=low_dones, low_ep_info=low_ep_info)

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        return self.env.obs_func(self.env.state())

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)

class JointOPEnv(gym.Env):
    '''
    This yields an Environment designed for data collection for AC Agents.
    '''
    def __init__(self, env):
        utils.EzPickle.__init__(**locals())
        if not isinstance(env, Hierarchical):
            raise ValueError("High Level Env must be created with a Hierarchical Environment")
        self.env = env
        self.observation_space = self.env.observation_space()
        self.action_space = self.env.skill_space()
        self.low_level_action_space = self.env.action_space()
        self.low_level_observation_space = self.env.agent_state_space()

    def step(self, skill_predict_func):
        # print("Predict Skill Func", skill_predict_func)
        skill, predict_func = skill_predict_func
        # High level action chooses the goal

        self.env.set_skill(skill)
        if isinstance(self.env.k, int):
            num_low_level_steps = self.env.k
        elif len(self.env.k) == 2:
            num_low_level_steps = self.env.np_random.uniform(low=self.env.k[0], high=self.env.k[1])
        else:
            raise Exception("specifics of k were not specified correctly")

        low_obs = []
        low_actions = []
        low_rewards = []
        low_dones = []
        low_done = False
        low_ob = self.env.agent_state_func(self.env.state())
        for _ in range(num_low_level_steps):
            low_action, low_unscaled_action = predict_func(low_ob)
            low_obs.append(low_ob)
            low_actions.append(low_action)

            if isinstance(self.low_level_action_space, gym.spaces.Box):
                low_unscaled_action = np.clip(low_unscaled_action, self.low_level_action_space.low, self.low_level_action_space.high)

            low_ob, low_reward, low_done, _ = self.env.low_level_step(low_unscaled_action)
            low_dones.append(low_done)
            low_rewards.append(low_reward)

            if low_done:
                break

        # Append the last s_{t+1}
        low_obs.append(low_ob)
        # Set the last low done to true
        low_dones[-1] = True
        
        low_ep_info = {'r' : sum(low_rewards), 'l' : len(low_dones)}
        reward = 0
        reward += -1*self.env.skill_penalty * np.linalg.norm(skill)
        reward += self.env.high_level_reward()
        done = self.env.high_level_is_done()
        return self.env.obs_func(self.env.state()), reward, done, dict(obs=low_obs, actions=low_actions, rewards=low_rewards,
                                                                        dones=low_dones, low_ep_info=low_ep_info)

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        return self.env.obs_func(self.env.state())

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)

class LowFinetuneEnv(gym.Env, utils.EzPickle):
    '''
    This env samples subgoals from the high level when training.
    '''
    def __init__(self, env, policy, finetune_time_limit=None):
        utils.EzPickle.__init__(**locals())
        if not isinstance(env, Hierarchical):
            raise ValueError("Low Level Env must be created with a Hierarchical Environment")
        self.env = env
        # Enforce that the low level environment won't reset
        self.env.reset_prob = 0.0
        self.observation_space = self.env.agent_state_space()
        self.action_space = self.env.action_space()
        # Load the high level policy
        if isinstance(policy, str):
            from bot_transfer.utils.loader import load_from_name
            self.high_level_policy, _ = load_from_name(policy, load_env=False, best=True)
        else:
            self.high_level_policy = policy
        self.high_level_time_limit = finetune_time_limit
        self.high_level_step_num = 0

    def step(self, action):
        return self.env.low_level_step(action)

    def reset(self, *args, **kwargs):
        # We don't reset based on the low level.
        # self.env.reset(*args, low=True, **kwargs)
        # Select the skill we will exectute
        self.high_level_step_num += 1
        high_done = self.env.high_level_is_done()
        if high_done or (self.high_level_time_limit and self.high_level_time_limit == self.high_level_step_num):
            # Do an actual reset.
            self.env.reset(*args, **kwargs)
            self.high_level_step_num = 0

        high_level_obs = self.env.obs_func(self.env.state())
        # Get the first high level componnent.
        skill = self.high_level_policy.predict(high_level_obs)[0]
        self.env.set_skill(skill)
        return self.env.agent_state_func(self.env.state())

    def render(self, mode='human', close=False):
        return self.env.render(mode=mode, close=close)
