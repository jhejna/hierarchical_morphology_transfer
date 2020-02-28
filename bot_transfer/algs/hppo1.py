from collections import deque
import time

import gym
import tensorflow as tf
import numpy as np
from mpi4py import MPI

from stable_baselines.common import Dataset, explained_variance, fmt_row, zipsame, SetVerbosity, \
    TensorboardWriter
from stable_baselines import logger
import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.mpi_moments import mpi_moments

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common.vec_env import VecEnv

from bot_transfer.algs.hrl import ActorCriticHRLModel


def flatten_lists(listoflists):
    """
    Flatten a python list of list
    :param listoflists: (list(list))
    :return: (list)
    """
    return [el for list_ in listoflists for el in list_]

def traj_segment_generator(low_policy, high_policy, env, horizon):

    # Initialize state variables
    step = 0
    action = env.action_space.sample()  # not used, just so we have the datatype
    observation = env.reset()

    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    current_ep_len = 0 # len of current episode
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    # Initialize history arrays
    observations = np.array([observation for _ in range(horizon)])
    rewards = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    episode_starts = np.zeros(horizon, 'bool')
    dones = np.zeros(horizon, 'bool')
    actions = np.array([action for _ in range(horizon)])
    states = high_policy.initial_state
    episode_start = True  # marks if we're on first timestep of an episode
    done = False

    low_obs = []
    low_rewards = []
    low_vpreds = []
    low_episode_starts = []
    low_dones = []
    low_actions = []

    low_ep_rets = []
    low_ep_lens = []

    while True:
        action, vpred, states, _ = high_policy.step(observation.reshape(-1, *observation.shape), states, done)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
            yield {
                    "low_obs": np.array(low_obs),
                    "low_rewards": np.array(low_rewards),
                    "low_dones": np.array(low_dones),
                    "low_episode_starts": np.array(low_episode_starts),
                    "low_true_rewards": np.array(low_rewards),
                    "low_vpred": np.array(low_vpreds),
                    "low_actions": np.array(low_actions),
                    "low_nextvpred" : 0, # Just set this to zero for now as we always complete low level.
                    "low_ep_rets": low_ep_rets,
                    "low_ep_lens": low_ep_lens,
                    "high_obs": observations,
                    "high_rewards": rewards,
                    "high_true_rewards": rewards,
                    "high_dones": dones,
                    "high_episode_starts": episode_starts,
                    "high_vpred": vpreds,
                    "high_actions": actions,
                    "high_nextvpred": vpred[0] * (1 - episode_start),
                    "high_ep_rets": ep_rets,
                    "high_ep_lens": ep_lens,
                    "total_timestep": current_it_len
            }
            _, vpred, _, _ = high_policy.step(observation.reshape(-1, *observation.shape))
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []

            # Reset all low level data collection.
            low_obs = []
            low_rewards = []
            low_vpreds = []
            low_episode_starts = []
            low_dones = []
            low_actions = []

            low_ep_rets = []
            low_ep_lens = []
            # Reset current iteration length
            current_it_len = 0

        i = step % horizon
        observations[i] = observation
        vpreds[i] = vpred[0]
        actions[i] = action[0]
        episode_starts[i] = episode_start

        clipped_action = action
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        observation, reward, done, info = env.step((clipped_action[0], low_policy.step))

        # add all the low level information collected
        low_obs.extend(info['states'])
        low_rewards.extend(info['rewards'])
        low_vpreds.extend(info['vpreds'])
        low_episode_starts.extend(info['starts'])
        low_dones.extend(info['dones'])
        low_actions.extend(info['actions'])

        low_ep_rets.append(sum(info['rewards']))
        low_ep_lens.append(len(info['rewards']))
        
        rewards[i] = reward
        dones[i] = done
        episode_start = done

        cur_ep_ret += reward
        current_it_len += 1
        current_ep_len += 1
        if done:
            # Retrieve unnormalized reward if using Monitor wrapper
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(current_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            current_ep_len = 0
            if not isinstance(env, VecEnv):
                observation = env.reset()
        step += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts = np.append(seg["high_episode_starts"], False)
    vpred = np.append(seg["high_vpred"], seg["high_nextvpred"])
    rew_len = len(seg["high_rewards"])
    seg["high_adv"] = np.empty(rew_len, 'float32')
    rewards = seg["high_rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        seg["high_adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["high_tdlamret"] = seg["high_adv"] + seg["high_vpred"]

    episode_starts = np.append(seg["low_episode_starts"], False)
    vpred = np.append(seg["low_vpred"], seg["low_nextvpred"])
    rew_len = len(seg["low_rewards"])
    seg["low_adv"] = np.empty(rew_len, 'float32')
    rewards = seg["low_rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        seg["low_adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["low_tdlamret"] = seg["low_adv"] + seg["low_vpred"]


class HPPO1(ActorCriticHRLModel):
    """
    Proximal Policy Optimization algorithm (MPI version).
    Paper: https://arxiv.org/abs/1707.06347
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param timesteps_per_actorbatch: (int) timesteps per actor per update
    :param clip_param: (float) clipping parameter epsilon
    :param entcoeff: (float) the entropy loss weight
    :param optim_epochs: (float) the optimizer's number of epochs
    :param optim_stepsize: (float) the optimizer's stepsize
    :param optim_batchsize: (int) the optimizer's the batch size
    :param gamma: (float) discount factor
    :param lam: (float) advantage estimation
    :param adam_epsilon: (float) the epsilon value for the adam optimizer
    :param schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
        'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, low_policy, high_policy, env, gamma=0.99, timesteps_per_actorbatch=256,
                       low_clip_param=0.2, high_clip_param=0.2, 
                       low_entcoeff=0.01, high_entcoeff=0.01,
                       optim_epochs=4,
                       low_optim_stepsize=1e-3, low_optim_batchsize=64, lam=0.95, low_adam_epsilon=1e-5,
                       high_optim_stepsize=1e-3, high_optim_batchsize=64, high_adam_epsilon=1e-5,
                        schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True,
                        low_policy_kwargs=None, high_policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1):

        super(HPPO1, self).__init__(low_policy=low_policy, high_policy=high_policy, env=env, verbose=verbose, requires_vec_env=False,
                         _init_setup_model=_init_setup_model, low_policy_kwargs=low_policy_kwargs, high_policy_kwargs=high_policy_kwargs,
                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.gamma = gamma
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.optim_epochs = optim_epochs
        self.schedule = schedule
        self.lam = lam
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        # Low level parameters
        self.low_clip_param = low_clip_param
        self.low_entcoeff = low_entcoeff
        self.low_optim_stepsize = low_optim_stepsize
        self.low_optim_batchsize = low_optim_batchsize
        self.low_adam_epsilon = low_adam_epsilon

        # high level parameters
        self.high_clip_param = high_clip_param
        self.high_entcoeff = high_entcoeff
        self.high_optim_stepsize = high_optim_stepsize
        self.high_optim_batchsize = high_optim_batchsize
        self.high_adam_epsilon = high_adam_epsilon
        
        self.graph = None
        self.sess = None
        # duplicate policy network
        self.low_policy_pi = None
        self.high_policy_pi = None
        # For Low Level
        self.low_loss_names = None
        self.low_lossandgrad = None
        self.low_adam = None
        self.low_assign_old_eq_new = None
        self.low_compute_losses = None
        self.low_params = None
        self.low_step = None
        self.low_proba_step = None
        self.low_initial_state = None
        self.low_summary = None
        self.low_episode_reward = None

        # For High Level
        self.high_loss_names = None
        self.high_lossandgrad = None
        self.high_adam = None
        self.high_assign_old_eq_new = None
        self.high_compute_losses = None
        self.high_params = None
        self.high_step = None
        self.high_proba_step = None
        self.high_initial_state = None
        self.high_summary = None
        self.high_episode_reward = None

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        return NotImplemented

    def setup_model(self):
        with SetVerbosity(self.verbose):

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)


                with tf.variable_scope("low", reuse=False):

                    low_summary_vars = list()

                    # Construct network for new policy
                    self.low_policy_pi = self.low_policy(self.sess, self.low_observation_space, self.low_action_space, self.n_envs, 1,
                                             None, reuse=False, **self.low_policy_kwargs)

                    # Network for old policy
                    with tf.variable_scope("oldpi", reuse=False):
                        old_pi = self.low_policy(self.sess, self.low_observation_space, self.low_action_space, self.n_envs, 1,
                                            None, reuse=False, **self.low_policy_kwargs)

                    with tf.variable_scope("loss", reuse=False):
                        # Target advantage function (if applicable)
                        atarg = tf.placeholder(dtype=tf.float32, shape=[None])

                        # Empirical return
                        ret = tf.placeholder(dtype=tf.float32, shape=[None])

                        # learning rate multiplier, updated with schedule
                        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])

                        # Annealed cliping parameter epislon
                        clip_param = self.low_clip_param * lrmult

                        obs_ph = self.low_policy_pi.obs_ph
                        action_ph = self.low_policy_pi.pdtype.sample_placeholder([None])

                        kloldnew = old_pi.proba_distribution.kl(self.low_policy_pi.proba_distribution)
                        ent = self.low_policy_pi.proba_distribution.entropy()
                        meankl = tf.reduce_mean(kloldnew)
                        meanent = tf.reduce_mean(ent)
                        pol_entpen = (-self.low_entcoeff) * meanent

                        # pnew / pold
                        ratio = tf.exp(self.low_policy_pi.proba_distribution.logp(action_ph) -
                                    old_pi.proba_distribution.logp(action_ph))

                        # surrogate from conservative policy iteration
                        surr1 = ratio * atarg
                        surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg

                        # PPO's pessimistic surrogate (L^CLIP)
                        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
                        vf_loss = tf.reduce_mean(tf.square(self.low_policy_pi.value_flat - ret))
                        total_loss = pol_surr + pol_entpen + vf_loss
                        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
                        self.low_loss_names = ["low_pol_surr", "low_pol_entpen", "low_vf_loss", "low_kl", "low_ent"]

                        low_summary_vars.append(tf.summary.scalar('low_entropy_loss', pol_entpen))
                        low_summary_vars.append(tf.summary.scalar('low_policy_gradient_loss', pol_surr))
                        low_summary_vars.append(tf.summary.scalar('low_value_function_loss', vf_loss))
                        low_summary_vars.append(tf.summary.scalar('low_approximate_kullback-leibler', meankl))
                        low_summary_vars.append(tf.summary.scalar('low_clip_factor', clip_param))
                        low_summary_vars.append(tf.summary.scalar('low_loss', total_loss))

                        self.low_params = tf_util.get_trainable_vars("low/model")

                        self.low_assign_old_eq_new = tf_util.function(
                            [], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                            zipsame(tf_util.get_globals_vars("low/oldpi"), tf_util.get_globals_vars("low/model"))])

                    with tf.variable_scope("Adam_mpi", reuse=False):
                        self.low_adam = MpiAdam(self.low_params, epsilon=self.low_adam_epsilon, sess=self.sess)

                    with tf.variable_scope("input_info", reuse=False):
                        low_summary_vars.append(tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret)))
                        low_summary_vars.append(tf.summary.scalar('learning_rate', tf.reduce_mean(self.low_optim_stepsize)))
                        low_summary_vars.append(tf.summary.scalar('advantage', tf.reduce_mean(atarg)))
                        low_summary_vars.append(tf.summary.scalar('clip_range', tf.reduce_mean(self.low_clip_param)))

                        if self.full_tensorboard_log:
                            low_summary_vars.append(tf.summary.histogram('discounted_rewards', ret))
                            low_summary_vars.append(tf.summary.histogram('learning_rate', self.low_optim_stepsize))
                            low_summary_vars.append(tf.summary.histogram('advantage', atarg))
                            low_summary_vars.append(tf.summary.histogram('clip_range', self.low_clip_param))
                            if tf_util.is_image(self.high_observation_space):
                                low_summary_vars.append(tf.summary.image('observation', obs_ph))
                            else:
                                low_summary_vars.append(tf.summary.histogram('observation', obs_ph))

                    self.low_step = self.low_policy_pi.step
                    self.low_proba_step = self.low_policy_pi.proba_step
                    self.low_initial_state = self.low_policy_pi.initial_state

                    self.low_summary = tf.summary.merge(low_summary_vars)

                    self.low_lossandgrad = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                                                        [self.low_summary, tf_util.flatgrad(total_loss, self.low_params)] + losses)
                    self.low_compute_losses = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                                                       losses)

                    low_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='low')
                    self.sess.run(tf.variables_initializer(low_vars))

                with tf.variable_scope("high", reuse=False):

                    high_summary_vars = list()

                    # Construct network for new policy
                    self.high_policy_pi = self.high_policy(self.sess, self.high_observation_space, self.high_action_space, self.n_envs, 1,
                                             None, reuse=False, **self.high_policy_kwargs)

                    # Network for old policy
                    with tf.variable_scope("oldpi", reuse=False):
                        old_pi = self.high_policy(self.sess, self.high_observation_space, self.high_action_space, self.n_envs, 1,
                                            None, reuse=False, **self.high_policy_kwargs)

                    with tf.variable_scope("loss", reuse=False):
                        # Target advantage function (if applicable)
                        atarg = tf.placeholder(dtype=tf.float32, shape=[None])

                        # Empirical return
                        ret = tf.placeholder(dtype=tf.float32, shape=[None])

                        # learning rate multiplier, updated with schedule
                        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])

                        # Annealed cliping parameter epislon
                        clip_param = self.high_clip_param * lrmult

                        obs_ph = self.high_policy_pi.obs_ph
                        action_ph = self.high_policy_pi.pdtype.sample_placeholder([None])

                        kloldnew = old_pi.proba_distribution.kl(self.high_policy_pi.proba_distribution)
                        ent = self.high_policy_pi.proba_distribution.entropy()
                        meankl = tf.reduce_mean(kloldnew)
                        meanent = tf.reduce_mean(ent)
                        pol_entpen = (-self.high_entcoeff) * meanent

                        # pnew / pold
                        ratio = tf.exp(self.high_policy_pi.proba_distribution.logp(action_ph) -
                                    old_pi.proba_distribution.logp(action_ph))

                        # surrogate from conservative policy iteration
                        surr1 = ratio * atarg
                        surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg

                        # PPO's pessimistic surrogate (L^CLIP)
                        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
                        vf_loss = tf.reduce_mean(tf.square(self.high_policy_pi.value_flat - ret))
                        total_loss = pol_surr + pol_entpen + vf_loss
                        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
                        self.high_loss_names = ["high_pol_surr", "high_pol_entpen", "high_vf_loss", "high_kl", "high_ent"]

                        high_summary_vars.append(tf.summary.scalar('high_entropy_loss', pol_entpen))
                        high_summary_vars.append(tf.summary.scalar('high_policy_gradient_loss', pol_surr))
                        high_summary_vars.append(tf.summary.scalar('high_value_function_loss', vf_loss))
                        high_summary_vars.append(tf.summary.scalar('high_approximate_kullback-leibler', meankl))
                        high_summary_vars.append(tf.summary.scalar('high_clip_factor', clip_param))
                        high_summary_vars.append(tf.summary.scalar('high_loss', total_loss))

                        self.high_params = tf_util.get_trainable_vars("high/model")

                        self.high_assign_old_eq_new = tf_util.function(
                            [], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                            zipsame(tf_util.get_globals_vars("high/oldpi"), tf_util.get_globals_vars("high/model"))])

                    with tf.variable_scope("Adam_mpi", reuse=False):
                        self.high_adam = MpiAdam(self.high_params, epsilon=self.high_adam_epsilon, sess=self.sess)

                    with tf.variable_scope("input_info", reuse=False):
                        high_summary_vars.append(tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret)))
                        high_summary_vars.append(tf.summary.scalar('learning_rate', tf.reduce_mean(self.high_optim_stepsize)))
                        high_summary_vars.append(tf.summary.scalar('advantage', tf.reduce_mean(atarg)))
                        high_summary_vars.append(tf.summary.scalar('clip_range', tf.reduce_mean(self.high_clip_param)))

                        if self.full_tensorboard_log:
                            high_summary_vars.append(tf.summary.histogram('discounted_rewards', ret))
                            high_summary_vars.append(tf.summary.histogram('learning_rate', self.high_optim_stepsize))
                            high_summary_vars.append(tf.summary.histogram('advantage', atarg))
                            high_summary_vars.append(tf.summary.histogram('clip_range', self.high_clip_param))
                            if tf_util.is_image(self.high_observation_space):
                                high_summary_vars.append(tf.summary.image('observation', obs_ph))
                            else:
                                high_summary_vars.append(tf.summary.histogram('observation', obs_ph))

                    self.high_step = self.high_policy_pi.step
                    self.high_proba_step = self.high_policy_pi.proba_step
                    self.high_initial_state = self.high_policy_pi.initial_state

                    self.high_summary = tf.summary.merge(high_summary_vars)

                    self.high_lossandgrad = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                                                        [self.high_summary, tf_util.flatgrad(total_loss, self.high_params)] + losses)
                    self.high_compute_losses = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                                                       losses)

                    high_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='high')
                    self.sess.run(tf.variables_initializer(high_vars))
                

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="PPO1",
              reset_num_timesteps=True, high_training_starts=0):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            assert issubclass(self.low_policy, ActorCriticPolicy), "Error: the input policy for the PPO1 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."
            assert issubclass(self.high_policy, ActorCriticPolicy), "Error: the input policy for the PPO1 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            with self.sess.as_default():
                self.low_adam.sync()
                self.high_adam.sync()

                # Prepare for rollouts
                seg_gen = traj_segment_generator(self.low_policy_pi, self.high_policy_pi, self.env, self.timesteps_per_actorbatch)

                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()

                # rolling buffer for episode lengths
                high_lenbuffer = deque(maxlen=100)
                # rolling buffer for episode rewards
                high_rewbuffer = deque(maxlen=100)

                # rolling buffer for episode lengths
                low_lenbuffer = deque(maxlen=100)
                # rolling buffer for episode rewards
                low_rewbuffer = deque(maxlen=100)

                self.low_episode_reward = np.zeros((self.n_envs,))
                self.high_episode_reward = np.zeros((self.n_envs,))

                while True:
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break
                    if total_timesteps and timesteps_so_far >= total_timesteps:
                        break

                    if self.schedule == 'constant':
                        cur_lrmult = 1.0
                    elif self.schedule == 'linear':
                        cur_lrmult = max(1.0 - float(timesteps_so_far) / total_timesteps, 0)
                    else:
                        raise NotImplementedError

                    logger.log("********** Iteration %i ************" % iters_so_far)

                    seg = seg_gen.__next__()
                    add_vtarg_and_adv(seg, self.gamma, self.lam)

                    # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                    high_observations, high_actions = seg["high_obs"], seg["high_actions"]
                    high_atarg, high_tdlamret = seg["high_adv"], seg["high_tdlamret"]

                    low_observations, low_actions = seg["low_obs"], seg["low_actions"]
                    low_atarg, low_tdlamret = seg["low_adv"], seg["low_tdlamret"]

                    # true_rew is the reward without discount
                    if writer is not None:
                        self.low_episode_reward = total_episode_reward_logger(self.low_episode_reward,
                                                                          seg["low_true_rewards"].reshape((self.n_envs, -1)),
                                                                          seg["low_dones"].reshape((self.n_envs, -1)),
                                                                          writer, self.num_timesteps)
                        self.high_episode_reward = total_episode_reward_logger(self.high_episode_reward,
                                                                          seg["high_true_rewards"].reshape((self.n_envs, -1)),
                                                                          seg["high_dones"].reshape((self.n_envs, -1)),
                                                                          writer, self.num_timesteps)

                    # predicted value function before udpate
                    high_vpredbefore = seg["high_vpred"]
                    low_vpredbefore = seg["low_vpred"]

                    # standardized advantage function estimate
                    high_atarg = (high_atarg - high_atarg.mean()) / high_atarg.std()
                    low_atarg = (low_atarg - low_atarg.mean()) / low_atarg.std()

                    high_dataset = Dataset(dict(ob=high_observations, ac=high_actions, atarg=high_atarg, vtarg=high_tdlamret),
                                      shuffle=not self.high_policy.recurrent)
                    low_dataset = Dataset(dict(ob=low_observations, ac=low_actions, atarg=low_atarg, vtarg=low_tdlamret),
                                      shuffle=not self.low_policy.recurrent)

                    high_optim_batchsize = self.high_optim_batchsize or observations.shape[0]
                    low_optim_batchsize = self.low_optim_batchsize or observations.shape[0]

                    # set old parameter values to new parameter values
                    self.low_assign_old_eq_new(sess=self.sess)
                    self.high_assign_old_eq_new(sess=self.sess)
                    logger.log("Optimizing...")
                    logger.log(fmt_row(13, self.low_loss_names))
                    logger.log(fmt_row(13, self.high_loss_names))

                    # Here we do a bunch of optimization epochs over the data
                    for k in range(self.optim_epochs):
                        # list of tuples, each of which gives the loss for a minibatch
                        low_losses = []
                        for i, batch in enumerate(low_dataset.iterate_once(low_optim_batchsize)):
                            steps = (self.num_timesteps +
                                     k * low_optim_batchsize +
                                     int(i * (low_optim_batchsize / len(low_dataset.data_map))))
                            if writer is not None:
                                # run loss backprop with summary, but once every 10 runs save the metadata
                                # (memory, compute time, ...)
                                if self.full_tensorboard_log and (1 + k) % 10 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, grad, *newlosses = self.low_lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess,
                                                                                 options=run_options,
                                                                                 run_metadata=run_metadata)
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                else:
                                    summary, grad, *newlosses = self.low_lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *newlosses = self.low_lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                       batch["atarg"], batch["vtarg"], cur_lrmult,
                                                                       sess=self.sess)
                            self.low_adam.update(grad, self.low_optim_stepsize * cur_lrmult)
                            low_losses.append(newlosses)
                        logger.log(fmt_row(13, np.mean(low_losses, axis=0)))

                        high_losses = []
                        for i, batch in enumerate(high_dataset.iterate_once(high_optim_batchsize)):
                            steps = (self.num_timesteps +
                                     k * high_optim_batchsize +
                                     int(i * (high_optim_batchsize / len(high_dataset.data_map))))


                            if writer is not None:
                                # run loss backprop with summary, but once every 10 runs save the metadata
                                # (memory, compute time, ...)

                                if self.full_tensorboard_log and (1 + k) % 10 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, grad, *newlosses = self.high_lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess,
                                                                                 options=run_options,
                                                                                 run_metadata=run_metadata)
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                else:
                                    summary, grad, *newlosses = self.high_lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *newlosses = self.high_lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                       batch["atarg"], batch["vtarg"], cur_lrmult,
                                                                       sess=self.sess)
                            
                            self.high_adam.update(grad, self.high_optim_stepsize * cur_lrmult)
                            high_losses.append(newlosses)
                        logger.log(fmt_row(13, np.mean(high_losses, axis=0)))

                    logger.log("Evaluating losses...")

                    low_losses = []
                    for batch in low_dataset.iterate_once(low_optim_batchsize):
                        newlosses = self.low_compute_losses(batch["ob"], batch["ob"], batch["ac"], batch["atarg"],
                                                        batch["vtarg"], cur_lrmult, sess=self.sess)
                        low_losses.append(newlosses)
                    low_mean_losses, _, _ = mpi_moments(low_losses, axis=0)
                    logger.log(fmt_row(13, low_mean_losses))
                    for (loss_val, name) in zipsame(low_mean_losses, self.low_loss_names):
                        logger.record_tabular("low_loss_" + name, loss_val)
                    logger.record_tabular("low_ev_tdlam_before", explained_variance(low_vpredbefore, low_tdlamret))

                    high_losses = []
                    for batch in high_dataset.iterate_once(high_optim_batchsize):
                        newlosses = self.high_compute_losses(batch["ob"], batch["ob"], batch["ac"], batch["atarg"],
                                                        batch["vtarg"], cur_lrmult, sess=self.sess)
                        high_losses.append(newlosses)
                    high_mean_losses, _, _ = mpi_moments(high_losses, axis=0)
                    logger.log(fmt_row(13, high_mean_losses))
                    for (loss_val, name) in zipsame(high_mean_losses, self.high_loss_names):
                        logger.record_tabular("high_loss_" + name, loss_val)
                    logger.record_tabular("high_ev_tdlam_before", explained_variance(high_vpredbefore, high_tdlamret))

                    # local values
                    high_lrlocal = (seg["high_ep_lens"], seg["high_ep_rets"])
                    low_lrlocal = (seg["low_ep_lens"], seg["low_ep_rets"])
                    
                    # list of tuples
                    high_listoflrpairs = MPI.COMM_WORLD.allgather(high_lrlocal)
                    low_listoflrpairs = MPI.COMM_WORLD.allgather(low_lrlocal)

                    h_lens, h_rews = map(flatten_lists, zip(*high_listoflrpairs))
                    l_lens, l_rews = map(flatten_lists, zip(*low_listoflrpairs))

                    high_lenbuffer.extend(h_lens)
                    high_rewbuffer.extend(h_rews)
                    low_lenbuffer.extend(l_lens)
                    low_rewbuffer.extend(l_rews)
                    
                    if len(high_lenbuffer) > 0:
                        logger.record_tabular("High_EpLenMean", np.mean(high_lenbuffer))
                        logger.record_tabular("High_EpRewMean", np.mean(high_rewbuffer))
                    if len(low_lenbuffer) > 0:
                        logger.record_tabular("Low_EpLenMean", np.mean(low_lenbuffer))
                        logger.record_tabular("Low_EpRewMean", np.mean(low_rewbuffer))

                    logger.record_tabular("High_EpThisIter", len(h_lens))
                    logger.record_tabular("Low_EpThisIter", len(l_lens))
                    episodes_so_far += len(h_lens)
                    current_it_timesteps = MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                    timesteps_so_far += current_it_timesteps
                    self.num_timesteps += current_it_timesteps
                    iters_so_far += 1
                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", self.num_timesteps)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)
                    if self.verbose >= 1 and MPI.COMM_WORLD.Get_rank() == 0:
                        logger.dump_tabular()

        return self

    def save(self, low_save_path, high_save_path, cloudpickle=False):
        low_data = {
            "gamma": self.gamma,
            "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
            "clip_param": self.low_clip_param,
            "entcoeff": self.low_entcoeff,
            "optim_epochs": self.optim_epochs,
            "optim_stepsize": self.low_optim_stepsize,
            "optim_batchsize": self.low_optim_batchsize,
            "lam": self.lam,
            "adam_epsilon": self.low_adam_epsilon,
            "schedule": self.schedule,
            "verbose": self.verbose,
            "policy": self.low_policy,
            "observation_space": self.low_observation_space,
            "action_space": self.low_action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.low_policy_kwargs
        }

        high_data = {
            "gamma": self.gamma,
            "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
            "clip_param": self.high_clip_param,
            "entcoeff": self.high_entcoeff,
            "optim_epochs": self.optim_epochs,
            "optim_stepsize": self.high_optim_stepsize,
            "optim_batchsize": self.high_optim_batchsize,
            "lam": self.lam,
            "adam_epsilon": self.high_adam_epsilon,
            "schedule": self.schedule,
            "verbose": self.verbose,
            "policy": self.high_policy,
            "observation_space": self.high_observation_space,
            "action_space": self.high_action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.high_policy_kwargs
        }
        
        low_params_to_save, high_params_to_save = self.get_parameters(trim_prefix=True)

        self._save_to_file(low_save_path, data=low_data, params=low_params_to_save, cloudpickle=cloudpickle)
        self._save_to_file(high_save_path, data=high_data, params=high_params_to_save, cloudpickle=cloudpickle)

if __name__ == "__main__":
    # Run tests
    from stable_baselines.common.policies import MlpPolicy

    from bot_transfer.envs.hierarchical import JointACEnv
    from bot_transfer.envs.point_mass import PointMassSmallVelocityMJ, PointMassSmallMJ
    
    env = JointACEnv(PointMassSmallMJ(k=15))

    from gym.wrappers import TimeLimit
    env = TimeLimit(env, 50)

    # from stable_baselines.common.vec_env import DummyVecEnv
    # env = DummyVecEnv([lambda: env])
    
    model = HPPO1(MlpPolicy, MlpPolicy, env, tensorboard_log="test_hppo", seed=1, verbose=1)
    model.learn(total_timesteps=150000, log_interval=5)
    # model.save("test")

    # loaded_test = HPPO1.load("test")
