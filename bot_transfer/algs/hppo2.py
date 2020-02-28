import time
import sys
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger

from bot_transfer.algs.hrl import ActorCriticHRLModel


class HPPO2(ActorCriticHRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
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
    def __init__(self, low_policy, high_policy, env,
                low_gamma=0.99, high_gamma=0.99,
                low_n_steps=128, high_n_steps=128,
                low_ent_coef=0.01, high_ent_coef=0.01,
                low_learning_rate=2.5e-4, high_learning_rate=2.5e-4,
                low_vf_coef=0.5, high_vf_coef=0.5,
                low_max_grad_norm=0.5, high_max_grad_norm=0.5,
                low_lam=0.95, high_lam=0.95,
                low_nminibatches=16, high_nminibatches=4,
                low_noptepochs=4, high_noptepochs=4,
                low_cliprange=0.2, high_cliprange=0.2,
                low_cliprange_vf=None, high_cliprange_vf=None,
                verbose=0, tensorboard_log=None, _init_setup_model=True,
                low_policy_kwargs=None, high_policy_kwargs=None,
                full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        super(HPPO2, self).__init__(low_policy=low_policy, high_policy=high_policy, env=env, verbose=verbose, requires_vec_env=True,
                                   _init_setup_model=_init_setup_model, low_policy_kwargs=low_policy_kwargs, high_policy_kwargs=high_policy_kwargs,
                                   seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        # Low Params
        self.low_learning_rate = low_learning_rate
        self.low_cliprange = low_cliprange
        self.low_cliprange_vf = low_cliprange_vf
        self.low_ent_coef = low_ent_coef
        self.low_vf_coef = low_vf_coef
        self.low_max_grad_norm = low_max_grad_norm
        self.low_gamma = low_gamma
        self.low_lam = low_lam
        self.low_nminibatches = low_nminibatches
        self.low_noptepochs = low_noptepochs

        # High Params
        self.high_learning_rate = high_learning_rate
        self.high_cliprange = high_cliprange
        self.high_cliprange_vf = high_cliprange_vf
        self.high_ent_coef = high_ent_coef
        self.high_vf_coef = high_vf_coef
        self.high_max_grad_norm = high_max_grad_norm
        self.high_gamma = high_gamma
        self.high_lam = high_lam
        self.high_nminibatches = high_nminibatches
        self.high_noptepochs = high_noptepochs

        # Agnostic Params (FOR NOW)
        self.n_steps = high_n_steps

        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        # Graph Parameters
        self.graph = None
        self.sess = None

        # Low Network Parameters
        self.low_action_ph = None
        self.low_advs_ph = None
        self.low_rewards_ph = None
        self.low_old_neglog_pac_ph = None
        self.low_old_vpred_ph = None
        self.low_learning_rate_ph = None
        self.low_clip_range_ph = None
        self.low_entropy = None
        self.low_vf_loss = None
        self.low_pg_loss = None
        self.low_approxkl = None
        self.low_clipfrac = None
        self.low_params = None
        self._low_train = None
        self.low_loss_names = None
        self.low_train_model = None
        self.low_act_model = None
        self.low_step = None
        self.low_proba_step = None
        self.low_value = None
        self.low_initial_state = None
        self.low_summary = None
        self.low_episode_reward = None

        # High Network Parameters
        self.high_action_ph = None
        self.high_advs_ph = None
        self.high_rewards_ph = None
        self.high_old_neglog_pac_ph = None
        self.high_old_vpred_ph = None
        self.high_learning_rate_ph = None
        self.high_clip_range_ph = None
        self.high_entropy = None
        self.high_vf_loss = None
        self.high_pg_loss = None
        self.high_approxkl = None
        self.high_clipfrac = None
        self.high_params = None
        self._high_train = None
        self.high_loss_names = None
        self.high_train_model = None
        self.high_act_model = None
        self.high_step = None
        self.high_proba_step = None
        self.high_value = None
        self.high_initial_state = None
        self.high_summary = None
        self.high_episode_reward = None

        # Unsure
        self.high_n_batch = None
        self.low_n_batch = None
        

        if _init_setup_model:
            self.setup_model()

    ''' REMOVE PRETRAINING SUPPORT
    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action
    '''

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.low_policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."
            assert issubclass(self.high_policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.high_n_batch = self.n_envs * self.n_steps
            self.low_n_batches = None            

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                with tf.variable_scope("low", reuse=False):
                    n_batch_step = None
                    n_batch_train = None
                    low_summary_vars = list()

                    act_model = self.low_policy(self.sess, self.low_observation_space, self.low_action_space, self.n_envs, 1,
                                            n_batch_step, reuse=False, **self.low_policy_kwargs)
                    with tf.variable_scope("train_model", reuse=True,
                                        custom_getter=tf_util.outer_scope_getter("train_model")):
                        train_model = self.low_policy(self.sess, self.low_observation_space, self.low_action_space,
                                                self.n_envs // self.low_nminibatches, self.n_steps, n_batch_train,
                                                reuse=True, **self.low_policy_kwargs)

                    with tf.variable_scope("loss", reuse=False):
                        self.low_action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                        self.low_advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                        self.low_rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                        self.low_old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                        self.low_old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                        self.low_learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                        self.low_clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                        neglogpac = train_model.proba_distribution.neglogp(self.low_action_ph)
                        self.low_entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                        vpred = train_model.value_flat

                        # Value function clipping: not present in the original PPO
                        if self.low_cliprange_vf is None:
                            # Default behavior (legacy from OpenAI baselines):
                            # use the same clipping as for the policy
                            self.low_clip_range_vf_ph = self.low_clip_range_ph
                            self.low_cliprange_vf = self.low_cliprange
                        elif isinstance(self.low_cliprange_vf, (float, int)) and self.low_cliprange_vf < 0:
                            # Original PPO implementation: no value function clipping
                            self.low_clip_range_vf_ph = None
                        else:
                            # Last possible behavior: clipping range
                            # specific to the value function
                            self.low_clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

                        if self.low_clip_range_vf_ph is None:
                            # No clipping
                            vpred_clipped = train_model.value_flat
                        else:
                            # Clip the different between old and new value
                            # NOTE: this depends on the reward scaling
                            vpred_clipped = self.low_old_vpred_ph + \
                                tf.clip_by_value(train_model.value_flat - self.low_old_vpred_ph,
                                                - self.low_clip_range_vf_ph, self.low_clip_range_vf_ph)

                        vf_losses1 = tf.square(vpred - self.low_rewards_ph)
                        vf_losses2 = tf.square(vpred_clipped - self.low_rewards_ph)
                        self.low_vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                        ratio = tf.exp(self.low_old_neglog_pac_ph - neglogpac)
                        pg_losses = -self.low_advs_ph * ratio
                        pg_losses2 = -self.low_advs_ph * tf.clip_by_value(ratio, 1.0 - self.low_clip_range_ph, 1.0 +
                                                                    self.low_clip_range_ph)
                        self.low_pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                        self.low_approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.low_old_neglog_pac_ph))
                        self.low_clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                        self.low_clip_range_ph), tf.float32))
                        loss = self.low_pg_loss - self.low_entropy * self.low_ent_coef + self.low_vf_loss * self.low_vf_coef

                        low_summary_vars.append(tf.summary.scalar('low_entropy_loss', self.low_entropy))
                        low_summary_vars.append(tf.summary.scalar('low_policy_gradient_loss', self.low_pg_loss))
                        low_summary_vars.append(tf.summary.scalar('low_value_function_loss', self.low_vf_loss))
                        low_summary_vars.append(tf.summary.scalar('low_approximate_kullback-leibler', self.low_approxkl))
                        low_summary_vars.append(tf.summary.scalar('low_clip_factor', self.low_clipfrac))
                        low_summary_vars.append(tf.summary.scalar('low_loss', loss))

                        with tf.variable_scope('model'):
                            self.low_params = tf_util.get_trainable_vars("low/model")
                            if self.full_tensorboard_log:
                                for var in self.low_params:
                                    low_summary_vars.append(tf.summary.histogram(var.name, var))
                        grads = tf.gradients(loss, self.low_params)
                        if self.low_max_grad_norm is not None:
                            grads, _grad_norm = tf.clip_by_global_norm(grads, self.low_max_grad_norm)
                        grads = list(zip(grads, self.low_params))
                    trainer = tf.train.AdamOptimizer(learning_rate=self.low_learning_rate_ph, epsilon=1e-5)
                    self._low_train = trainer.apply_gradients(grads)

                    self.low_loss_names = ['low_policy_loss', 'low_value_loss', 'low_policy_entropy', 'low_approxkl', 'low_clipfrac']

                    with tf.variable_scope("input_info", reuse=False):
                        low_summary_vars.append(tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.low_rewards_ph)))
                        low_summary_vars.append(tf.summary.scalar('learning_rate', tf.reduce_mean(self.low_learning_rate_ph)))
                        low_summary_vars.append(tf.summary.scalar('advantage', tf.reduce_mean(self.low_advs_ph)))
                        low_summary_vars.append(tf.summary.scalar('clip_range', tf.reduce_mean(self.low_clip_range_ph)))
                        if self.low_clip_range_vf_ph is not None:
                            low_summary_vars.append(tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.low_clip_range_vf_ph)))

                        low_summary_vars.append(tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.low_old_neglog_pac_ph)))
                        low_summary_vars.append(tf.summary.scalar('old_value_pred', tf.reduce_mean(self.low_old_vpred_ph)))

                        if self.full_tensorboard_log:
                            low_summary_vars.append(tf.summary.histogram('discounted_rewards', self.low_rewards_ph))
                            low_summary_vars.append(tf.summary.histogram('learning_rate', self.low_learning_rate_ph))
                            low_summary_vars.append(tf.summary.histogram('advantage', self.low_advs_ph))
                            low_summary_vars.append(tf.summary.histogram('clip_range', self.low_clip_range_ph))
                            low_summary_vars.append(tf.summary.histogram('old_neglog_action_probabilty', self.low_old_neglog_pac_ph))
                            low_summary_vars.append(tf.summary.histogram('old_value_pred', self.low_old_vpred_ph))
                            if tf_util.is_image(self.low_observation_space):
                                low_summary_vars.append(tf.summary.image('observation', train_model.obs_ph))
                            else:
                               low_summary_vars.append(tf.summary.histogram('observation', train_model.obs_ph))

                    self.low_train_model = train_model
                    self.low_act_model = act_model
                    self.low_step = act_model.step
                    self.low_proba_step = act_model.proba_step
                    self.low_value = act_model.value
                    self.low_initial_state = act_model.initial_state
                    # TODO: Going to need to change this initializer
                    low_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='low')
                    self.sess.run(tf.variables_initializer(low_vars))

                    self.low_summary = tf.summary.merge(low_summary_vars)

                with tf.variable_scope("high", reuse=False):
                    n_batch_step = None
                    n_batch_train = None
                    high_summary_vars = list()

                    act_model = self.high_policy(self.sess, self.high_observation_space, self.high_action_space, self.n_envs, 1,
                                            n_batch_step, reuse=False, **self.high_policy_kwargs)
                    with tf.variable_scope("train_model", reuse=True,
                                        custom_getter=tf_util.outer_scope_getter("train_model")):
                        train_model = self.high_policy(self.sess, self.high_observation_space, self.high_action_space,
                                                self.n_envs // self.high_nminibatches, self.n_steps, n_batch_train,
                                                reuse=True, **self.high_policy_kwargs)

                    with tf.variable_scope("loss", reuse=False):
                        self.high_action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                        self.high_advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                        self.high_rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                        self.high_old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                        self.high_old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                        self.high_learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                        self.high_clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                        neglogpac = train_model.proba_distribution.neglogp(self.high_action_ph)
                        self.high_entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                        vpred = train_model.value_flat

                        # Value function clipping: not present in the original PPO
                        if self.high_cliprange_vf is None:
                            # Default behavior (legacy from OpenAI baselines):
                            # use the same clipping as for the policy
                            self.high_clip_range_vf_ph = self.high_clip_range_ph
                            self.high_cliprange_vf = self.high_cliprange
                        elif isinstance(self.high_cliprange_vf, (float, int)) and self.high_cliprange_vf < 0:
                            # Original PPO implementation: no value function clipping
                            self.high_clip_range_vf_ph = None
                        else:
                            # Last possible behavior: clipping range
                            # specific to the value function
                            self.high_clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

                        if self.high_clip_range_vf_ph is None:
                            # No clipping
                            vpred_clipped = train_model.value_flat
                        else:
                            # Clip the different between old and new value
                            # NOTE: this depends on the reward scaling
                            vpred_clipped = self.high_old_vpred_ph + \
                                tf.clip_by_value(train_model.value_flat - self.high_old_vpred_ph,
                                                - self.high_clip_range_vf_ph, self.high_clip_range_vf_ph)


                        vf_losses1 = tf.square(vpred - self.high_rewards_ph)
                        vf_losses2 = tf.square(vpred_clipped - self.high_rewards_ph)
                        self.high_vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                        ratio = tf.exp(self.high_old_neglog_pac_ph - neglogpac)
                        pg_losses = -self.high_advs_ph * ratio
                        pg_losses2 = -self.high_advs_ph * tf.clip_by_value(ratio, 1.0 - self.high_clip_range_ph, 1.0 +
                                                                    self.high_clip_range_ph)
                        self.high_pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                        self.high_approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.high_old_neglog_pac_ph))
                        self.high_clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                        self.high_clip_range_ph), tf.float32))
                        loss = self.high_pg_loss - self.high_entropy * self.high_ent_coef + self.high_vf_loss * self.high_vf_coef

                        high_summary_vars.append(tf.summary.scalar('entropy_loss', self.high_entropy))
                        high_summary_vars.append(tf.summary.scalar('policy_gradient_loss', self.high_pg_loss))
                        high_summary_vars.append(tf.summary.scalar('value_function_loss', self.high_vf_loss))
                        high_summary_vars.append(tf.summary.scalar('approximate_kullback-leibler', self.high_approxkl))
                        high_summary_vars.append(tf.summary.scalar('clip_factor', self.high_clipfrac))
                        high_summary_vars.append(tf.summary.scalar('loss', loss))

                        with tf.variable_scope('model'):
                            # TODO: change trainable varibales
                            self.high_params = tf_util.get_trainable_vars("high/model")
                            print("HIGH PARAMS", self.high_params)
                            
                            if self.full_tensorboard_log:
                                for var in self.high_params:
                                    high_summary_vars.append(tf.summary.histogram(var.name, var))
                        grads = tf.gradients(loss, self.high_params)
                        if self.high_max_grad_norm is not None:
                            grads, _grad_norm = tf.clip_by_global_norm(grads, self.high_max_grad_norm)
                        grads = list(zip(grads, self.high_params))
                    trainer = tf.train.AdamOptimizer(learning_rate=self.high_learning_rate_ph, epsilon=1e-5)
                    self._high_train = trainer.apply_gradients(grads)

                    self.high_loss_names = ['high_policy_loss', 'high_value_loss', 'high_policy_entropy', 'high_approxkl', 'high_clipfrac']

                    with tf.variable_scope("input_info", reuse=False):
                        high_summary_vars.append(tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.high_rewards_ph)))
                        high_summary_vars.append(tf.summary.scalar('learning_rate', tf.reduce_mean(self.high_learning_rate_ph)))
                        high_summary_vars.append(tf.summary.scalar('advantage', tf.reduce_mean(self.high_advs_ph)))
                        high_summary_vars.append(tf.summary.scalar('clip_range', tf.reduce_mean(self.high_clip_range_ph)))
                        if self.high_clip_range_vf_ph is not None:
                            high_summary_vars.append(tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.high_clip_range_vf_ph)))

                        high_summary_vars.append(tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.high_old_neglog_pac_ph)))
                        high_summary_vars.append(tf.summary.scalar('old_value_pred', tf.reduce_mean(self.high_old_vpred_ph)))

                        if self.full_tensorboard_log:
                            high_summary_vars.append(tf.summary.histogram('discounted_rewards', self.high_rewards_ph))
                            high_summary_vars.append(tf.summary.histogram('learning_rate', self.high_learning_rate_ph))
                            high_summary_vars.append(tf.summary.histogram('advantage', self.high_advs_ph))
                            high_summary_vars.append(tf.summary.histogram('clip_range', self.high_clip_range_ph))
                            high_summary_vars.append(tf.summary.histogram('old_neglog_action_probabilty', self.high_old_neglog_pac_ph))
                            high_summary_vars.append(tf.summary.histogram('old_value_pred', self.high_old_vpred_ph))
                            if tf_util.is_image(self.high_observation_space):
                                high_summary_vars.append(tf.summary.image('observation', train_model.obs_ph))
                            else:
                                high_summary_vars.append(tf.summary.histogram('observation', train_model.obs_ph))

                    self.high_train_model = train_model
                    self.high_act_model = act_model
                    self.high_step = act_model.step
                    self.high_proba_step = act_model.proba_step
                    self.high_value = act_model.value
                    self.high_initial_state = act_model.initial_state
                    # TODO
                    high_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='high')
                    self.sess.run(tf.variables_initializer(high_vars))

                    self.high_summary = tf.summary.merge(high_summary_vars)

    def _low_train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {self.low_train_model.obs_ph: obs, self.low_action_ph: actions,
                  self.low_advs_ph: advs, self.low_rewards_ph: returns,
                  self.low_learning_rate_ph: learning_rate, self.low_clip_range_ph: cliprange,
                  self.low_old_neglog_pac_ph: neglogpacs, self.low_old_vpred_ph: values}
        if states is not None:
            td_map[self.low_train_model.states_ph] = states
            td_map[self.low_train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.low_clip_range_vf_ph] = cliprange_vf

        assert states is None
        if states is None:
            update_fac = self.low_n_batch // self.low_nminibatches // self.low_noptepochs + 1
        else:
            update_fac = self.low_n_batch // self.low_nminibatches // self.low_noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.low_summary, self.low_pg_loss, self.low_vf_loss, self.low_entropy, self.low_approxkl, self.low_clipfrac, self._low_train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.low_summary, self.low_pg_loss, self.low_vf_loss, self.low_entropy, self.low_approxkl, self.low_clipfrac, self._low_train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.low_pg_loss, self.low_vf_loss, self.low_entropy, self.low_approxkl, self.low_clipfrac, self._low_train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def _high_train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {self.high_train_model.obs_ph: obs, self.high_action_ph: actions,
                  self.high_advs_ph: advs, self.high_rewards_ph: returns,
                  self.high_learning_rate_ph: learning_rate, self.high_clip_range_ph: cliprange,
                  self.high_old_neglog_pac_ph: neglogpacs, self.high_old_vpred_ph: values}
        if states is not None:
            td_map[self.high_train_model.states_ph] = states
            td_map[self.high_train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.high_clip_range_vf_ph] = cliprange_vf
        
        assert states is None
        if states is None:
            update_fac = self.high_n_batch // self.high_nminibatches // self.high_noptepochs + 1
        else:
            update_fac = self.high_n_batch // self.high_nminibatches // self.high_noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.high_summary, self.high_pg_loss, self.high_vf_loss, self.high_entropy, self.high_approxkl, self.high_clipfrac, self._high_train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.high_summary, self.high_pg_loss, self.high_vf_loss, self.high_entropy, self.high_approxkl, self.high_clipfrac, self._high_train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.high_pg_loss, self.high_vf_loss, self.high_entropy, self.high_approxkl, self.high_clipfrac, self._high_train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True, high_training_starts=0):
        # Transform to callable if needed
        self.low_learning_rate = get_schedule_fn(self.low_learning_rate)
        self.high_learning_rate = get_schedule_fn(self.high_learning_rate)

        self.low_cliprange = get_schedule_fn(self.low_cliprange)
        self.high_cliprange = get_schedule_fn(self.high_cliprange)

        low_cliprange_vf = get_schedule_fn(self.low_cliprange_vf)
        high_cliprange_vf = get_schedule_fn(self.high_cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, low_gamma=self.low_gamma, high_gamma=self.high_gamma, 
                            low_lam=self.low_lam, high_lam=self.high_lam)
            self.high_episode_reward = np.zeros((self.n_envs,))
            self.low_episode_reward = np.zeros((self.n_envs,))
            self.low_timesteps = 0

            high_ep_info_buf = deque(maxlen=100)
            low_ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            n_updates = total_timesteps // self.high_n_batch

            for update in range(1, n_updates + 1):
                
                # TODO: Perhaps modify for a low batch and a high batchsize.
                assert self.high_n_batch % self.high_nminibatches == 0
                high_batch_size = self.high_n_batch // self.high_nminibatches    

                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                low_lr_now = self.low_learning_rate(frac)
                high_lr_now = self.high_learning_rate(frac)
                low_cliprange_now = self.low_cliprange(frac)
                high_cliprange_now = self.high_cliprange(frac)
                low_cliprange_vf_now = low_cliprange_vf(frac)
                high_cliprange_vf_now = high_cliprange_vf(frac)

                # true_reward is the reward without discount
                random_high_level = True if update * self.high_n_batch < high_training_starts else False          
                low_obs, low_returns, low_masks, low_actions, low_values, low_neglogpacs, low_true_reward, low_ep_infos, \
                    high_obs, high_returns, high_masks, high_actions, high_values, high_neglogpacs, high_true_reward, high_ep_infos = runner.run(random_high_level=random_high_level)
                
                # Set Low n_batch after collecting data. Only here do we know how long the low level ran for.
                # This is due to the early termination option.
                self.low_n_batch = low_obs.shape[0]
                # assert self.low_n_batch % self.n_envs == 0
                # Update the low batch size based off of this.
                low_batch_size = self.low_n_batch // self.low_nminibatches
                
                # Update Global Timesteps
                assert self.high_n_batch == high_obs.shape[0], "Make sure n_batch is equal to total amount of data collected."
                self.num_timesteps += self.high_n_batch
                self.low_timesteps += self.low_n_batch
                high_ep_info_buf.extend(high_ep_infos)
                low_ep_info_buf.extend(low_ep_infos)

                # Update Low Level
                low_mb_loss_vals = []
                update_fac = self.low_n_batch // self.low_nminibatches // self.low_noptepochs + 1
                inds = np.arange(self.low_n_batch)
                for epoch_num in range(self.low_noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, self.low_n_batch, low_batch_size):
                        timestep = self.low_timesteps // update_fac + ((self.low_noptepochs * self.low_n_batch + epoch_num *
                                                                        self.low_n_batch + start) // low_batch_size)
                        end = start + low_batch_size
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (low_obs, low_returns, low_masks, low_actions, low_values, low_neglogpacs))
                        low_mb_loss_vals.append(self._low_train_step(low_lr_now, low_cliprange_now, *slices, writer=writer,
                                                                update=timestep, cliprange_vf=low_cliprange_vf_now))
                
                low_loss_vals = np.mean(low_mb_loss_vals, axis=0)

                # Update High Level
                high_mb_loss_vals = []
                update_fac = self.high_n_batch // self.high_nminibatches // self.high_noptepochs + 1
                inds = np.arange(self.high_n_batch)
                for epoch_num in range(self.high_noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, self.high_n_batch, high_batch_size):
                        timestep = self.num_timesteps // update_fac + ((self.high_noptepochs * self.high_n_batch + epoch_num *
                                                                        self.high_n_batch + start) // high_batch_size)
                        end = start + high_batch_size
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (high_obs, high_returns, high_masks, high_actions, high_values, high_neglogpacs))
                        high_mb_loss_vals.append(self._high_train_step(high_lr_now, high_cliprange_now, *slices, writer=writer,
                                                                update=timestep, cliprange_vf=high_cliprange_vf_now))
                
                high_loss_vals = np.mean(high_mb_loss_vals, axis=0)

                # Update Loggers
                t_now = time.time()
                fps = int(self.high_n_batch / (t_now - t_start))
                if writer is not None:
                    self.low_episode_reward = total_episode_reward_logger(self.low_episode_reward,
                                                                      low_true_reward.reshape((self.n_envs, self.low_n_batch // self.n_envs)),
                                                                      low_masks.reshape((self.n_envs, self.low_n_batch // self.n_envs)),
                                                                      writer, self.low_timesteps)
                    self.high_episode_reward = total_episode_reward_logger(self.high_episode_reward,
                                                                      high_true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      high_masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    low_explained_var = explained_variance(low_values, low_returns)
                    high_explained_var = explained_variance(high_values, high_returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("low_explained_variance", float(low_explained_var))
                    logger.logkv("high_explained_variance", float(high_explained_var))
                    if len(low_ep_info_buf) > 0 and len(low_ep_info_buf[0]) > 0:
                        logger.logkv('low_ep_reward_mean', safe_mean([ep_info['r'] for ep_info in low_ep_info_buf]))
                        logger.logkv('low_ep_len_mean', safe_mean([ep_info['l'] for ep_info in low_ep_info_buf]))
                    if len(high_ep_info_buf) > 0 and len(high_ep_info_buf[0]) > 0:
                        logger.logkv('high_ep_reward_mean', safe_mean([ep_info['r'] for ep_info in high_ep_info_buf]))
                        logger.logkv('high_ep_len_mean', safe_mean([ep_info['l'] for ep_info in high_ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(low_loss_vals, self.low_loss_names):
                        logger.logkv(loss_name, loss_val)
                    for (loss_val, loss_name) in zip(high_loss_vals, self.high_loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

            return self

    def save(self, low_save_path, high_save_path, cloudpickle=False):
        low_data = {
            "gamma": self.low_gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.low_vf_coef,
            "ent_coef": self.low_ent_coef,
            "max_grad_norm": self.low_max_grad_norm,
            "learning_rate": self.low_learning_rate,
            "lam": self.low_lam,
            "nminibatches": self.low_nminibatches,
            "noptepochs": self.low_noptepochs,
            "cliprange": self.low_cliprange,
            "cliprange_vf": self.low_cliprange_vf,
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
            "gamma": self.high_gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.high_vf_coef,
            "ent_coef": self.high_ent_coef,
            "max_grad_norm": self.high_max_grad_norm,
            "learning_rate": self.high_learning_rate,
            "lam": self.high_lam,
            "nminibatches": self.high_nminibatches,
            "noptepochs": self.high_noptepochs,
            "cliprange": self.high_cliprange,
            "cliprange_vf": self.high_cliprange_vf,
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


class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, n_steps):
        """
        A runner to learn the policy of an environment for a model
        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        """
        self.env = env
        self.model = model
        n_env = env.num_envs
        self.batch_ob_shape = (n_env*n_steps,) + env.observation_space.shape
        self.obs = np.zeros((n_env,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.n_steps = n_steps
        self.states = model.low_initial_state
        self.dones = [False for _ in range(n_env)]

    @abstractmethod
    def run(self):
        """
        Run a learning step of the model
        """
        raise NotImplementedError

class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, low_gamma, high_gamma, low_lam, high_lam):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        print("ENV TYPE", type(env))
        self.low_lam = low_lam
        self.low_gamma = low_gamma
        self.high_lam = high_lam
        self.high_gamma = high_gamma

    def run(self, random_high_level=False):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        high_mb_obs, high_mb_rewards, high_mb_actions, high_mb_values, high_mb_dones, high_mb_neglogpacs = [], [], [], [], [], []
        low_mb_obs, low_mb_rewards, low_mb_actions, low_mb_values, low_mb_dones, low_mb_neglogpacs = [], [], [], [], [], []

        low_ep_infos = []
        high_ep_infos = []

        # TODO: This only currently works when all low episodes are of the same length. If they become variable length, this no longer works.
        # TODO: This needs to be fixed eventually.
        for _ in range(self.n_steps):
            if random_high_level:
                actions = [self.env.action_space.sample() for _ in range(self.model.n_envs)]
                values = np.zeros(self.model.n_envs)
                self.states = None
                neglogpacs = np.zeros(self.model.n_envs)
                self.env.reset()
            else:
                actions, values, self.states, neglogpacs = self.model.high_step(self.obs, self.states, self.dones)
            high_mb_obs.append(self.obs.copy())
            high_mb_actions.append(actions)
            high_mb_values.append(values)
            high_mb_neglogpacs.append(neglogpacs)
            high_mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            # Vectorize the call:
            vec_actions = [(ac, self.model.low_step) for ac in clipped_actions]
            self.obs[:], rewards, self.dones, infos = self.env.step(vec_actions)
            
            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    high_ep_infos.append(maybe_ep_info)

            # Get low level data.
            if random_high_level:
                infos.sort(key=lambda x: len(x['actions']))
                # merge the infos
                new_infos = list()
                for i in range(len(infos) // 2):
                    merged_info = dict()
                    for k in infos[i].keys():
                        if k == 'low_ep_info':
                            merged_info[k] = infos[i][k]
                        else:
                            merged_info[k] = infos[i][k]
                            merged_info[k].extend(infos[2*(len(infos) // 2) - i - 1][k])
                    new_infos.append(merged_info)
                if len(infos) % 2 == 1:
                    new_infos.append(infos[-1])
                infos = new_infos
            
            num_envs = len(infos)
            # print("Low Ep Lengths", [len(infos[i]['obs']) for i in range(num_envs)])
            num_low_steps = min([len(infos[i]['obs']) for i in range(num_envs)])
            for low_step in range(num_low_steps):
                low_mb_obs.append( np.array([infos[i]['obs'][low_step] for i in range(num_envs)]) )
                low_mb_actions.append( np.array([infos[i]['actions'][low_step][0] for i in range(num_envs)]) )
                low_mb_values.append( [infos[i]['values'][low_step][0] for i in range(num_envs)] )
                low_mb_neglogpacs.append( [infos[i]['neglogpacs'][low_step][0] for i in range(num_envs)] )
                low_mb_dones.append([infos[i]['dones'][low_step] for i in range(num_envs)])
                low_mb_rewards.append([infos[i]['rewards'][low_step] for i in range(num_envs)])

            # print("Low obs shape", low_mb_obs[0].shape)
            # print(len(low_mb_obs))

            low_ep_infos.extend([info['low_ep_info'] for info in infos])
            # low_last_values = [infos[i]['last_val'] for i in range(num_envs)]

            # Add reward info
            high_mb_rewards.append(rewards)
        
        low_mb_obs = np.asarray(low_mb_obs, dtype=self.obs.dtype)
        low_mb_rewards = np.asarray(low_mb_rewards, dtype=np.float32)
        low_mb_actions = np.asarray(low_mb_actions)
        low_mb_values = np.asarray(low_mb_values, dtype=np.float32)
        low_mb_neglogpacs = np.asarray(low_mb_neglogpacs, dtype=np.float32)
        low_mb_dones = np.asarray(low_mb_dones, dtype=np.bool)
        # low_last_values = self.model.high_value(self.obs, self.states, self.dones)

        # batch of steps to batch of rollouts
        high_mb_obs = np.asarray(high_mb_obs, dtype=self.obs.dtype)
        high_mb_rewards = np.asarray(high_mb_rewards, dtype=np.float32)
        high_mb_actions = np.asarray(high_mb_actions)
        high_mb_values = np.asarray(high_mb_values, dtype=np.float32)
        high_mb_neglogpacs = np.asarray(high_mb_neglogpacs, dtype=np.float32)
        high_mb_dones = np.asarray(high_mb_dones, dtype=np.bool)
        high_last_values = self.model.high_value(self.obs, self.states, self.dones)

        # Low Level discount/bootstrap off value fn
        low_mb_advs = np.zeros_like(low_mb_rewards)
        low_true_reward = np.copy(low_mb_rewards)
        last_gae_lam = 0
        n_low_steps = low_mb_obs.shape[0]
        for step in reversed(range(n_low_steps)):
            if step == n_low_steps - 1:
                # Set to zero -- we don't collect data here.
                nextnonterminal = 0
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - low_mb_dones[step + 1]
                nextvalues = low_mb_values[step + 1]
            delta = low_mb_rewards[step] + self.low_gamma * nextvalues * nextnonterminal - low_mb_values[step]
            low_mb_advs[step] = last_gae_lam = delta + self.low_gamma * self.low_lam * nextnonterminal * last_gae_lam
        low_mb_returns = low_mb_advs + low_mb_values

        # High Level discount/bootstrap off value fn
        high_mb_advs = np.zeros_like(high_mb_rewards)
        high_true_reward = np.copy(high_mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = high_last_values
            else:
                nextnonterminal = 1.0 - high_mb_dones[step + 1]
                nextvalues = high_mb_values[step + 1]
            delta = high_mb_rewards[step] + self.high_gamma * nextvalues * nextnonterminal - high_mb_values[step]
            high_mb_advs[step] = last_gae_lam = delta + self.high_gamma * self.high_lam * nextnonterminal * last_gae_lam
        high_mb_returns = high_mb_advs + high_mb_values


        low_mb_obs, low_mb_returns, low_mb_dones, low_mb_actions, low_mb_values, low_mb_neglogpacs, low_true_reward , \
            high_mb_obs, high_mb_returns, high_mb_dones, high_mb_actions, high_mb_values, high_mb_neglogpacs, high_true_reward = \
            map(swap_and_flatten, (low_mb_obs, low_mb_returns, low_mb_dones, low_mb_actions, low_mb_values, low_mb_neglogpacs, low_true_reward,
                                   high_mb_obs, high_mb_returns, high_mb_dones, high_mb_actions, high_mb_values, high_mb_neglogpacs, high_true_reward))


        return low_mb_obs, low_mb_returns, low_mb_dones, low_mb_actions, low_mb_values, low_mb_neglogpacs, low_true_reward , low_ep_infos, \
            high_mb_obs, high_mb_returns, high_mb_dones, high_mb_actions, high_mb_values, high_mb_neglogpacs, high_true_reward, high_ep_infos


def get_schedule_fn(value_schedule):
    """
    Transform (if needed) learning rate and clip range
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constfn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)

if __name__ == "__main__":
    from stable_baselines.common.policies import MlpPolicy

    from bot_transfer.envs.hierarchical import JointAC2Env
    from bot_transfer.envs.point_mass import PointMassSmallMJ
    from gym.wrappers import TimeLimit
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines.bench import Monitor

    def make_env():
        env = JointAC2Env(PointMassSmallMJ(k=20))
        env = TimeLimit(env, 50)
        env = Monitor(env, None)
        return env

    env = DummyVecEnv([make_env for i in range(4)])
    
    model = HPPO2(MlpPolicy, MlpPolicy, env, tensorboard_log="test_hppo", seed=1, verbose=1)
    model.learn(total_timesteps=1000, log_interval=5)
    model.save("test")

    # loaded_test = HPPO1.load("test")
