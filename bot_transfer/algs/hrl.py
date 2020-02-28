from abc import ABC, abstractmethod
import os
import json
import zipfile
import glob
import cloudpickle
import numpy as np
import tensorflow as tf
import gym
import warnings
from collections import OrderedDict

from stable_baselines.common.policies import get_policy_from_name, ActorCriticPolicy
from stable_baselines.common.vec_env import VecEnvWrapper, VecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds

from stable_baselines.common.save_util import (
    is_json_serializable, data_to_json, json_to_data, params_to_bytes, bytes_to_params
)
from stable_baselines import logger


class BaseHRLModel(ABC):

    def __init__(self, low_policy, high_policy, env, verbose=0, *, requires_vec_env,
                 low_policy_base, high_policy_base, low_policy_kwargs=None,
                 high_policy_kwargs=None, seed=None, n_cpu_tf_sess=None):
        

        # Init the policies
        if isinstance(low_policy, str) and low_policy_base is not None:
            self.low_policy = get_policy_from_name(low_policy_base, low_policy)
        else:
            self.low_policy = low_policy

        if isinstance(high_policy, str) and high_policy_base is not None:
            self.high_policy = get_policy_from_name(high_policy_base, high_policy)
        else:
            self.high_policy = high_policy

        self.env = env
        self.verbose = verbose
        self._requires_vec_env = requires_vec_env

        self.low_policy_kwargs = {} if low_policy_kwargs is None else low_policy_kwargs
        self.high_policy_kwargs = {} if high_policy_kwargs is None else high_policy_kwargs

        # Define high and low level spaces
        self.low_action_space = None
        self.high_action_space = None
        self.low_observation_space = None
        self.high_observation_space = None

        self.sess = None
        self.graph = None
        self.low_params = None
        self.high_params = None
        self._param_load_ops_low = None
        self._param_load_ops_high = None

        #  This is all copied over.
        self.n_envs = None
        self._vectorize_action = False
        self.num_timesteps = 0
        self.seed = seed
        self.n_cpu_tf_sess = n_cpu_tf_sess

        if env is not None:
            if isinstance(env, str):
                if self.verbose >= 1:
                    print("Creating environment from the given name, wrapped in a DummyVecEnv.")
                self.env = env = DummyVecEnv([lambda: gym.make(env)])

            # get the low level states etc.
            try:
                self.low_observation_space = env.low_level_observation_space
                self.low_action_space = env.low_level_action_space
            except:
                self.low_observation_space = env.envs[0].low_level_observation_space
                self.low_action_space = env.envs[0].low_level_action_space

            self.high_observation_space = env.observation_space
            self.high_action_space = env.action_space

            if requires_vec_env:
                if isinstance(env, VecEnv):
                    self.n_envs = env.num_envs
                else:
                    raise ValueError("Error: the model requires a vectorized environment, please use a VecEnv wrapper.")
            else:
                if isinstance(env, VecEnv):
                    if env.num_envs == 1:
                        self.env = _UnvecWrapper(env)
                        self._vectorize_action = True
                    else:
                        raise ValueError("Error: the model requires a non vectorized environment or a single vectorized"
                                         " environment.")
                self.n_envs = 1

    def get_env(self):
        """
        returns the current environment (can be None if not defined)
        :return: (Gym Environment) The current environment
        """
        return self.env

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        :param env: (Gym Environment) The environment for learning a policy
        """
        if env is None and self.env is None:
            if self.verbose >= 1:
                print("Loading a model without an environment, "
                      "this model cannot be trained until it has a valid environment.")
            return
        elif env is None:
            raise ValueError("Error: trying to replace the current environment with None")

        # sanity checking the environment
        try:
            assert self.low_observation_space == env.low_level_observation_space, \
                "Error: the environment passed must have at least the same state space as the model was trained on."
            assert self.low_action_space == env.low_level_action_space, \
                "Error: the environment passed must have at least the same action space as the model was trained on."
        except:
            print("SPACES", self.low_observation_space, env.envs[0].low_level_observation_space)
            assert self.low_observation_space == env.envs[0].low_level_observation_space, \
                "Error: the environment passed must have at least the same state space as the model was trained on."
            assert self.low_action_space == env.envs[0].low_level_action_space, \
                "Error: the environment passed must have at least the same action space as the model was trained on."
        
        assert self.high_observation_space == env.observation_space, \
            "Error: the environment passed must have at least the same observation space as the model was trained on."
        assert self.high_action_space == env.action_space, \
            "Error: the environment passed must have at least the same skill space as the model was trained on."

        if self._requires_vec_env:
            assert isinstance(env, VecEnv), \
                "Error: the environment passed is not a vectorized environment, however {} requires it".format(
                    self.__class__.__name__)
            assert not self.policy.recurrent or self.n_envs == env.num_envs, \
                "Error: the environment passed must have the same number of environments as the model was trained on." \
                "This is due to the Lstm policy not being capable of changing the number of environments."
            self.n_envs = env.num_envs
        else:
            # for models that dont want vectorized environment, check if they make sense and adapt them.
            # Otherwise tell the user about this issue
            if isinstance(env, VecEnv):
                if env.num_envs == 1:
                    env = _UnvecWrapper(env)
                    self._vectorize_action = True
                else:
                    raise ValueError("Error: the model requires a non vectorized environment or a single vectorized "
                                     "environment.")
            else:
                self._vectorize_action = False

            self.n_envs = 1

        self.env = env

    def _init_num_timesteps(self, reset_num_timesteps=True):
        """
        Initialize and resets num_timesteps (total timesteps since beginning of training)
        if needed. Mainly used logging and plotting (tensorboard).
        :param reset_num_timesteps: (bool) Set it to false when continuing training
            to not create new plotting curves in tensorboard.
        :return: (bool) Whether a new tensorboard log needs to be created
        """
        if reset_num_timesteps:
            self.num_timesteps = 0

        new_tb_log = self.num_timesteps == 0
        return new_tb_log

    @abstractmethod
    def setup_model(self):
        """
        Create all the functions and tensorflow graphs necessary to train the model
        """
        pass

    def set_random_seed(self, seed):
        """
        :param seed: (int) Seed for the pseudo-random generators. If None,
            do not change the seeds.
        """
        # Ignore if the seed is None
        if seed is None:
            return
        # Seed python, numpy and tf random generator
        set_global_seeds(seed)
        if self.env is not None:
            if isinstance(self.env, VecEnv):
                # Use a different seed for each env
                for idx in range(self.env.num_envs):
                    self.env.env_method("seed", seed + idx)
            else:
                self.env.seed(seed)
            # Seed the action space
            # useful when selecting random actions
            self.env.action_space.seed(seed)
        self.low_action_space.seed(seed)

    def _setup_learn(self):
        """
        Check the environment.
        """
        if self.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment with"
                             "set_env(self, env) method.")

    @abstractmethod
    def get_parameter_list(self):
        """
        Get tensorflow Variables of model's parameters
        This includes all variables necessary for continuing training (saving / loading).
        :return: (list) List of tensorflow Variables
        """
        pass

    def get_parameters(self, trim_prefix=False):
        """
        Get current model parameters as dictionary of variable name -> ndarray.
        :return: (OrderedDict) Dictionary of variable name -> ndarray of model's parameters.
        """
        parameters_low, parameters_high = self.get_parameter_list()
        parameters_low_values = self.sess.run(parameters_low)
        parameters_high_values = self.sess.run(parameters_high)

        if trim_prefix:
            return_dictionary_low = OrderedDict((param.name[4:], value) for param, value in zip(parameters_low, parameters_low_values))
            return_dictionary_high = OrderedDict((param.name[5:], value) for param, value in zip(parameters_high, parameters_high_values))
        else:
            return_dictionary_low = OrderedDict((param.name, value) for param, value in zip(parameters_low, parameters_low_values))
            return_dictionary_high = OrderedDict((param.name, value) for param, value in zip(parameters_high, parameters_high_values))
        return return_dictionary_low, return_dictionary_high

    def _setup_load_operations(self):
        """
        Create tensorflow operations for loading model parameters
        """
        # Assume tensorflow graphs are static -> check
        # that we only call this function once
        if self._param_load_ops_low is not None:
            raise RuntimeError("Low parameter load operations have already been created")
        if self._param_load_ops_high is not None:
            raise RuntimeError("High parameter load operations have already been created")

        # For each loadable parameter, create appropiate
        # placeholder and an assign op, and store them to
        # self.load_param_ops as dict of variable.name -> (placeholder, assign)
        low_loadable_parameters, high_loadable_parameters = self.get_parameter_list()
        
        # loadable_parameters = loadable_parameters[0] + loadable_parameters[1]
        
        # Use OrderedDict to store order for backwards compatibility with
        # list-based params
        self._param_load_ops_low = OrderedDict()
        with self.graph.as_default():
            for param in low_loadable_parameters:
                placeholder = tf.placeholder(dtype=param.dtype, shape=param.shape)
                # param.name is unique (tensorflow variables have unique names)
                self._param_load_ops_low[param.name] = (placeholder, param.assign(placeholder))
        
        self._param_load_ops_high = OrderedDict()
        with self.graph.as_default():
            for param in high_loadable_parameters:
                placeholder = tf.placeholder(dtype=param.dtype, shape=param.shape)
                # param.name is unique (tensorflow variables have unique names)
                self._param_load_ops_high[param.name] = (placeholder, param.assign(placeholder))

    @abstractmethod
    def skill_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        return NotImplemented

    @abstractmethod
    def predict_skill(self, observation, state=None, mask=None, deterministic=False):
        return NotImplemented

    @abstractmethod
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run",
              reset_num_timesteps=True, high_training_starts=0):
        """
        Return a trained model.
        :param total_timesteps: (int) The total number of samples to train on
        :param callback: (function (dict, dict)) -> boolean function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        :return: (BaseHRLModel) the trained model
        """
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation
        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        pass

    @abstractmethod
    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        """
        If ``actions`` is ``None``, then get the model's action probability distribution from a given observation.
        Depending on the action space the output is:
            - Discrete: probability for each possible action
            - Box: mean and standard deviation of the action output
        However if ``actions`` is not ``None``, this function will return the probability that the given actions are
        taken with the given parameters (observation, state, ...) on this model. For discrete action spaces, it
        returns the probability mass; for continuous action spaces, the probability density. This is since the
        probability mass will always be zero in continuous spaces, see http://blog.christianperone.com/2019/01/
        for a good explanation
        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param actions: (np.ndarray) (OPTIONAL) For calculating the likelihood that the given actions are chosen by
            the model for each of the given parameters. Must have the same number of actions and observations.
            (set to None to return the complete action probability distribution)
        :param logp: (bool) (OPTIONAL) When specified with actions, returns probability in log-space.
            This has no effect if actions is None.
        :return: (np.ndarray) the model's (log) action probability
        """
        pass

    def load_parameters(self, low_load_path_or_dict, high_load_path_or_dict, exact_match=True):
        """
        Load model parameters from a file or a dictionary
        Dictionary keys should be tensorflow variable names, which can be obtained
        with ``get_parameters`` function. If ``exact_match`` is True, dictionary
        should contain keys for all model's parameters, otherwise RunTimeError
        is raised. If False, only variables included in the dictionary will be updated.
        This does not load agent's hyper-parameters.
        .. warning::
            This function does not update trainer/optimizer variables (e.g. momentum).
            As such training after using this function may lead to less-than-optimal results.
        :param load_path_or_dict: (str or file-like or dict) Save parameter location
            or dict of parameters as variable.name -> ndarrays to be loaded.
        :param exact_match: (bool) If True, expects load dictionary to contain keys for
            all variables in the model. If False, loads parameters only for variables
            mentioned in the dictionary. Defaults to True.
        """
        # Make sure we have assign ops
        if self._param_load_ops_low is None or self._param_load_ops_high is None:
            self._setup_load_operations()

        low_params = None
        if isinstance(low_load_path_or_dict, dict):
            # Assume `load_path_or_dict` is dict of variable.name -> ndarrays we want to load
            low_params = low_load_path_or_dict
        elif isinstance(low_load_path_or_dict, list):
            warnings.warn("Loading model parameters from a list. This has been replaced " +
                          "with parameter dictionaries with variable names and parameters. " +
                          "If you are loading from a file, consider re-saving the file.",
                          DeprecationWarning)
            # Assume `load_path_or_dict` is list of ndarrays.
            # Create param dictionary assuming the parameters are in same order
            # as `get_parameter_list` returns them.
            low_params = dict()
            for i, param_name in enumerate(self._param_load_ops_low.keys()):
                low_params[param_name] = low_load_path_or_dict[i]
        else:
            # Assume a filepath or file-like.
            # Use existing deserializer to load the parameters.
            # We only need the parameters part of the file, so
            # only load that part.
            _, low_params = BaseHRLModel._load_from_file(low_load_path_or_dict, load_data=False)

        high_params = None
        if isinstance(high_load_path_or_dict, dict):
            # Assume `load_path_or_dict` is dict of variable.name -> ndarrays we want to load
            high_params = high_load_path_or_dict
        elif isinstance(high_load_path_or_dict, list):
            warnings.warn("Loading model parameters from a list. This has been replaced " +
                          "with parameter dictionaries with variable names and parameters. " +
                          "If you are loading from a file, consider re-saving the file.",
                          DeprecationWarning)
            # Assume `load_path_or_dict` is list of ndarrays.
            # Create param dictionary assuming the parameters are in same order
            # as `get_parameter_list` returns them.
            high_params = dict()
            for i, param_name in enumerate(self._param_load_ops_high.keys()):
                high_params[param_name] = high_load_path_or_dict[i]
        else:
            # Assume a filepath or file-like.
            # Use existing deserializer to load the parameters.
            # We only need the parameters part of the file, so
            # only load that part.
            _, high_params = BaseHRLModel._load_from_file(low_load_path_or_dict, load_data=False)

        feed_dict = {}
        param_update_ops = []
        # Keep track of not-updated variables
        not_updated_variables = set(self._param_load_ops_low.keys())
        for param_name, param_value in low_params.items():
            if not param_name.startswith('low/'):
                param_name = 'low/' + param_name
            placeholder, assign_op = self._param_load_ops_low[param_name]
            feed_dict[placeholder] = param_value
            # Create list of tf.assign operations for sess.run
            param_update_ops.append(assign_op)
            # Keep track which variables are updated
            not_updated_variables.remove(param_name)

        # Check that we updated all parameters if exact_match=True
        if exact_match and len(not_updated_variables) > 0:
            raise RuntimeError("Low Load dictionary did not contain all variables. " +
                               "Missing variables: {}".format(", ".join(not_updated_variables)))

        self.sess.run(param_update_ops, feed_dict=feed_dict)

        feed_dict = {}
        param_update_ops = []
        # Keep track of not-updated variables
        not_updated_variables = set(self._param_load_ops_high.keys())
        for param_name, param_value in high_params.items():
            if not param_name.startswith('high/'):
                param_name = 'high/' + param_name
            placeholder, assign_op = self._param_load_ops_high[param_name]
            feed_dict[placeholder] = param_value
            # Create list of tf.assign operations for sess.run
            param_update_ops.append(assign_op)
            # Keep track which variables are updated
            not_updated_variables.remove(param_name)

        # Check that we updated all parameters if exact_match=True
        if exact_match and len(not_updated_variables) > 0:
            raise RuntimeError("Low Load dictionary did not contain all variables. " +
                               "Missing variables: {}".format(", ".join(not_updated_variables)))

        self.sess.run(param_update_ops, feed_dict=feed_dict)

    @abstractmethod
    def save(self, low_save_path, high_save_path, cloudpickle=False):
        """
        Save the current parameters to file
        :param save_path: (str or file-like) The save location
        :param cloudpickle: (bool) Use older cloudpickle format instead of zip-archives.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, low_load_path, high_load_path, env=None, custom_objects=None, **kwargs):
        """
        Load the model from file
        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        raise NotImplementedError()

    @staticmethod
    def _save_to_file_cloudpickle(save_path, data=None, params=None):
        """Legacy code for saving models with cloudpickle
        :param save_path: (str or file-like) Where to store the model
        :param data: (OrderedDict) Class parameters being stored
        :param params: (OrderedDict) Model parameters being stored
        """
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".pkl"

            with open(save_path, "wb") as file_:
                cloudpickle.dump((data, params), file_)
        else:
            # Here save_path is a file-like object, not a path
            cloudpickle.dump((data, params), save_path)

    @staticmethod
    def _save_to_file_zip(save_path, data=None, params=None):
        """Save model to a .zip archive
        :param save_path: (str or file-like) Where to store the model
        :param data: (OrderedDict) Class parameters being stored
        :param params: (OrderedDict) Model parameters being stored
        """
        # data/params can be None, so do not
        # try to serialize them blindly
        if data is not None:
            serialized_data = data_to_json(data)
        if params is not None:
            serialized_params = params_to_bytes(params)
            # We also have to store list of the parameters
            # to store the ordering for OrderedDict.
            # We can trust these to be strings as they
            # are taken from the Tensorflow graph.
            serialized_param_list = json.dumps(
                list(params.keys()),
                indent=4
            )

        # Check postfix if save_path is a string
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".zip"

        # Create a zip-archive and write our objects
        # there. This works when save_path
        # is either str or a file-like
        with zipfile.ZipFile(save_path, "w") as file_:
            # Do not try to save "None" elements
            if data is not None:
                file_.writestr("data", serialized_data)
            if params is not None:
                file_.writestr("parameters", serialized_params)
                file_.writestr("parameter_list", serialized_param_list)

    @staticmethod
    def _save_to_file(save_path, data=None, params=None, cloudpickle=False):
        """Save model to a zip archive or cloudpickle file.
        :param save_path: (str or file-like) Where to store the model
        :param data: (OrderedDict) Class parameters being stored
        :param params: (OrderedDict) Model parameters being stored
        :param cloudpickle: (bool) Use old cloudpickle format
            (stable-baselines<=2.7.0) instead of a zip archive.
        """
        if cloudpickle:
            BaseHRLModel._save_to_file_cloudpickle(save_path, data, params)
        else:
            BaseHRLModel._save_to_file_zip(save_path, data, params)

    @staticmethod
    def _load_from_file_cloudpickle(load_path):
        """Legacy code for loading older models stored with cloudpickle
        :param load_path: (str or file-like) where from to load the file
        :return: (dict, OrderedDict) Class parameters and model parameters
        """
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".pkl"):
                    load_path += ".pkl"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

            with open(load_path, "rb") as file_:
                data, params = cloudpickle.load(file_)
        else:
            # Here load_path is a file-like object, not a path
            data, params = cloudpickle.load(load_path)

        return data, params

    @staticmethod
    def _load_from_file(load_path, load_data=True, custom_objects=None):
        """Load model data from a .zip archive
        :param load_path: (str or file-like) Where to load model from
        :param load_data: (bool) Whether we should load and return data
            (class parameters). Mainly used by `load_parameters` to
            only load model parameters (weights).
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :return: (dict, OrderedDict) Class parameters and model parameters
        """
        # Check if file exists if load_path is
        # a string
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".zip"):
                    load_path += ".zip"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

        # Open the zip archive and load data.

        try:
            with zipfile.ZipFile(load_path, "r") as file_:
                namelist = file_.namelist()
                # If data or parameters is not in the
                # zip archive, assume they were stored
                # as None (_save_to_file allows this).
                data = None
                params = None
                if "data" in namelist and load_data:
                    # Load class parameters and convert to string
                    # (Required for json library in Python 3.5)
                    json_data = file_.read("data").decode()
                    data = json_to_data(json_data, custom_objects=custom_objects)
                if "parameters" in namelist:
                    # Load parameter list and and parameters
                    parameter_list_json = file_.read("parameter_list").decode()
                    parameter_list = json.loads(parameter_list_json)
                    serialized_params = file_.read("parameters")
                    params = bytes_to_params(
                        serialized_params, parameter_list
                    )
        except zipfile.BadZipFile:
            # load_path wasn't a zip file. Possibly a cloudpickle
            # file. Show a warning and fall back to loading cloudpickle.
            warnings.warn("It appears you are loading from a file with old format. " +
                          "Older cloudpickle format has been replaced with zip-archived " +
                          "models. Consider saving the model with new format.",
                          DeprecationWarning)
            # Attempt loading with the cloudpickle format.
            # If load_path is file-like, seek back to beginning of file
            if not isinstance(load_path, str):
                load_path.seek(0)
            data, params = BaseHRLModel._load_from_file_cloudpickle(load_path)

        return data, params

    @staticmethod
    def _softmax(x_input):
        """
        An implementation of softmax.
        :param x_input: (numpy float) input vector
        :return: (numpy float) output vector
        """
        x_exp = np.exp(x_input.T - np.max(x_input.T, axis=0))
        return (x_exp / x_exp.sum(axis=0)).T

    @staticmethod
    def _is_vectorized_observation(observation, observation_space):
        """
        For every observation type, detects and validates the shape,
        then returns whether or not the observation is vectorized.
        :param observation: (np.ndarray) the input observation to validate
        :param observation_space: (gym.spaces) the observation space
        :return: (bool) whether the given observation is vectorized or not
        """
        if isinstance(observation_space, gym.spaces.Box):
            if observation.shape == observation_space.shape:
                return False
            elif observation.shape[1:] == observation_space.shape:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for ".format(observation.shape) +
                                 "Box environment, please use {} ".format(observation_space.shape) +
                                 "or (n_env, {}) for the observation shape."
                                 .format(", ".join(map(str, observation_space.shape))))
        elif isinstance(observation_space, gym.spaces.Discrete):
            if observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
                return False
            elif len(observation.shape) == 1:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for ".format(observation.shape) +
                                 "Discrete environment, please use (1,) or (n_env, 1) for the observation shape.")
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            if observation.shape == (len(observation_space.nvec),):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for MultiDiscrete ".format(observation.shape) +
                                 "environment, please use ({},) or ".format(len(observation_space.nvec)) +
                                 "(n_env, {}) for the observation shape.".format(len(observation_space.nvec)))
        elif isinstance(observation_space, gym.spaces.MultiBinary):
            if observation.shape == (observation_space.n,):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == observation_space.n:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for MultiBinary ".format(observation.shape) +
                                 "environment, please use ({},) or ".format(observation_space.n) +
                                 "(n_env, {}) for the observation shape.".format(observation_space.n))
        else:
            raise ValueError("Error: Cannot determine if the observation is vectorized with the space type {}."
                             .format(observation_space))

class _UnvecWrapper(VecEnvWrapper):
    def __init__(self, venv):
        """
        Unvectorize a vectorized environment, for vectorized environment that only have one environment
        :param venv: (VecEnv) the vectorized environment to wrap
        """
        super().__init__(venv)
        assert venv.num_envs == 1, "Error: cannot unwrap a environment wrapper that has more than one environment."

    def seed(self, seed=None):
        return self.venv.env_method('seed', seed)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.venv, attr)

    def __set_attr__(self, attr, value):
        if attr in self.__dict__:
            setattr(self, attr, value)
        else:
            setattr(self.venv, attr, value)

    def compute_reward(self, achieved_goal, desired_goal, _info):
        return float(self.venv.env_method('compute_reward', achieved_goal, desired_goal, _info)[0])

    @staticmethod
    def unvec_obs(obs):
        """
        :param obs: (Union[np.ndarray, dict])
        :return: (Union[np.ndarray, dict])
        """
        if isinstance(obs, dict):
            obs_ = OrderedDict()
            for key in obs.keys():
                obs_[key] = obs[key][0]
            del obs
            return obs_
        elif isinstance(obs, tuple):
            zeroth_items = [item[0] for item in obs]
            del obs
            return tuple(zeroth_items)
        else:
            return obs[0]

    def reset(self):
        return self.unvec_obs(self.venv.reset())

    def step_async(self, actions):
        self.venv.step_async([actions])

    def step_wait(self):
        obs, rewards, dones, information = self.venv.step_wait()
        return self.unvec_obs(obs), float(rewards[0]), dones[0], information[0]

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

class ActorCriticHRLModel(BaseHRLModel):

    def __init__(self, low_policy, high_policy, env, _init_setup_model, verbose=0, low_policy_base=ActorCriticPolicy,
                 high_policy_base=ActorCriticPolicy, requires_vec_env=False, low_policy_kwargs=None,
                 high_policy_kwargs=None, seed=None, n_cpu_tf_sess=None):
        super(ActorCriticHRLModel, self).__init__(low_policy, high_policy, env, verbose=verbose, requires_vec_env=requires_vec_env,
                                                 low_policy_base=low_policy_base, high_policy_base=high_policy_base,
                                                 low_policy_kwargs=low_policy_kwargs, high_policy_kwargs=high_policy_kwargs,
                                                 seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.sess = None
        self.low_initial_state = None
        self.high_initial_state = None
        self.low_step = None
        self.high_step = None
        self.low_proba_step = None
        self.high_proba_step = None
        self.low_params = None
        self.high_params = None

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def learn(self, total_timesteps, callback=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True):
        pass

    def predict(self, observation, state=None, mask=None, deterministic=False):
        if state is None:
            state = self.low_initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.low_observation_space)

        observation = observation.reshape((-1,) + self.low_observation_space.shape)
        actions, _, states, _ = self.low_step(observation, state, mask, deterministic=deterministic)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.low_action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.low_action_space.low, self.low_action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, states

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if state is None:
            state = self.low_initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.low_observation_space)

        observation = observation.reshape((-1,) + self.low_observation_space.shape)
        actions_proba = self.low_proba_step(observation, state, mask)

        if len(actions_proba) == 0:  # empty list means not implemented
            warnings.warn("Warning: action probability is not implemented for {} action space. Returning None."
                          .format(type(self.low_action_space).__name__))
            return None

        if actions is not None:  # comparing the action distribution, to given actions
            prob = None
            logprob = None
            actions = np.array([actions])
            if isinstance(self.low_action_space, gym.spaces.Discrete):
                actions = actions.reshape((-1,))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                prob = actions_proba[np.arange(actions.shape[0]), actions]

            elif isinstance(self.low_action_space, gym.spaces.MultiDiscrete):
                actions = actions.reshape((-1, len(self.low_action_space.nvec)))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                # Discrete action probability, over multiple categories
                actions = np.swapaxes(actions, 0, 1)  # swap axis for easier categorical split
                prob = np.prod([proba[np.arange(act.shape[0]), act]
                                         for proba, act in zip(actions_proba, actions)], axis=0)

            elif isinstance(self.low_action_space, gym.spaces.MultiBinary):
                actions = actions.reshape((-1, self.low_action_space.n))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                # Bernoulli action probability, for every action
                prob = np.prod(actions_proba * actions + (1 - actions_proba) * (1 - actions), axis=1)

            elif isinstance(self.low_action_space, gym.spaces.Box):
                actions = actions.reshape((-1, ) + self.low_action_space.shape)
                mean, logstd = actions_proba
                std = np.exp(logstd)

                n_elts = np.prod(mean.shape[1:])  # first dimension is batch size
                log_normalizer = n_elts/2 * np.log(2 * np.pi) + 1/2 * np.sum(logstd, axis=1)

                # Diagonal Gaussian action probability, for every action
                logprob = -np.sum(np.square(actions - mean) / (2 * std), axis=1) - log_normalizer

            else:
                warnings.warn("Warning: action_probability not implemented for {} actions space. Returning None."
                              .format(type(self.low_action_space).__name__))
                return None

            # Return in space (log or normal) requested by user, converting if necessary
            if logp:
                if logprob is None:
                    logprob = np.log(prob)
                ret = logprob
            else:
                if prob is None:
                    prob = np.exp(logprob)
                ret = prob

            # normalize action proba shape for the different gym spaces
            ret = ret.reshape((-1, 1))
        else:
            ret = actions_proba

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            ret = ret[0]

        return ret

    def predict_skill(self, observation, state=None, mask=None, deterministic=False):
        if state is None:
            state = self.high_initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.high_observation_space)

        observation = observation.reshape((-1,) + self.high_observation_space.shape)
        actions, _, states, _ = self.high_step(observation, state, mask, deterministic=deterministic)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.high_action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.high_action_space.low, self.high_action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, states

    def skill_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if state is None:
            state = self.high_initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.high_observation_space)

        observation = observation.reshape((-1,) + self.high_observation_space.shape)
        actions_proba = self.high_proba_step(observation, state, mask)

        if len(actions_proba) == 0:  # empty list means not implemented
            warnings.warn("Warning: action probability is not implemented for {} action space. Returning None."
                          .format(type(self.low_action_space).__name__))
            return None

        if actions is not None:  # comparing the action distribution, to given actions
            prob = None
            logprob = None
            actions = np.array([actions])
            if isinstance(self.high_action_space, gym.spaces.Discrete):
                actions = actions.reshape((-1,))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                prob = actions_proba[np.arange(actions.shape[0]), actions]

            elif isinstance(self.high_action_space, gym.spaces.MultiDiscrete):
                actions = actions.reshape((-1, len(self.high_action_space.nvec)))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                # Discrete action probability, over multiple categories
                actions = np.swapaxes(actions, 0, 1)  # swap axis for easier categorical split
                prob = np.prod([proba[np.arange(act.shape[0]), act]
                                         for proba, act in zip(actions_proba, actions)], axis=0)

            elif isinstance(self.high_action_space, gym.spaces.MultiBinary):
                actions = actions.reshape((-1, self.high_action_space.n))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                # Bernoulli action probability, for every action
                prob = np.prod(actions_proba * actions + (1 - actions_proba) * (1 - actions), axis=1)

            elif isinstance(self.high_action_space, gym.spaces.Box):
                actions = actions.reshape((-1, ) + self.high_action_space.shape)
                mean, logstd = actions_proba
                std = np.exp(logstd)

                n_elts = np.prod(mean.shape[1:])  # first dimension is batch size
                log_normalizer = n_elts/2 * np.log(2 * np.pi) + 1/2 * np.sum(logstd, axis=1)

                # Diagonal Gaussian action probability, for every action
                logprob = -np.sum(np.square(actions - mean) / (2 * std), axis=1) - log_normalizer

            else:
                warnings.warn("Warning: action_probability not implemented for {} actions space. Returning None."
                              .format(type(self.low_action_space).__name__))
                return None

            # Return in space (log or normal) requested by user, converting if necessary
            if logp:
                if logprob is None:
                    logprob = np.log(prob)
                ret = logprob
            else:
                if prob is None:
                    prob = np.exp(logprob)
                ret = prob

            # normalize action proba shape for the different gym spaces
            ret = ret.reshape((-1, 1))
        else:
            ret = actions_proba

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            ret = ret[0]

        return ret

    def get_parameter_list(self):
        return self.low_params, self.high_params

    @abstractmethod
    def save(self, low_save_path, high_save_path, cloudpickle=False):
        pass

    @classmethod
    def load(cls, load_path_low, load_path_high, env=None, custom_objects=None, **kwargs):
        """
        Load the model from file
        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        low_data, low_params = cls._load_from_file(load_path_low, custom_objects=custom_objects)
        high_data, high_params = cls._load_from_file(load_path_high, custom_objects=custom_objects)


        if 'low_policy_kwargs' in kwargs and kwargs['low_policy_kwargs'] != low_data['policy_kwargs']:
            raise ValueError("The specified low level policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(low_data['policy_kwargs'],
                                                                              kwargs['low_policy_kwargs']))

        if 'high_policy_kwargs' in kwargs and kwargs['high_policy_kwargs'] != high_data['policy_kwargs']:
            raise ValueError("The specified high level policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(high_data['policy_kwargs'],
                                                                              kwargs['high_policy_kwargs']))

        # This line will need to be updated
        model = cls(low_policy=low_data["policy"], high_policy=high_data["policy"], env=None, _init_setup_model=False)

        #  First add low level data
        for key in low_data.keys():
            if 'low_' + key in model.__dict__:
                model.__dict__['low_' + key] = low_data[key]
            elif key in model.__dict__:
                model.__dict__[key] = low_data[key]

        # Allow high level data to override
        for key in high_data.keys():
            if 'high_' + key in model.__dict__:
                model.__dict__['high_' + key] = high_data[key]
            elif key in model.__dict__:
                model.__dict__[key] = high_data[key]
        
        model.__dict__.update(kwargs)

        model.set_env(env)
        model.setup_model()
        model.load_parameters(low_params, high_params)

        return model

class OffPolicyHRLModel(BaseHRLModel):

    def __init__(self, low_policy, high_policy, env, low_replay_buffer=None, high_replay_buffer=None, 
                 _init_setup_model=False, verbose=0, *,
                 requires_vec_env=False, low_policy_base, high_policy_base, low_policy_kwargs=None,
                 high_policy_kwargs=None, seed=None, n_cpu_tf_sess=None):

        super(OffPolicyHRLModel, self).__init__(low_policy, high_policy, env, verbose=verbose, requires_vec_env=requires_vec_env,
                                               low_policy_base=low_policy_base, low_policy_kwargs=low_policy_kwargs,
                                               high_policy_base=high_policy_base, high_policy_kwargs=high_policy_kwargs,
                                               seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.low_replay_buffer = low_replay_buffer
        self.high_replay_buffer = high_replay_buffer

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def learn(self, total_timesteps, callback=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True, replay_wrapper=None):
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        pass

    @abstractmethod
    def predict_skill(self, observation, state=None, mask=None, deterministic=False):
        pass

    @abstractmethod
    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        pass

    @abstractmethod
    def skill_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        pass

    @abstractmethod
    def save(self, low_save_path, high_save_path, cloudpickle=False):
        pass

    @classmethod
    def load(cls, load_path_low, load_path_high, env=None, custom_objects=None, **kwargs):
        """
        Load the model from file
        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        low_data, low_params = cls._load_from_file(load_path_low, custom_objects=custom_objects)
        high_data, high_params = cls._load_from_file(load_path_high, custom_objects=custom_objects)


        if 'low_policy_kwargs' in kwargs and kwargs['low_policy_kwargs'] != low_data['policy_kwargs']:
            raise ValueError("The specified low level policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(low_data['policy_kwargs'],
                                                                              kwargs['low_policy_kwargs']))

        if 'high_policy_kwargs' in kwargs and kwargs['high_policy_kwargs'] != high_data['policy_kwargs']:
            raise ValueError("The specified high level policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(high_data['policy_kwargs'],
                                                                              kwargs['high_policy_kwargs']))

        # This line will need to be updated
        model = cls(low_policy=low_data["policy"], high_policy=high_data["policy"], env=None, _init_setup_model=False)

        #  First add low level data
        for key in low_data.keys():
            if 'low_' + key in model.__dict__:
                model.__dict__['low_' + key] = low_data[key]
            elif key in model.__dict__:
                model.__dict__[key] = low_data[key]

        # Allow high level data to override
        for key in high_data.keys():
            if 'high_' + key in model.__dict__:
                model.__dict__['high_' + key] = high_data[key]
            elif key in model.__dict__:
                model.__dict__[key] = high_data[key]
        
        model.__dict__.update(kwargs)

        model.set_env(env)
        model.setup_model()
        model.load_parameters(low_params, high_params)

        return model