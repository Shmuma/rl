import os
import configparser
import gym
import gym.spaces
import gym.wrappers
import numpy as np
import logging as log

import keras.backend as K
import tensorflow as tf
import collections


def HistoryWrapper(steps):
    class _HistoryWrapper(gym.Wrapper):
        """
        Track history of observations for given amount of steps
        Initial steps are zero-filled
        """
        def __init__(self, env):
            super(_HistoryWrapper, self).__init__(env)
            self.steps = steps
            self.history = self._make_history()
            self.observation_space = self._make_observation_space(steps, env.observation_space)

        @staticmethod
        def _make_observation_space(steps, orig_obs):
            assert isinstance(orig_obs, gym.spaces.Box)
            low = np.repeat(np.expand_dims(orig_obs.low, 0), steps, axis=0)
            high = np.repeat(np.expand_dims(orig_obs.high, 0), steps, axis=0)
            return gym.spaces.Box(low, high)

        def _make_history(self, last_item = None):
            size = self.steps if last_item is None else self.steps-1
            res = collections.deque([np.zeros(shape=self.env.observation_space.shape)] * size)
            if last_item is not None:
                res.append(last_item)
            return res

        def _step(self, action):
            obs, reward, done, info = self.env.step(action)
            self.history.popleft()
            self.history.append(obs)
            return self.history, reward, done, info

        def _reset(self):
            self.history = self._make_history(last_item=self.env.reset())
            return self.history

    return _HistoryWrapper


def make_env(env_name, monitor_dir=None, wrappers=()):
    """
    Make gym environment with optional monitor
    :param env_name: name of the environment to create
    :param monitor_dir: optional directory to save monitor results
    :param wrappers: list of optional Wrapper object instances
    :return: environment object
    """
    env = gym.make(env_name)
    for wrapper in wrappers:
        env = wrapper(env)
    if monitor_dir:
        env = gym.wrappers.Monitor(env, monitor_dir)
    return env


def summarize_gradients(model):
    """
    Add summaries of gradients
    :param model: compiled keras model
    """
    gradients = model.optimizer.get_gradients(model.total_loss, model._collected_trainable_weights)
    for var, grad in zip(model._collected_trainable_weights, gradients):
        n = var.name.split(':', maxsplit=1)[0]
        tf.summary.scalar("gradrms_" + n, K.sqrt(K.mean(K.square(grad))))


def summary_value(name, value, writer, step_no):
    """
    Add given actual value to summary writer
    :param name: name of value to add
    :param value: scalar value
    :param writer: SummaryWriter instance
    :param step_no: global step index
    """
    summ = tf.Summary()
    summ_value = summ.value.add()
    summ_value.simple_value = value
    summ_value.tag = name
    writer.add_summary(summ, global_step=step_no)


class ParamsTweaker:
    logger = log.getLogger("ParamsTweaker")

    def __init__(self, file_name="tweak_params.txt"):
        self.file_name = file_name
        self.params = {}

    def add(self, name, var):
        self.params[name] = var

    def check(self):
        if not os.path.exists(self.file_name):
            return

        self.logger.info("Tweak file detected: %s", self.file_name)
        with open(self.file_name, "rt", encoding='utf-8') as fd:
            for idx, l in enumerate(fd):
                name, val = list(map(str.strip, l.split('=', maxsplit=2)))
                var = self.params.get(name)
                if not var:
                    self.logger.info("Unknown param '%s' found in file at line %d, ignored", name, idx+1)
                    continue
                self.logger.info("Param %s <-- %s", name, val)
                K.set_value(var, float(val))
        os.remove(self.file_name)
    pass


class Configuration:
    def __init__(self, file_name):
        self.file_name = file_name
        self.config = configparser.ConfigParser()
        if not self.config.read(file_name):
            raise FileNotFoundError(file_name)

    @property
    def env_name(self):
        return self.config.get('game', 'env')

    @property
    def history(self):
        return self.config.getint('game', 'history', fallback=1)

    @property
    def image_shape(self):
        x = self.config.getint('game', 'image_x')
        y = self.config.getint('game', 'image_y')
        if x is not None and y is not None:
            return (x, y)
        return None

    @property
    def max_steps(self):
        return self.config.getint('game', 'max_steps')

    @property
    def a3c_beta(self):
        return self.config.getfloat('a3c', 'entropy_beta')

    @property
    def a3c_steps(self):
        return self.config.getint('a3c', 'reward_steps')

    @property
    def a3c_gamma(self):
        return self.config.getfloat('a3c', 'gamma')

    @property
    def batch_size(self):
        return self.config.getint('training', 'batch_size')

    @property
    def learning_rate(self):
        return self.config.getfloat('training', 'learning_rate')

    @property
    def gradient_clip_norm(self):
        return self.config.getfloat('training', 'grad_clip_norm')

    @property
    def swarms_count(self):
        return self.config.getint('swarm', 'swarms')

    @property
    def swarm_size(self):
        return self.config.getint('swarm', 'swarm_size')


class EnvFactory:
    def __init__(self, config):
        self.config = config

    def __call__(self):
        env = gym.make(self.config.env_name)
        history = self.config.history
        if history > 1:
            env = HistoryWrapper(history)(env)
        return env
