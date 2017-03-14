import gym
import gym.wrappers
import numpy as np

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
