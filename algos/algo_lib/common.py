import gym
import gym.wrappers
import numpy as np

import keras.backend as K
import tensorflow as tf


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

        def _make_history(self):
            return [np.zeros(shape=self.env.observation_space.shape) for _ in range(steps)]

        def _step(self, action):
            obs, reward, done, info = self.env.step(action)
            self.history.pop(0)
            self.history.append(obs)
            return np.array(self.history), reward, done, info

        def _reset(self):
            self.history = self._make_history()
            self.history.pop(0)
            self.history.append(self.env.reset())
            return np.array(self.history)

    return _HistoryWrapper


def make_env(env_name, monitor_dir=None, history_steps=2):
    """
    Make gym environment with optional monitor and given amount of history steps
    :param env_name: name of the environment to create
    :param monitor_dir: optional directory to save monitor results
    :param history_steps: count of steps to preserve as history in state
    :return: environment object
    """
    env = HistoryWrapper(history_steps)(gym.make(env_name))
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
