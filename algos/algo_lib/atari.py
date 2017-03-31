# Atari-specific options for environments
import gym
import gym.spaces
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense
import numpy as np
import cv2

from . import common


class AtariEnvFactory:
    def __init__(self, config):
        self.config = config
        self.common_factory = common.EnvFactory(config)

    def __call__(self):
        env = self.common_factory()
        return RescaleWrapper(self.config)(env)


class RescaleWrapper:
    def __init__(self, config):
        self.config = config

    class _RescaleWrapper(gym.Wrapper):
        """
        Track history of observations for given amount of steps
        Initial steps are zero-filled
        """
        def __init__(self, config, env):
            super(RescaleWrapper._RescaleWrapper, self).__init__(env)
            self.shape = config.image_shape
            self.observation_space = self._make_observation_space(env.observation_space, self.shape)

        def _step(self, action):
            obs, reward, done, info = self.env.step(action)
            return self._preprocess(obs), reward, done, info

        def _reset(self):
            return self._preprocess(self.env.reset())

        @staticmethod
        def _make_observation_space(orig_space, target_shape):
            assert isinstance(orig_space, gym.spaces.Box)
            shape = target_shape + (orig_space.shape[0] * orig_space.shape[-1], )
            low = np.ones(shape) * orig_space.low.min()
            high = np.ones(shape) * orig_space.high.max()
            return gym.spaces.Box(low, high)

        def _preprocess(self, state):
            """
            Convert input from atari game + history buffer to shape expected by net_input function.
            :param state: input state
            :return:
            """
            state = np.transpose(state, (1, 2, 3, 0))
            state = np.reshape(state, (state.shape[0], state.shape[1], state.shape[2] * state.shape[3]))

            state = state.astype(np.float32)
            res = cv2.resize(state, self.shape)
            res /= 255
            return res

    def __call__(self, env):
        return self._RescaleWrapper(self.config, env)


def net_input(env):
    """
    Create input part of the network with optional prescaling.
    :return: input_tensor, output_tensor
    """
    in_t = Input(shape=env.observation_space.shape, name='input')
    out_t = Conv2D(32, 5, 5, activation='relu', border_mode='same')(in_t)
    out_t = MaxPooling2D((2, 2))(out_t)
    out_t = Conv2D(32, 5, 5, activation='relu', border_mode='same')(out_t)
    out_t = MaxPooling2D((2, 2))(out_t)
    out_t = Conv2D(64, 4, 4, activation='relu', border_mode='same')(out_t)
    out_t = MaxPooling2D((2, 2))(out_t)
    out_t = Conv2D(64, 3, 3, activation='relu', border_mode='same')(out_t)
    out_t = Flatten(name='flat')(out_t)
    out_t = Dense(512, name='l1', activation='relu')(out_t)

    return in_t, out_t


