# Atari-specific options for environments
import gym
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense
import numpy as np
import cv2

# how many steps to keep in input shape
HISTORY_STEPS = 2

# shape of prescaled image. If None, no prescaling will be performed
IMAGE_RESCALE = (84, 84)

INPUT_SHAPE = IMAGE_RESCALE + (HISTORY_STEPS*3,)


def RescaleWrapper(shape=IMAGE_RESCALE):
    class _RescaleWrapper(gym.Wrapper):
        """
        Track history of observations for given amount of steps
        Initial steps are zero-filled
        """
        def __init__(self, env):
            super(_RescaleWrapper, self).__init__(env)
            self.shape = shape

        def _step(self, action):
            obs, reward, done, info = self.env.step(action)
            return preprocess_state(obs, self.shape), reward, done, info

        def _reset(self):
            return preprocess_state(self.env.reset(), shape=self.shape)

    return _RescaleWrapper


def net_input():
    """
    Create input part of the network with optional prescaling.
    :return: input_tensor, output_tensor
    """
    in_t = Input(shape=INPUT_SHAPE, name='input')
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


def preprocess_state(state, shape=IMAGE_RESCALE):
    """
    Convert input from atari game + history buffer to shape expected by net_input function.
    :param state: input state
    :return:
    """
    state = np.transpose(state, (1, 2, 3, 0))
    state = np.reshape(state, (state.shape[0], state.shape[1], state.shape[2]*state.shape[3]))

    state = state.astype(np.float32)
    res = cv2.resize(state, shape)
    res /= 255
    return res
