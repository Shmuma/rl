#!/usr/bin/env python
# Quick-n-dirty implementation of Advantage Actor-Critic method from https://arxiv.org/abs/1602.01783
import logging
import numpy as np
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import gym

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda, merge
from keras.optimizers import RMSprop
from keras.objectives import mean_squared_error
from keras import backend as K
import tensorflow as tf

HISTORY_STEPS = 4
SIMPLE_L1_SIZE = 20
SIMPLE_L2_SIZE = 20


def HistoryWrapper(steps):
    class HistoryWrapper(gym.Wrapper):
        """
        Track history of observations for given amount of steps
        Initial steps are zero-filled
        """
        def __init__(self, env):
            super(HistoryWrapper, self).__init__(env)
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

    return HistoryWrapper


def make_env(env_name):
    return HistoryWrapper(HISTORY_STEPS)(gym.make(env_name))


def make_model(state_shape, n_actions, train_mode=True):
    in_t = Input(shape=(HISTORY_STEPS,) + state_shape, name='input')
    fl_t = Flatten(name='flat')(in_t)
    l1_t = Dense(SIMPLE_L1_SIZE, activation='relu', name='l1')(fl_t)
    l2_t = Dense(SIMPLE_L2_SIZE, activation='relu', name='l2')(l1_t)
    policy_t = Dense(n_actions, activation='softmax', name='policy')(l2_t)
    value_t = Dense(1, name='value')(l2_t)

    # we're not going to train -- just return policy and value from our state
    run_model = Model(input=in_t, output=[policy_t, value_t])
    if not train_mode:
        return run_model

    # we're training, life is much more interesting...
    reward_t = Input(shape=(1, ), name='sum_reward')
    action_t = Input(shape=(1, ), name='action', dtype='int32')

    # value loss
    value_loss_t = merge([value_t, reward_t], mode=lambda vals: mean_squared_error(vals[0], vals[1]),
                         output_shape=(1, ), name='value_loss')
    BETA = 0.01
    entropy_loss_t = Lambda(lambda p: BETA * K.sum(p * K.log(p)), output_shape=(1, ), name='entropy_loss')(policy_t)

    def policy_loss_func(args):
        policy_t, value_t, reward_t, action_t = args
        oh = K.one_hot(action_t, nb_classes=n_actions)
        p = K.log(policy_t) * oh
        return p * (reward_t - value_t)

    policy_loss_t = merge([policy_t, value_t, reward_t, action_t], mode=policy_loss_func,
                          output_shape=(1, ), name='policy_loss')

    train_model = Model(input=[in_t, reward_t, action_t], output=[value_loss_t, policy_loss_t, entropy_loss_t])
    return run_model, train_model


def zero_loss_func(true_data, pred_data):
    return K.zeros(shape=(None,))


def identity_loss_func(_, loss_data):
    return loss_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default="MountainCar-v0", help="Environment name to use")
    args = parser.parse_args()

    env = make_env(args.env)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    run_m, train_m = make_model(state_shape, n_actions, train_mode=True)
    train_m.summary()

    # loss is kinda tricky here, as our model has three loss components and it depends not from given labels,
    # but from various components of input.
    loss_dict = {}

    # those outputs do not contribute to gradients
#    for name in ('policy', 'value'):
#        loss_dict[name] = zero_loss_func
#        loss_weights[name] = 0.0
    # those do
    for name in ('policy_loss', 'value_loss', 'entropy_loss'):
        loss_dict[name] = identity_loss_func

    train_m.compile(optimizer=RMSprop(), loss=loss_dict)
    # test run, to check correctness
    st = env.reset()
    obs, reward, done, _ = env.step(0)
    r = train_m.predict_on_batch([
        np.array([obs]), np.array([reward]), np.array([0])
    ])
    print(r)


    pass
