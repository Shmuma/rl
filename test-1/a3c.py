#!/usr/bin/env python
# Quick-n-dirty implementation of Advantage Actor-Critic method from https://arxiv.org/abs/1602.01783
from .wrappers import HistoryWrapper

import logging
import numpy as np
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import gym, gym.wrappers

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda, merge
from keras.optimizers import Adagrad, RMSprop

HISTORY_STEPS = 4
SIMPLE_L1_SIZE = 50
SIMPLE_L2_SIZE = 50


def make_env(env_name, monitor_dir):
    env = HistoryWrapper(HISTORY_STEPS)(gym.make(env_name))
    if monitor_dir:
        env = gym.wrappers.Monitor(env, monitor_dir)
    return env


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
    # reward_t = Input(batch_shape=(None, 1), name='sum_reward')
    # action_t = Input(batch_shape=(None, 1), name='action', dtype='int32')

    # value loss
    # value_loss_t = merge([value_t, reward_t], mode=lambda vals: mean_squared_error(vals[0], vals[1]),
    #                      output_shape=(1, ), name='value_loss')
    # BETA = 0.01
    # entropy_loss_t = Lambda(lambda p: BETA * K.sum(p * K.log(p)), output_shape=(1, ), name='entropy_loss')(policy_t)

    # def policy_loss_func(args):
    #     policy_t, action_t = args
    #     oh = K.one_hot(action_t, nb_classes=n_actions)
    #     oh = K.squeeze(oh, 1)
    #     p = K.log(policy_t) * oh
    #     p = K.sum(p, axis=-1, keepdims=True)
    #     return p * K.stop_gradient(value_t - reward_t)
    #
    # policy_loss_t = merge([policy_t, action_t], mode=policy_loss_func,
    #                       output_shape=(1, ), name='policy_loss')
#    loss_t = merge([policy_loss_t, policy_loss_t], mode='ave', name='loss')

    policy_model = Model(input=in_t, output=policy_t)
    value_model = Model(input=in_t, output=value_t)
    return run_model, policy_model, value_model


def create_batch(env, run_model, num_episodes, steps_limit=1000, gamma=1.0):
    """
    Play given amount of episodes and prepare data to train on
    :param env: Environment instance
    :param run_model: Model to take actions
    :param num_episodes: count of episodes to run
    :return: batch in format required by model
    """
    episodes = []
    rewards = []
    values = []
    for _ in range(num_episodes):
        state = env.reset()
        sum_reward = 0.0
        step = 0
        while True:
            # chose action to take
            policy, value = run_model.predict_on_batch(np.array([state]))
            policy = policy[0]
            value = value[0]
            values.append(value)
            action = np.argmax(policy)
            next_state, reward, done, _ = env.step(action)
            sum_reward = gamma*sum_reward + reward
            policy_tgt = np.copy(policy)
            policy_tgt[action] *= sum_reward - value
            episodes.append((state, sum_reward, policy_tgt))
            state = next_state
            step += 1
            if done or (steps_limit is not None and steps_limit == step):
                rewards.append(sum_reward)
                break
    logger.info("Mean final reward: %.3f, max: %.3f, value: %s", np.mean(rewards), np.max(rewards), np.mean(values))
    # convert data to train format
    np.random.shuffle(episodes)
    return list(map(np.array, zip(*episodes)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default="CartPole-v0", help="Environment name to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and save data into provided dir, default=disabled")
    args = parser.parse_args()

    env = make_env(args.env, args.monitor)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    run_m, policy_m, value_m = make_model(state_shape, n_actions, train_mode=True)
    policy_m.summary()

    policy_m.compile(optimizer=Adagrad(), loss='mse')
    value_m.compile(optimizer=Adagrad(), loss='mse')

    # test run, to check correctness
    # if args.monitor is None:
    #     st = env.reset()
    #     obs, reward, done, _ = env.step(0)
    #     r = train_m.predict_on_batch([
    #         np.array([obs]), np.array([reward]), np.array([0])
    #     ])
    #     print(r)

    epoch_limit = 10
    step_limit = 200
    if args.monitor is not None:
        step_limit = None

    for iter in range(100):
        batch, rewards, policy_y = create_batch(env, run_m, num_episodes=100, steps_limit=step_limit)
#        fake_y = np.zeros(shape=(len(batch[2]),))
        # iterate until our losses decreased 10 times or epoches limit exceeded
        start_p_loss, start_v_loss = None, None
        p_loss, v_loss = None, None
        converged = False
        for epoch in range(epoch_limit):
            p_h = policy_m.fit(batch, policy_y, verbose=0, batch_size=128)
            p_loss = np.min(p_h.history['loss'])

            v_h = value_m.fit(batch, rewards, verbose=0, batch_size=128)
            v_loss = np.min(v_h.history['loss'])

            if start_p_loss is None:
                start_p_loss = np.max(p_h.history['loss'])
                start_v_loss = np.max(v_h.history['loss'])
            else:
                if start_p_loss / p_loss > 2 and start_v_loss / v_loss > 2:
                    logger.info("%d: after %d epoches: p_loss %.3f -> %.3f, v_loss %.3f -> %.3f",
                                iter, epoch, start_p_loss, p_loss, start_v_loss, v_loss)
                    converged = True
                    break
        if not converged:
            logger.info("%d: haven't converged: p_loss %.3f -> %.3f, v_loss %.3f -> %.3f",
                        iter, start_p_loss, p_loss, start_v_loss, v_loss)
    pass
