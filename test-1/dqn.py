#!/usr/bin/env python
# n-step Q-learning
import argparse
import logging

import numpy as np

from rl_lib.wrappers import HistoryWrapper

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import gym, gym.wrappers

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adagrad, RMSprop

HISTORY_STEPS = 2
SIMPLE_L1_SIZE = 50
SIMPLE_L2_SIZE = 50


def make_env(env_name, monitor_dir):
    env = HistoryWrapper(HISTORY_STEPS)(gym.make(env_name))
    if monitor_dir:
        env = gym.wrappers.Monitor(env, monitor_dir)
    return env


def make_model(state_shape, n_actions):
    in_t = Input(shape=(HISTORY_STEPS,) + state_shape, name='input')
    fl_t = Flatten(name='flat')(in_t)
    l1_t = Dense(SIMPLE_L1_SIZE, activation='relu', name='l1')(fl_t)
    l2_t = Dense(SIMPLE_L2_SIZE, activation='relu', name='l2')(l1_t)
    value_t = Dense(n_actions, name='value')(l2_t)

    return Model(input=in_t, output=value_t)


def create_batch(iter_no, env, run_model, num_episodes, steps_limit=1000, gamma=1.0):
    """
    Play given amount of episodes and prepare data to train on
    :param env: Environment instance
    :param run_model: Model to take actions
    :param num_episodes: count of episodes to run
    :return: batch in format required by model
    """
    samples = []
    rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        step = 0
        sum_reward = 0.0
        episode = []
        while True:
            # chose action to take
            q_value = run_model.predict_on_batch(np.array([state]))[0]
            action = np.argmax(q_value)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, q_value, action, reward))
            sum_reward = reward + gamma * sum_reward

            state = next_state
            step += 1

            # if episode is done, we save it in last_q
            if done:
                last_q = None
                rewards.append(sum_reward)
                break
            # otherwise, we'll need last_q as estimation of total reward
            elif steps_limit is not None and steps_limit == step:
                last_q = run_model.predict_on_batch(np.array([state]))[0]
                rewards.append(sum_reward)
                break

        # now we need to unroll our episodes backward to generate training samples
        for state, q_value, action, reward in reversed(episode):
            # get approximated target reward for this state
            R = reward
            if last_q is not None:
                R += gamma * max(last_q)
            target_q = np.copy(q_value)
            target_q[action] = R
            samples.append((state, target_q))
            last_q = q_value

    logger.info("%d: Have %d samples, mean final reward: %.3f, max: %.3f",
                iter_no, len(samples), np.mean(rewards), np.max(rewards))
    # convert data to train format
    np.random.shuffle(samples)
    return list(map(np.array, zip(*samples)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default="CartPole-v0", help="Environment name to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and save data into provided dir, default=disabled")
    args = parser.parse_args()

    env = make_env(args.env, args.monitor)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    model = make_model(state_shape, n_actions)
    model.summary()

    model.compile(optimizer=Adagrad(), loss='mse')

    # test run, to check correctness
    if args.monitor is None:
        st = env.reset()
        r = model.predict_on_batch([
            np.array([st])
        ])
        print(r)

    epoch_limit = 10
    step_limit = 300
    if args.monitor is not None:
        step_limit = None

    for iter in range(100):
        batch, target_y = create_batch(iter, env, model, num_episodes=20, steps_limit=step_limit)
        # iterate until our losses decreased 10 times or epoches limit exceeded
        start_loss = None
        loss = None
        converged = False
        for epoch in range(epoch_limit):
            p_h = model.fit(batch, target_y, verbose=0, batch_size=128)
            loss = np.min(p_h.history['loss'])

            if start_loss is None:
                start_loss = np.max(p_h.history['loss'])
            else:
                if start_loss / loss > 1.5:
                    # logger.info("%d: after %d epoches: loss %.3f -> %.3f",
                    #             iter, epoch, start_loss, loss)
                    converged = True
                    break
        # if not converged:
        #     logger.info("%d: haven't converged: loss %.3f -> %.3f",
        #                 iter, start_loss, loss)
    pass
