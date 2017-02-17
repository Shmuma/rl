#!/usr/bin/env python
# Quick-n-dirty implementation of Advantage Actor-Critic method from https://arxiv.org/abs/1602.01783
import argparse
import logging

import numpy as np

from rl_lib.wrappers import HistoryWrapper

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import gym, gym.wrappers

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Permute, Reshape, BatchNormalization
from keras.optimizers import Adagrad
from keras import backend as K

HISTORY_STEPS = 3
SIMPLE_L1_SIZE = 50
SIMPLE_L2_SIZE = 50


def make_env(env_name, monitor_dir):
    env = HistoryWrapper(HISTORY_STEPS)(gym.make(env_name))
    if monitor_dir:
        env = gym.wrappers.Monitor(env, monitor_dir)
    return env


def make_model(state_shape, n_actions, train_mode=True):
    in_t = Input(shape=(HISTORY_STEPS,) + state_shape, name='input')
    # bring together color channel (4-th dim) and history (1-st dim)
    post_in_t = Permute(dims=(2, 3, 4, 1))(in_t)

    channels = state_shape[-1]
    res_shape = (state_shape[0], state_shape[1], channels * HISTORY_STEPS)
    # put color channel and history together
    post_in_t = Reshape(target_shape=res_shape)(post_in_t)
    post_in_t = BatchNormalization()(post_in_t)

    out_t = Conv2D(64, 3, 3, activation='relu')(post_in_t)
    out_t = Conv2D(64, 3, 3, activation='relu')(out_t)
    out_t = MaxPooling2D((3, 3))(out_t)
    out_t = Conv2D(128, 2, 2, activation='relu')(out_t)
    out_t = MaxPooling2D((3, 3))(out_t)
    out_t = Flatten(name='flat')(out_t)
    out_t = Dense(SIMPLE_L1_SIZE, activation='relu', name='l1')(out_t)
    out_t = Dense(SIMPLE_L2_SIZE, activation='relu', name='l2')(out_t)
    policy_t = Dense(n_actions, activation='softmax', name='policy')(out_t)
    value_t = Dense(1, name='value')(out_t)

    # we're not going to train -- just return policy and value from our state
    run_model = Model(input=in_t, output=[policy_t, value_t])
    if not train_mode:
        return run_model

    # we're training, life is much more interesting...
#    reward_t = Input(batch_shape=(None, 1), name='sum_reward')
    action_t = Input(batch_shape=(None, 1), name='action', dtype='int32')
    advantage_t = Input(batch_shape=(None, 1), name="advantage")

    X_ENTROPY_BETA = 0.01

    def policy_loss_func(args):
        p_t, act_t, adv_t = args
        oh_t = K.one_hot(act_t, n_actions)
        oh_t = K.squeeze(oh_t, 1)
        p_oh_t = K.log(K.epsilon() + K.sum(oh_t * p_t, axis=-1, keepdims=True))
        res_t = 2.0 * K.sigmoid(adv_t) * p_oh_t
#        x_entropy_t = K.sum(p_t * K.log(1e-6 + p_t), axis=-1, keepdims=True)
        return -res_t# - X_ENTROPY_BETA * x_entropy_t

    loss_args = [policy_t, action_t, advantage_t]
    policy_loss_t = Lambda(policy_loss_func, output_shape=(1,), name='policy_loss')(loss_args)

    value_model = Model(input=in_t, output=value_t)
    policy_model = Model(input=[in_t, action_t, advantage_t], output=policy_loss_t)

    return run_model, value_model, policy_model


def create_batch(iter_no, env, run_model, num_episodes, steps_limit=1000, gamma=1.0, eps=0.20, min_samples=None):
    """
    Play given amount of episodes and prepare data to train on
    :param env: Environment instance
    :param run_model: Model to take actions
    :param num_episodes: count of episodes to run
    :return: batch in format required by model
    """
    samples = []
    rewards = []
    values = []
    advantages = []

    episodes_counter = 0
    while True:
        state = env.reset()
        step = 0
        sum_reward = 0.0
        episode = []
        loc_rewards = []
        while True:
            # chose action to take
            probs, value = run_model.predict_on_batch([
                np.array([state]),
            ])
            probs, value = probs[0], value[0][0]
            values.append(value)
            if np.random.random() < eps:
                action = np.random.randint(0, len(probs))
                probs = np.ones_like(probs)
                probs /= np.sum(probs)
            else:
                action = np.random.choice(len(probs), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, probs, value, action))
            loc_rewards.append(reward)
            sum_reward = reward + gamma * sum_reward
            state = next_state
            step += 1

            if done or (steps_limit is not None and steps_limit == step):
                rewards.append(sum_reward)
                break

        # create reversed reward
        sum_reward = 0.0
        if not done:
            sum_reward = value
        rev_rewards = []
        for r in reversed(loc_rewards):
            sum_reward = sum_reward * gamma + r
            rev_rewards.append(sum_reward)

        # generate samples from episode
        for reward, (state, probs, value, action) in zip(rev_rewards, reversed(episode)):
            advantage = reward - value
            advantages.append(advantage)
            samples.append((state, action, reward, advantage))
        episodes_counter += 1

        if min_samples is None:
            if episodes_counter == num_episodes:
                break
        elif len(samples) >= min_samples and episodes_counter >= num_episodes:
            break

    logger.info("%d: eps=%.3f: got %d samples from %d episodes, mean final reward: %.3f, max: %.3f, "
                "mean value: %.3f, max value: %.3f, mean adv: %.3f",
                iter_no, eps, len(samples), episodes_counter, np.mean(rewards), np.max(rewards),
                np.mean(values), np.max(values), np.mean(advantages))
    # convert data to train format
    np.random.shuffle(samples)

    return list(map(np.array, zip(*samples)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default="CartPole-v0", help="Environment name to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and save data into provided dir, default=disabled")
    parser.add_argument("--eps", type=float, default=0.0, help="Ratio of random steps, default=0.2")
    parser.add_argument("--eps-decay", default=1.0, type=float, help="Set eps decay, default=1.0")
    parser.add_argument("-i", "--iters", type=int, default=100, help="Count if iterations to take, default=100")
    parser.add_argument("--limit", type=int, help="Limit count of steps in each episode, default=no limit")
    parser.add_argument("--episodes", type=int, default=1, help="Count of episodes to play")
    args = parser.parse_args()

    env = make_env(args.env, args.monitor)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    run_model, value_model, policy_model = make_model(state_shape, n_actions, train_mode=True)
    run_model.summary()

    value_model.compile(optimizer=Adagrad(), loss='mse')
    policy_model.compile(optimizer=Adagrad(), loss=lambda y_true, y_pred: y_pred)
    eps = args.eps

    for iter in range(args.iters):
        batch, action, reward, advantage = create_batch(iter, env, run_model, eps=eps, num_episodes=args.episodes,
                                                steps_limit=args.limit, min_samples=2000)
        l = value_model.fit(batch, reward, verbose=0, batch_size=128, nb_epoch=3)
        l = policy_model.fit([batch, action, advantage], np.zeros_like(reward), verbose=0, batch_size=128, nb_epoch=3)
        eps *= args.eps_decay
    pass
