#!/usr/bin/env python
# Stochastic policy gradient: http://karpathy.github.io/2016/05/31/rl/
import argparse
import logging

import numpy as np

from rl_lib.wrappers import HistoryWrapper

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import gym, gym.wrappers

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda
from keras.optimizers import Adagrad, RMSprop
from keras import backend as K

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
    action_t = Input(shape=(1,), dtype='int32', name='action')
    advantage_t = Input(shape=(1,), name='advantage')

    fl_t = Flatten(name='flat')(in_t)
    l1_t = Dense(SIMPLE_L1_SIZE, activation='relu', name='l1')(fl_t)
    l2_t = Dense(SIMPLE_L2_SIZE, activation='relu', name='l2')(l1_t)
    policy_t = Dense(n_actions, name='policy', activation='softmax')(l2_t)

    def loss_func(args):
        p_t, act_t, adv_t = args
        oh_t = K.one_hot(act_t, n_actions)
        oh_t = K.squeeze(oh_t, 1)
        p_oh_t = K.log(1e-6 + K.sum(oh_t * p_t, axis=-1, keepdims=True))
        res_t = adv_t * p_oh_t
        return -res_t

    loss_t = Lambda(loss_func, output_shape=(1,), name='loss')([policy_t, action_t, advantage_t])

    return Model(input=[in_t, action_t, advantage_t], output=[policy_t, loss_t])


def create_batch(iter_no, env, run_model, num_episodes, steps_limit=1000, gamma=1.0, tau=0.20, min_samples=None):
    """
    Play given amount of episodes and prepare data to train on
    :param env: Environment instance
    :param run_model: Model to take actions
    :param num_episodes: count of episodes to run
    :return: batch in format required by model
    """
    samples = []
    rewards = []

    episodes_counter = 0
    while True:
        state = env.reset()
        step = 0
        sum_reward = 0.0
        episode = []
        loc_rewards = []
        while True:
            # chose action to take
            probs = run_model.predict_on_batch([
                np.array([state]),
                np.array([0]),
                np.array([0.0])
            ])[0][0]
            if np.random.random() < tau:
                action = np.random.randint(0, len(probs))
            else:
                action = np.random.choice(len(probs), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, probs, action))
            loc_rewards.append(reward)
            sum_reward = reward + gamma * sum_reward

            state = next_state
            step += 1

            if done or (steps_limit is not None and steps_limit == step):
                rewards.append(sum_reward)
                break

        # create reversed reward
        sum_reward = 0.0
        rev_rewards = []
        for r in reversed(loc_rewards):
            sum_reward = sum_reward * gamma + r
            rev_rewards.append(sum_reward)
        rev_rewards = np.copy(rev_rewards)
        rev_rewards -= np.mean(rev_rewards)
        rev_rewards /= np.std(rev_rewards)

        # generate samples from episode
        for reward, (state, probs, action) in zip(rev_rewards, reversed(episode)):
            samples.append((state, action, reward))
        episodes_counter += 1

        if min_samples is None:
            if episodes_counter == num_episodes:
                break
        elif len(samples) >= min_samples and episodes_counter >= num_episodes:
            break

    logger.info("%d: Have %d samples from %d episodes, mean final reward: %.3f, max: %.3f",
                iter_no, len(samples), episodes_counter, np.mean(rewards), np.max(rewards))
    # convert data to train format
    np.random.shuffle(samples)
    return list(map(np.array, zip(*samples)))


def create_fake_target(n_actions, batch_len):
    return [
        np.array([[0.0] * n_actions] * batch_len),
        np.array([0.0] * batch_len)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default="CartPole-v0", help="Environment name to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and save data into provided dir, default=disabled")
    parser.add_argument("-t", "--tau", type=float, default=0.2, help="Ratio of random steps, default=0.2")
    parser.add_argument("-i", "--iters", type=int, default=100, help="Count if iterations to take, default=100")
    args = parser.parse_args()

    env = make_env(args.env, args.monitor)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    model = make_model(state_shape, n_actions)
    model.summary()

    loss_dict = {
        # our model already outputs loss, so just take it as-is
        'loss': lambda y_true, y_pred: y_pred,
        # this will make zero gradients contribution
        'policy': lambda y_true, y_pred: y_true,
    }

    model.compile(optimizer=Adagrad(), loss=loss_dict)

    # gradient check
    if False:
        batch, action, advantage = create_batch(0, env, model, tau=0, num_episodes=1, steps_limit=10, min_samples=None)
        r = model.predict_on_batch([batch, action, advantage])
        fake_out = create_fake_target(n_actions, len(batch))
        l = model.train_on_batch([batch, action, advantage], fake_out)
        r2 = model.predict_on_batch([batch, action, advantage])
        logger.info("Test fit, mean loss: %s -> %s", np.mean(r[1]), np.mean(r2[1]))

    step_limit = 300
    if args.monitor is not None:
        step_limit = None

    for iter in range(args.iters):
        batch, action, advantage = create_batch(iter, env, model, tau=args.tau, num_episodes=10,
                                                steps_limit=step_limit, min_samples=5000)
        fake_out = create_fake_target(n_actions, len(batch))
        l = model.train_on_batch([batch, action, advantage], fake_out)
        #logger.info("Loss: %s", l[0])
    pass
