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
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adagrad, RMSprop

HISTORY_STEPS = 4
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
    value_t = Dense(n_actions, name='value', activation='softmax')(l2_t)

    return Model(input=in_t, output=value_t)


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
    episodes = []

    samples_counter = 0
    while True:
        state = env.reset()
        step = 0
        sum_reward = 0.0
        episode = []
        while True:
            # chose action to take
            probs = run_model.predict_on_batch(np.array([state]))[0]
            if np.random.random() < tau:
                action = np.random.randint(0, len(probs))
            else:
                action = np.random.choice(len(probs), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, probs, action, reward))
            sum_reward = reward + gamma * sum_reward

            state = next_state
            step += 1
            samples_counter += 1

            if done or (steps_limit is not None and steps_limit == step):
                rewards.append(sum_reward)
                break
        episodes.append(episode)
        if min_samples is None:
            if len(episodes) == num_episodes:
                break
        elif samples_counter >= min_samples and len(episodes) >= num_episodes:
            break

    # mean_final_reward = np.mean(rewards)
    # disp = np.max(rewards) - np.min(rewards)
    # convert episodes into samples
    for episode, episode_reward in zip(episodes, rewards):
        # now we need to unroll our episode backward to generate training samples
        for state, probs, action, reward in reversed(episode):
            target = episode_reward * np.log(probs[action])
            samples.append((state, target))

    logger.info("%d: Have %d samples from %d episodes, mean final reward: %.3f, max: %.3f",
                iter_no, len(samples), len(episodes), np.mean(rewards), np.max(rewards))
    # convert data to train format
    np.random.shuffle(samples)
    return list(map(np.array, zip(*samples)))


def test_loss(y_true, y_pred):
    return y_true + y_pred*0.0


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

    model.compile(optimizer=Adagrad(), loss=test_loss)

    # gradient check
    if True:
        batch, target_y = create_batch(0, env, model, tau=0, num_episodes=1, steps_limit=10, min_samples=None)
        r = model.predict_on_batch(batch)
        l = model.train_on_batch(batch, target_y)
        r2 = model.predict_on_batch(batch)


    epoch_limit = 2
    step_limit = 300
    if args.monitor is not None:
        step_limit = None

    for iter in range(args.iters):
        batch, target_y = create_batch(iter, env, model, tau=args.tau,
                                       num_episodes=10, steps_limit=step_limit, min_samples=500)
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
                if start_loss / loss > 2:
                    break
    pass
