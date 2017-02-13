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
from keras.layers import Input, Dense, Flatten, Lambda
from keras.optimizers import Adagrad, RMSprop
from keras import backend as K

HISTORY_STEPS = 1
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
#    reward_t = Input(batch_shape=(None, 1), name='sum_reward')
    action_t = Input(batch_shape=(None, 1), name='action', dtype='int32')
    advantage_t = Input(batch_shape=(None, 1), name="advantage")

    X_ENTROPY_BETA = 0.01

    def policy_loss_func(args):
        p_t, act_t, adv_t = args
        oh_t = K.one_hot(act_t, n_actions)
        oh_t = K.squeeze(oh_t, 1)
        p_oh_t = K.log(1e-6 + K.sum(oh_t * p_t, axis=-1, keepdims=True))
        res_t = adv_t * p_oh_t
        x_entropy_t = K.sum(p_t * K.log(1e-6 + p_t), axis=-1, keepdims=True)
        return -res_t - X_ENTROPY_BETA * x_entropy_t

    loss_args = [policy_t, action_t, advantage_t]
    policy_loss_t = Lambda(policy_loss_func, output_shape=(1,), name='policy_loss')(loss_args)
    return Model(input=[in_t, action_t, advantage_t], output=[policy_t, value_t, policy_loss_t])

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

    # policy_model = Model(input=in_t, output=policy_t)
    # value_model = Model(input=in_t, output=value_t)
    # return run_model, policy_model, value_model

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

    episodes_counter = 0
    while True:
        state = env.reset()
        step = 0
        sum_reward = 0.0
        episode = []
        loc_rewards = []
        while True:
            # chose action to take
            probs, value, loss = run_model.predict_on_batch([
                np.array([state]),
                np.array([0]),
                np.array([0.0])
            ])
            probs, value = probs[0], value[0][0]
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
        rev_rewards = []
        for r in reversed(loc_rewards):
            sum_reward = sum_reward * gamma + r
            rev_rewards.append(sum_reward)
        # rev_rewards = np.copy(rev_rewards)
        # rev_rewards -= np.mean(rev_rewards)
        # rev_rewards /= np.std(rev_rewards)

        # generate samples from episode
        for reward, (state, probs, value, action) in zip(rev_rewards, reversed(episode)):
            advantage = reward - value
            samples.append((state, action, advantage, reward))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default="CartPole-v0", help="Environment name to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and save data into provided dir, default=disabled")
    parser.add_argument("--eps", type=float, default=0.1, help="Ratio of random steps, default=0.2")
    parser.add_argument("-i", "--iters", type=int, default=100, help="Count if iterations to take, default=100")
    args = parser.parse_args()

    env = make_env(args.env, args.monitor)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    model = make_model(state_shape, n_actions, train_mode=True)
    model.summary()

    losses_dict = {
        # policy loss is already calculated, just return it
        'policy_loss': lambda y_true, y_pred: y_pred,
        # value loss is calculated against reversed reward
        'value': 'mse',
        # policy result doesn't need to contribute to loss, just kill gradients
        'policy': lambda y_true, y_pred: y_true
    }

    model.compile(optimizer=RMSprop(lr=0.0001), loss=losses_dict)

    # gradient check
    if False:
        batch, action, advantage, reward = create_batch(0, env, model, eps=0.0, num_episodes=1, steps_limit=10, min_samples=None)
        r = model.predict_on_batch([batch, action, advantage])
        l = model.train_on_batch([batch, action, advantage], [reward]*3)
        r2 = model.predict_on_batch([batch, action, advantage])
        logger.info("Test fit, mean loss: %s -> %s", np.mean(r[2]), np.mean(r2[2]))

    step_limit = 500
    if args.monitor is not None:
        step_limit = None

    for iter in range(args.iters):
        batch, action, advantage, reward = create_batch(iter, env, model, eps=args.eps, num_episodes=1,
                                                steps_limit=step_limit, min_samples=500)
        l = model.fit([batch, action, advantage], [reward]*3, verbose=0)
#        logger.info("Loss: %s", l[0])
    pass
