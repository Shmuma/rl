#!/usr/bin/env python
# Quick-n-dirty implementation of Advantage Actor-Critic method from https://arxiv.org/abs/1602.01783
import argparse
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda
from keras.optimizers import Adam, Adagrad
from keras import backend as K

from algo_lib.common import make_env, summarize_gradients, summary_value
from algo_lib.a3c import make_models
from algo_lib.player import Player, generate_batches

HISTORY_STEPS = 4
SIMPLE_L1_SIZE = 50
SIMPLE_L2_SIZE = 50

SUMMARY_EVERY_BATCH = 10


def make_model(in_t, out_t, n_actions, train_mode=True):
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
        res_t = adv_t * p_oh_t
        x_entropy_t = K.sum(p_t * K.log(K.epsilon() + p_t), axis=-1, keepdims=True)
        print(x_entropy_t)
        print(res_t)
        return -res_t + X_ENTROPY_BETA * x_entropy_t

    loss_args = [policy_t, action_t, advantage_t]
    policy_loss_t = Lambda(policy_loss_func, output_shape=(1,), name='policy_loss')(loss_args)

    value_policy_model = Model(input=[in_t, action_t, advantage_t], output=[value_t, policy_loss_t])
    return run_model, value_policy_model


def create_batch(iter_no, env, run_model, num_episodes, steps_limit=None,
                 gamma=1.0, eps=0.20, min_samples=None, n_steps=10):
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
        # list of episode steps (state, action)
        episode = []
        # list of tuples (reward, value)
        reward_value = []
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
            episode.append((state, action))
            reward_value.append((reward, value))
            sum_reward = reward + gamma * sum_reward
            state = next_state
            step += 1

            if done or (steps_limit is not None and steps_limit == step):
                rewards.append(sum_reward)
                break

        # generate samples from episode
        for idx, (state, action) in enumerate(episode):
            # calculate discounted reward and advantage for this step
            reward_value_window = reward_value[idx:idx+n_steps+1]
            if len(reward_value_window) > n_steps:
                last_value = reward_value_window.pop()[1]
            else:
                last_value = 0

            sum_reward = last_value
            for reward, _ in reversed(reward_value_window):
                sum_reward = sum_reward * gamma + reward

            advantage = sum_reward - reward_value[idx][1]
            advantages.append(advantage)
            samples.append((state, action, sum_reward, advantage))

        episodes_counter += 1

        if min_samples is None:
            if episodes_counter == num_episodes:
                break
        elif len(samples) >= min_samples and episodes_counter >= num_episodes:
            break

    logger.info("%d: Have %d samples from %d episodes, mean final reward: %.3f, max: %.3f, "
                "mean value: %.3f, max value: %.3f, mean adv: %.3f, eps=%f",
                iter_no, len(samples), episodes_counter, np.mean(rewards), np.max(rewards),
                np.mean(values), np.max(values), np.mean(advantages), eps)
    # convert data to train format
    np.random.shuffle(samples)

    batch, action, sum_reward, advantage = list(map(np.array, zip(*samples)))
    return batch, action, sum_reward, advantage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default="CartPole-v0", help="Environment name to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and save data into provided dir, default=disabled")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma for reward discount, default=1.0")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    args = parser.parse_args()

    env = make_env(args.env, args.monitor, history_steps=HISTORY_STEPS)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    in_t = Input(shape=(HISTORY_STEPS,) + state_shape, name='input')
    fl_t = Flatten(name='flat')(in_t)
    l1_t = Dense(SIMPLE_L1_SIZE, activation='relu', name='in_l1')(fl_t)
    out_t = Dense(SIMPLE_L2_SIZE, activation='relu', name='in_l2')(l1_t)

    run_model, value_policy_model = make_models(in_t, out_t, n_actions)
    value_policy_model.summary()

    loss_dict = {
        'value': 'mse',
        'policy_loss': lambda y_true, y_pred: y_pred
    }
    # Adam(lr=0.001, epsilon=1e-3, clipnorm=0.1)
    value_policy_model.compile(optimizer=Adam(lr=0.0002, clipnorm=0.1), loss=loss_dict)

    # keras summary magic
    summary_writer = tf.summary.FileWriter("logs/" + args.name)
    summarize_gradients(value_policy_model)
    value_policy_model.metrics_names.append("value_summary")
    value_policy_model.metrics_tensors.append(tf.summary.merge_all())

    players = [
        Player(env, reward_steps=10, gamma=0.99, max_steps=40000, player_index=idx)
        for idx in range(10)
    ]

    for iter_idx, (x_batch, y_batch) in enumerate(generate_batches(run_model, players, 128)):
        l = value_policy_model.train_on_batch(x_batch, y_batch)

        if iter_idx % SUMMARY_EVERY_BATCH == 0:
            l_dict = dict(zip(value_policy_model.metrics_names, l))
            done_rewards = Player.gather_done_rewards(*players)

            if done_rewards:
                summary_value("reward_episode_mean", np.mean(done_rewards), summary_writer, iter_idx)
                summary_value("reward_episode_max", np.max(done_rewards), summary_writer, iter_idx)

            summary_value("reward_batch", np.mean(y_batch[0]), summary_writer, iter_idx)
            summary_value("loss_value", l_dict['value_loss'], summary_writer, iter_idx)
            summary_value("loss_full", l_dict['loss'], summary_writer, iter_idx)
            summary_writer.add_summary(l_dict['value_summary'], global_step=iter_idx)
            summary_writer.flush()
        run_model.set_weights(value_policy_model.get_weights())
    pass
