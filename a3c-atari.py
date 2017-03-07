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
from keras.layers import Input, Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Layer
from keras.optimizers import Adam, Adagrad
from keras import backend as K
import tensorflow as tf

import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.image

PLAYERS_COUNT = 50
HISTORY_STEPS = 4

#IMAGE_SIZE = (210, 160)
IMAGE_SIZE = (84, 84)
IMAGE_SHAPE = IMAGE_SIZE + (3*HISTORY_STEPS,)

BATCH_SIZE = 256
SYNC_MODEL_EVERY_BATCH = 50

def make_env(env_name, monitor_dir):
    env = HistoryWrapper(HISTORY_STEPS)(gym.make(env_name))
    if monitor_dir:
        env = gym.wrappers.Monitor(env, monitor_dir)
    return env


class SummaryWriter(Layer):
    def __init__(self, **kwargs):
        self.add_update(tf.summary.merge_all())
        super().__init__(**kwargs)

    def build(self, input_shape):
        pass

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        return x



def make_model(n_actions, train_mode=True):
    in_t = Input(shape=IMAGE_SHAPE, name='input')
    out_t = Conv2D(32, 5, 5, activation='relu', border_mode='same')(in_t)
    out_t = MaxPooling2D((2, 2))(out_t)
    out_t = Conv2D(32, 5, 5, activation='relu', border_mode='same')(out_t)
    out_t = MaxPooling2D((2, 2))(out_t)
    out_t = Conv2D(64, 4, 4, activation='relu', border_mode='same')(out_t)
    out_t = MaxPooling2D((2, 2))(out_t)
    out_t = Conv2D(64, 3, 3, activation='relu', border_mode='same')(out_t)
    out_t = Flatten(name='flat')(out_t)
    out_t = Dense(512, name='l1', activation='relu')(out_t)

    policy_t = Dense(n_actions, activation='softmax', name='policy')(out_t)
    value_t = Dense(1, name='value')(out_t)


    # we're not going to train -- just return policy and value from our state
    run_model = Model(input=in_t, output=[policy_t, value_t])
    if not train_mode:
        return run_model

    # we're training, life is much more interesting...
    action_t = Input(batch_shape=(None, 1), name='action', dtype='int32')
    advantage_t = Input(batch_shape=(None, 1), name="advantage")

    tf.summary.scalar("value", K.mean(value_t))
    tf.summary.scalar("advantage", K.mean(advantage_t))

    X_ENTROPY_BETA = 0.01

    def policy_loss_func(args):
        p_t, act_t, adv_t = args
        oh_t = K.one_hot(act_t, n_actions)
        oh_t = K.squeeze(oh_t, 1)
        p_oh_t = K.log(K.epsilon() + K.sum(oh_t * p_t, axis=-1, keepdims=True))
        res_t = adv_t * p_oh_t
        x_entropy_t = K.sum(p_t * K.log(K.epsilon() + p_t), axis=-1, keepdims=True)
        full_policy_loss_t = -res_t + X_ENTROPY_BETA * x_entropy_t
        tf.summary.scalar("loss_entropy", K.mean(x_entropy_t))
        tf.summary.scalar("loss_policy", K.mean(-res_t))
        tf.summary.scalar("loss_full", K.mean(full_policy_loss_t))
        return full_policy_loss_t

    loss_args = [policy_t, action_t, advantage_t]
    policy_loss_t = Lambda(policy_loss_func, output_shape=(1,), name='policy_loss')(loss_args)

    value_policy_model = Model(input=[in_t, action_t, advantage_t], output=[value_t, policy_loss_t])
    return value_policy_model


def preprocess(state):
    state = np.transpose(state, (1, 2, 3, 0))
    state = np.reshape(state, (state.shape[0], state.shape[1], state.shape[2]*state.shape[3]))

    state = state.astype(np.float32)
    res = cv2.resize(state, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    res -= 128
    res /= 128
    return res


image_index = {}

def save_state(rgb_arr, prefix='state'):
    # global image_index
    # idx = image_index.get(prefix, 0)
    # matplotlib.image.imsave("%s_%05d.png" % (prefix, idx), rgb_arr)
    # image_index[prefix] = idx + 1
    pass


class Player:
    def __init__(self, env, model, reward_steps, gamma, max_steps, player_index):
        self.env = env
        self.model = model
        self.reward_steps = reward_steps
        self.gamma = gamma
        self.state = env.reset()

        self.memory = []
        self.episode_reward = 0.0
        self.step_index = 0
        self.max_steps = max_steps
        self.player_index = player_index

    def play(self, steps):
        result = []

        for _ in range(steps):
            self.step_index += 1
            pr_state = preprocess(self.state)
            probs, value = self.model.predict_on_batch([
                np.array([pr_state]),
            ])
            probs, value = probs[0], value[0][0]
            # take action
            action = np.random.choice(len(probs), p=probs)
            self.state, reward, done, _ = self.env.step(action)

            self.episode_reward += reward
            self.memory.append((pr_state, action, reward, value))
            if done or self.step_index > self.max_steps:
                self.state = self.env.reset()
                logging.info("%3d: Episode done @ step %d: sum reward %d",
                             self.player_index, self.step_index, int(self.episode_reward))
                self.episode_reward = 0.0
                self.step_index = 0
                result.extend(self._memory_to_samples(is_done=done))
                break
            elif len(self.memory) == self.reward_steps + 1:
                result.extend(self._memory_to_samples(is_done=False))

        return result

    def _memory_to_samples(self, is_done):
        """
        From existing memory, generate samples
        :param is_done: is episode done
        :return: list of training samples
        """
        result = []
        R, last_item = 0.0, None

        if not is_done:
            last_item = self.memory.pop()
            R = last_item[-1]

        for state, action, reward, value in reversed(self.memory):
            R = reward + R * self.gamma
            advantage = R - value
            result.append((state, action, R, advantage))

        self.memory = [] if is_done else [last_item]
        return result


def generate_batches(players, batch_size):
    samples = []

    while True:
        for player in players:
            samples.extend(player.play(10))
        while len(samples) >= batch_size:
            states, actions, rewards, advantages = list(map(np.array, zip(*samples[:batch_size])))
            yield [states, actions, advantages], [rewards, rewards]
            samples = samples[batch_size:]


def summary(y_true, y_pred):
    return tf.summary.merge_all()


def make_reward_summary(rewards):
    summ = tf.Summary()
    summ_value = summ.value.add()
    summ_value.simple_value = np.mean(rewards)
    summ_value.tag = "reward"
    return summ


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-e", "--env", default="CartPole-v0", help="Environment name to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and save data into provided dir, default=disabled")
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma for reward discount, default=1.0")
    parser.add_argument("-i", "--iters", type=int, default=10000, help="Count of iterations to take, default=100")
    parser.add_argument("--steps", type=int, default=10, help="Count of steps to use in reward estimation")
    args = parser.parse_args()

    env = make_env(args.env, args.monitor)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    value_policy_model = make_model(n_actions, train_mode=True)
    run_model = make_model(n_actions, train_mode=False)
    value_policy_model.summary()

    loss_dict = {
        'value': 'mse',
        'policy_loss': lambda y_true, y_pred: y_pred
    }

    # A bit of keras magic
    metrics_dict = {
        'value': summary
    }

    value_policy_model.compile(optimizer=Adagrad(), loss=loss_dict, metrics=metrics_dict)

    summary_writer = tf.summary.FileWriter("logs/" + args.name)

    players = [Player(make_env(args.env, args.monitor), run_model, reward_steps=args.steps, gamma=args.gamma,
                      max_steps=40000, player_index=idx)
               for idx in range(PLAYERS_COUNT)]

    for iter_idx, (x_batch, y_batch) in enumerate(generate_batches(players, BATCH_SIZE)):
        for _ in range(3):
            l = value_policy_model.train_on_batch(x_batch, y_batch)
        l_dict = dict(zip(value_policy_model.metrics_names, l))

        summary_writer.add_summary(make_reward_summary(y_batch[0]), global_step=iter_idx)
        summary_writer.add_summary(l_dict['value_summary'], global_step=iter_idx)
        summary_writer.flush()
        if iter_idx % SYNC_MODEL_EVERY_BATCH == 0:
            run_model.set_weights(value_policy_model.get_weights())
            logger.info("Models synchronized, iter %d", iter_idx)

