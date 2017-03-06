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
from keras.optimizers import Adam, Adagrad
from keras import backend as K
from keras.callbacks import TensorBoard

import cv2

PLAYERS_COUNT = 50
HISTORY_STEPS = 4
SIMPLE_L1_SIZE = 50
SIMPLE_L2_SIZE = 50

IMAGE_SIZE = (210, 160)
IMAGE_SHAPE = IMAGE_SIZE + (3*HISTORY_STEPS,)

BATCH_SIZE = 256

def make_env(env_name, monitor_dir):
    env = HistoryWrapper(HISTORY_STEPS)(gym.make(env_name))
    if monitor_dir:
        env = gym.wrappers.Monitor(env, monitor_dir)
    return env


def make_model(in_t, out_t, n_actions, train_mode=True):
    policy_t = Dense(n_actions, activation='softmax', name='policy')(out_t)
    value_t = Dense(1, name='value')(out_t)

    # we're not going to train -- just return policy and value from our state
    run_model = Model(input=in_t, output=[policy_t, value_t])
    if not train_mode:
        return run_model

    # we're training, life is much more interesting...
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
        return -res_t + X_ENTROPY_BETA * x_entropy_t

    loss_args = [policy_t, action_t, advantage_t]
    policy_loss_t = Lambda(policy_loss_func, output_shape=(1,), name='policy_loss')(loss_args)

    value_policy_model = Model(input=[in_t, action_t, advantage_t], output=[value_t, policy_loss_t])
    return run_model, value_policy_model


def preprocess(state):
    state = np.transpose(state, (1, 2, 3, 0))
    state = np.reshape(state, (state.shape[0], state.shape[1], state.shape[2]*state.shape[3]))

    state = state.astype(np.float32)
#    res = cv2.resize(state, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    res = state
    res -= 128
    res /= 128
    return res


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
                np.array([preprocess(state)]),
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
            samples.append((preprocess(state), action, sum_reward, advantage))

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


image_index = {}

def save_state(rgb_arr, prefix='state'):
    global image_index
    idx = image_index.get(prefix, 0)
    matplotlib.image.imsave("%s_%05d.png" % (prefix, idx), rgb_arr)
    image_index[prefix] = idx + 1


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
            probs, value = run_model.predict_on_batch([
                np.array([pr_state]),
            ])
            probs, value = probs[0], value[0][0]
            # take action
            action = np.random.choice(len(probs), p=probs)
            self.state, reward, done, _ = env.step(action)

            self.episode_reward += reward
            self.memory.append((pr_state, action, reward, value))
            if done or self.step_index > self.max_steps:
                self.state = env.reset()
                logging.info("%d: Episode done @ step %d: sum reward %f",
                             self.player_index, self.step_index, self.episode_reward)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-e", "--env", default="CartPole-v0", help="Environment name to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and save data into provided dir, default=disabled")
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma for reward discount, default=1.0")
    parser.add_argument("--eps", type=float, default=0.0, help="Ratio of random steps, default=0.0")
    parser.add_argument("--eps-decay", default=1.0, type=float, help="Set eps decay, default=1.0")
    parser.add_argument("-i", "--iters", type=int, default=10000, help="Count of iterations to take, default=100")
    parser.add_argument("--steps", type=int, default=10, help="Count of steps to use in reward estimation")
    parser.add_argument("--min-episodes", type=int, default=10, help="Minimum amount of episodes to play, default=10")
    parser.add_argument("--min-samples", type=int, default=60000, help="Minimum amount of learning samples to generate, default=60000")
    parser.add_argument("--max-steps", type=int, default=6000, help="Maximum count of steps per episode, default=6000")
    args = parser.parse_args()

    env = make_env(args.env, args.monitor)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

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

    run_model, value_policy_model = make_model(in_t, out_t, n_actions, train_mode=True)
    value_policy_model.summary()

    loss_dict = {
        'value': 'mse',
        'policy_loss': lambda y_true, y_pred: y_pred
    }

    value_policy_model.compile(optimizer=Adam(lr=0.001, epsilon=1e-3), loss=loss_dict)

    players = [Player(make_env(args.env, args.monitor), run_model, reward_steps=args.steps, gamma=args.gamma,
                      max_steps=1000, player_index=idx)
               for idx in range(PLAYERS_COUNT)]

    for iter_idx, (x_batch, y_batch) in enumerate(generate_batches(players, BATCH_SIZE)):
        l = value_policy_model.train_on_batch(x_batch, y_batch)
#        logger.info("Learning iteration %d", iter_idx)
#        print(l)

