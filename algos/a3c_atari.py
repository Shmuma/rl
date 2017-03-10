#!/usr/bin/env python
# Quick-n-dirty implementation of Advantage Actor-Critic method from https://arxiv.org/abs/1602.01783
import os
import argparse
import logging
import time
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

from algo_lib.common import make_env, summarize_gradients, summary_value
from algo_lib.atari_opts import HISTORY_STEPS, net_input
from algo_lib.a3c import make_run_model, make_train_model


PLAYERS_COUNT = 50

BATCH_SIZE = 128

SUMMARY_EVERY_BATCH = 10
SYNC_MODEL_EVERY_BATCH = 1
SAVE_MODEL_EVERY_BATCH = 3000


class Player:
    def __init__(self, env, reward_steps, gamma, max_steps, player_index):
        self.env = env
        self.reward_steps = reward_steps
        self.gamma = gamma
        self.state = env.reset()

        self.memory = []
        self.episode_reward = 0.0
        self.step_index = 0
        self.max_steps = max_steps
        self.player_index = player_index

        self.done_rewards = []

    @classmethod
    def step_players(cls, model, players):
        """
        Do one step for list of players
        :param model: model to use for predictions
        :param players: player instances
        :return: list of samples
        """
        probs, values = model.predict_on_batch(np.array([
            p.state for p in players
        ]))
        result = []
        for idx, player in enumerate(players):
            action = np.random.choice(len(probs[idx]), p=probs[idx])
            result.extend(player.step(action, values[idx][0]))
        return result

    def step(self, action, value):
        result = []
        new_state, reward, done, _ = self.env.step(action)
        self.episode_reward += reward
        self.memory.append((self.state, action, reward, value))
        self.state = new_state

        if done or self.step_index > self.max_steps:
            self.state = self.env.reset()
            logging.info("%3d: Episode done @ step %d: sum reward %d",
                         self.player_index, self.step_index, int(self.episode_reward))
            self.done_rewards.append(self.episode_reward)
            self.episode_reward = 0.0
            self.step_index = 0
            result.extend(self._memory_to_samples(is_done=done))
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

    @classmethod
    def gather_done_rewards(cls, *players):
        """
        Collect rewards from list of players
        :param players: list of players
        :return: list of steps, list of rewards of done episodes
        """
        res = []
        for p in players:
            res.extend(p.done_rewards)
            p.done_rewards = []
        return res


def generate_batches(model, players, batch_size):
    samples = []

    while True:
        samples.extend(Player.step_players(model, players))
        while len(samples) >= batch_size:
            states, actions, rewards, advantages = list(map(np.array, zip(*samples[:batch_size])))
            yield [states, actions, advantages], [rewards, rewards]
            samples = samples[batch_size:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-e", "--env", default="Breakout-v0", help="Environment name to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and save data into provided dir, default=disabled")
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma for reward discount, default=0.99")
    parser.add_argument("-i", "--iters", type=int, default=10000, help="Count of iterations to take, default=100")
    parser.add_argument("--steps", type=int, default=5, help="Count of steps to use in reward estimation")
    args = parser.parse_args()

    # limit GPU memory
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    K.set_session(tf.Session(config=config))

    env = make_env(args.env, args.monitor, history_steps=HISTORY_STEPS)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    tr_input_t, tr_conv_out_t = net_input(state_shape)
    value_policy_model = make_train_model(tr_input_t, tr_conv_out_t, n_actions)

    r_input_t, r_conv_out_t = net_input(state_shape)
    run_model = make_run_model(r_input_t, r_conv_out_t, n_actions)

    value_policy_model.summary()

    loss_dict = {
        'value': 'mse',
        'policy_loss': lambda y_true, y_pred: y_pred
    }

    value_policy_model.compile(optimizer=Adam(lr=0.001, epsilon=1e-3, clipnorm=0.1), loss=loss_dict)

    # keras summary magic
    summary_writer = tf.summary.FileWriter("logs/" + args.name)
    summarize_gradients(value_policy_model)
    value_policy_model.metrics_names.append("value_summary")
    value_policy_model.metrics_tensors.append(tf.summary.merge_all())

    players = [Player(make_env(args.env, args.monitor, history_steps=HISTORY_STEPS), reward_steps=args.steps,
                      gamma=args.gamma, max_steps=40000, player_index=idx)
               for idx in range(PLAYERS_COUNT)]

    bench_samples = 0
    bench_ts = time.time()

    for iter_idx, (x_batch, y_batch) in enumerate(generate_batches(run_model, players, BATCH_SIZE)):
        l = value_policy_model.train_on_batch(x_batch, y_batch)
        bench_samples += BATCH_SIZE

        if iter_idx % SUMMARY_EVERY_BATCH == 0:
            l_dict = dict(zip(value_policy_model.metrics_names, l))
            done_rewards = Player.gather_done_rewards(*players)

            if done_rewards:
                summary_value("reward_episode_mean", np.mean(done_rewards), summary_writer, iter_idx)
                summary_value("reward_episode_max", np.max(done_rewards), summary_writer, iter_idx)

            summary_value("speed", bench_samples / (time.time() - bench_ts), summary_writer, iter_idx)
            summary_value("reward_batch", np.mean(y_batch[0]), summary_writer, iter_idx)
            summary_value("loss_value", l_dict['value_loss'], summary_writer, iter_idx)
            summary_value("loss_full", l_dict['loss'], summary_writer, iter_idx)
            summary_writer.add_summary(l_dict['value_summary'], global_step=iter_idx)
            summary_writer.flush()
            bench_samples = 0
            bench_ts = time.time()

        if iter_idx % SYNC_MODEL_EVERY_BATCH == 0:
            run_model.set_weights(value_policy_model.get_weights())
#            logger.info("Models synchronized, iter %d", iter_idx)

        if iter_idx % SAVE_MODEL_EVERY_BATCH == 0 and iter_idx > 0:
            value_policy_model.save(os.path.join("logs", args.name, "model-%06d.h5" % iter_idx))
