#!/usr/bin/env python
import gym
import universe
import os
import argparse
import logging
import numpy as np
import multiprocessing as mp
import queue

import time
import datetime
import tensorflow as tf
import keras.backend as K

from keras.optimizers import Adam

from algo_lib.common import make_env, summarize_gradients, summary_value
from algo_lib.atari_opts import net_input, HISTORY_STEPS, preprocess_state
from algo_lib.a3c import make_train_model, make_run_model
from algo_lib.player import Player

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PLAYERS_SWARMS = 4
PLAYERS_PER_SWARM = 12
BATCH_SIZE = 128

SUMMARY_EVERY_BATCH = 100
SYNC_MODEL_EVERY_BATCH = 1
SAVE_MODEL_EVERY_BATCH = 3000


class AsyncPlayersSwarm:
    def __init__(self, swarms_count, swarm_size, env_name, history_steps, gamma,
                 reward_steps, batch_size, max_steps, state_filter):
        self.batch_size = batch_size
        self.samples_queue = mp.Queue(maxsize=batch_size * 2)
        self.done_rewards_queue = mp.Queue()
        self.control_queues = []
        self.processes = []
        for _ in range(swarms_count):
            ctrl_queue = mp.Queue()
            self.control_queues.append(ctrl_queue)
            args = (swarm_size, env_name, history_steps, gamma, reward_steps, max_steps, state_filter,
                    ctrl_queue, self.samples_queue, self.done_rewards_queue)
            proc = mp.Process(target=AsyncPlayersSwarm.player, args=args)
            self.processes.append(proc)
            proc.start()

    def push_model_weights(self, weights):
        for q in self.control_queues:
            q.put(weights)

    def get_batch(self):
        batch = []
        while len(batch) < self.batch_size:
            batch.append(self.samples_queue.get())
        states, actions, rewards = list(map(np.array, zip(*batch)))
        # normalize rewards
        # rewards -= rewards.mean()
        # rewards /= (rewards.std() + K.epsilon())
        return [states, actions, rewards], [rewards, rewards, rewards]

    def get_done_rewards(self):
        res = []
        try:
            while True:
                res.append(self.done_rewards_queue.get_nowait())
        except queue.Empty:
            pass
        return res

    @classmethod
    def player(cls, players_count, env_name, history_steps, gamma, reward_steps,
               max_steps, state_filter, ctrl_queue, out_queue, done_rewards_queue):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        with tf.device("/cpu:0"):
            players = [Player(make_env(env_name, history_steps=history_steps), reward_steps,
                              gamma, max_steps, idx, state_filter)
                       for idx in range(players_count)]
            input_t, conv_out_t = net_input()
            n_actions = env.action_space.n
            model = make_run_model(input_t, conv_out_t, n_actions)
            while True:
                # check ctrl queue for new model
                if not ctrl_queue.empty():
                    weights = ctrl_queue.get()
                    # stop requested
                    if weights is None:
                        break
                    model.set_weights(weights)

                for sample in Player.step_players(model, players):
                    out_queue.put(sample)
                for rw in Player.gather_done_rewards(*players):
                    done_rewards_queue.put(rw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-e", "--env", default="Breakout-v0", help="Environment name to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and save data into provided dir, default=disabled")
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma for reward discount, default=0.99")
    parser.add_argument("--steps", type=int, default=5, help="Count of steps to use in reward estimation")
    args = parser.parse_args()

    # limit GPU memory
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    # K.set_session(tf.Session(config=config))

    env = make_env(args.env, args.monitor, history_steps=HISTORY_STEPS)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    tr_input_t, tr_conv_out_t = net_input()
    value_policy_model = make_train_model(tr_input_t, tr_conv_out_t, n_actions)

    value_policy_model.summary()

    loss_dict = {
        'policy': lambda y_true, y_pred: tf.constant(0.0),
        'value': 'mse',
        'policy_loss': lambda y_true, y_pred: y_pred
    }

    value_policy_model.compile(optimizer=Adam(lr=0.001, epsilon=1e-3, clipnorm=0.1), loss=loss_dict)

    input_t, conv_out_t = net_input()
    n_actions = env.action_space.n
    model = make_run_model(input_t, conv_out_t, n_actions)
    model.summary()

    # keras summary magic
    summary_writer = tf.summary.FileWriter("logs/" + args.name)
    summarize_gradients(value_policy_model)
    value_policy_model.metrics_names.append("value_summary")
    value_policy_model.metrics_tensors.append(tf.summary.merge_all())

    players = AsyncPlayersSwarm(PLAYERS_SWARMS, PLAYERS_PER_SWARM, args.env, HISTORY_STEPS, args.gamma, args.steps,
                                max_steps=40000, batch_size=BATCH_SIZE, state_filter=preprocess_state)
    iter_idx = 0
    bench_samples = 0
    bench_ts = time.time()

    batch_time = 0
    train_time = 0

    while True:
        iter_idx += 1
        batch_ts = time.time()
        x_batch, y_batch = players.get_batch()
        batch_time += time.time() - batch_ts
        train_ts = time.time()
        l = value_policy_model.train_on_batch(x_batch, y_batch)
        train_time += time.time() - train_ts
        bench_samples += BATCH_SIZE

        if iter_idx % SUMMARY_EVERY_BATCH == 0:
            summary_value("time_batch", batch_time / SUMMARY_EVERY_BATCH, summary_writer, iter_idx)
            summary_value("time_train", train_time / SUMMARY_EVERY_BATCH, summary_writer, iter_idx)
            batch_time = 0
            train_time = 0
            l_dict = dict(zip(value_policy_model.metrics_names, l))
            done_rewards = players.get_done_rewards()

            if done_rewards:
                summary_value("reward_episode_mean", np.mean(done_rewards), summary_writer, iter_idx)
                summary_value("reward_episode_max", np.max(done_rewards), summary_writer, iter_idx)
                summary_value("reward_episode_min", np.min(done_rewards), summary_writer, iter_idx)

            # summary_value("rewards_norm_mean", np.mean(y_batch[0]), summary_writer, iter_idx)
            summary_value("speed", bench_samples / (time.time() - bench_ts), summary_writer, iter_idx)
            summary_value("loss_value", l_dict['value_loss'], summary_writer, iter_idx)
            summary_value("loss", l_dict['loss'], summary_writer, iter_idx)
            summary_writer.add_summary(l_dict['value_summary'], global_step=iter_idx)
            summary_writer.flush()
            bench_samples = 0
            logger.info("Iter %d: speed %s per batch", iter_idx,
                        datetime.timedelta(seconds=(time.time() - bench_ts)/SUMMARY_EVERY_BATCH))
            bench_ts = time.time()


        if iter_idx % SYNC_MODEL_EVERY_BATCH == 0:
            players.push_model_weights(value_policy_model.get_weights())

        if iter_idx % SAVE_MODEL_EVERY_BATCH == 0:
            value_policy_model.save(os.path.join("logs", args.name, "model-%06d.h5" % iter_idx))
