#!/usr/bin/env python
import os
import argparse
import logging
import numpy as np
import multiprocessing as mp
import queue

import time
import datetime
import tensorflow as tf

from keras.optimizers import Adam

from algo_lib import common
from algo_lib import atari_opts

from algo_lib.common import summarize_gradients, summary_value, ParamsTweaker
from algo_lib.atari_opts import net_input, RescaleWrapper, HISTORY_STEPS
from algo_lib.a3c import make_train_model, make_run_model
from algo_lib.player import Player

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# PLAYERS_SWARMS = 3
# PLAYERS_PER_SWARM = 16
# BATCH_SIZE = 128

SUMMARY_EVERY_BATCH = 100
SYNC_MODEL_EVERY_BATCH = 1
SAVE_MODEL_EVERY_BATCH = 3000


class AsyncPlayersSwarm:
    def __init__(self, config, env_factory):
        self.config = config
        self.batch_size = config.batch_size
        self.samples_queue = mp.Queue(maxsize=self.batch_size * 10)
        self.done_rewards_queue = mp.Queue()
        self.control_queues = []
        self.processes = []
        for _ in range(config.swarms_count):
            ctrl_queue = mp.Queue()
            self.control_queues.append(ctrl_queue)
            args = (config, env_factory, ctrl_queue, self.samples_queue, self.done_rewards_queue)
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
        return [states, actions, rewards], [rewards, rewards]

    def get_done_rewards(self):
        res = []
        try:
            while True:
                res.append(self.done_rewards_queue.get_nowait())
        except queue.Empty:
            pass
        return res

    @classmethod
    def player(cls, config, env_factory, ctrl_queue, out_queue, done_rewards_queue):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        with tf.device("/cpu:0"):
            players = [Player(env_factory(), config.a3c_steps, config.a3c_gamma, config.max_steps, idx)
                       for idx in range(config.swarm_size)]
            input_t, conv_out_t = net_input()
            n_actions = players[0].env.action_space.n
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
    # work-around for TF multiprocessing problems
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--read", help="Model file name to read")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-i", "--ini", required=True, help="Ini file with configuration")
    args = parser.parse_args()

    config = common.Configuration(args.ini)

    env_factory = common.EnvFactory(config, atari_opts.RescaleWrapper(config))

    env = env_factory()
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    logger.info("Created environment %s, state: %s, actions: %s", config.env_name, state_shape, n_actions)

    tr_input_t, tr_conv_out_t = net_input(env)
    value_policy_model = make_train_model(tr_input_t, tr_conv_out_t, n_actions, entropy_beta=config.a3c_beta)

    value_policy_model.summary()

    loss_dict = {
        'policy_loss': lambda y_true, y_pred: y_pred,
        'value': 'mse',
    }

    optimizer = Adam(lr=0.001, epsilon=1e-3, clipnorm=0.1)
    value_policy_model.compile(optimizer=optimizer, loss=loss_dict)

    input_t, conv_out_t = net_input(env)

    # keras summary magic
    summary_writer = tf.summary.FileWriter("logs/" + args.name)
    summarize_gradients(value_policy_model)
    value_policy_model.metrics_names.append("value_summary")
    value_policy_model.metrics_tensors.append(tf.summary.merge_all())

    if args.read:
        logger.info("Loading model from %s", args.read)
        value_policy_model.load_weights(args.read)

    tweaker = ParamsTweaker()
    tweaker.add("lr", optimizer.lr)
#    tweaker.add("beta", entropy_beta_t)

    players = AsyncPlayersSwarm(config, env_factory)
    players.push_model_weights(value_policy_model.get_weights())
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
        bench_samples += config.batch_size

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

        tweaker.check()
