#!/usr/bin/env python
import os
import argparse
import logging
import numpy as np

import time
import datetime
import tensorflow as tf
import multiprocessing as mp

from keras.optimizers import Adam

from algo_lib import common
from algo_lib import atari
from algo_lib import player

from algo_lib.a3c import make_train_model, make_run_model

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SUMMARY_EVERY_BATCH = 100
SYNC_MODEL_EVERY_BATCH = 1
SAVE_MODEL_EVERY_BATCH = 3000


if __name__ == "__main__":
    # work-around for TF multiprocessing problems
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--read", help="Model file name to read")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-i", "--ini", required=True, help="Ini file with configuration")
    args = parser.parse_args()

    config = common.Configuration(args.ini)

    env_factory = atari.AtariEnvFactory(config)

    env = env_factory()
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    logger.info("Created environment %s, state: %s, actions: %s", config.env_name, state_shape, n_actions)

    input_t, conv_out_t = atari.net_input(env)
    value_policy_model = make_train_model(input_t, conv_out_t, n_actions, entropy_beta=config.a3c_beta)
    value_policy_model.summary()
    run_model = make_run_model(input_t, conv_out_t, n_actions)
    run_model.summary()

    loss_dict = {
        'policy_loss': lambda y_true, y_pred: y_pred,
        'value_loss': lambda y_true, y_pred: y_pred,
        'entropy_loss': lambda y_true, y_pred: y_pred,
    }

    optimizer = Adam(lr=config.learning_rate, epsilon=1e-3, clipnorm=config.gradient_clip_norm)
    value_policy_model.compile(optimizer=optimizer, loss=loss_dict)

    # keras summary magic
    summary_writer = tf.summary.FileWriter("logs/" + args.name)
    common.summarize_gradients(value_policy_model)
    value_policy_model.metrics_names.append("value_summary")
    value_policy_model.metrics_tensors.append(tf.summary.merge_all())

    if args.read:
        logger.info("Loading model from %s", args.read)
        value_policy_model.load_weights(args.read)

    tweaker = common.ParamsTweaker()
    tweaker.add("lr", optimizer.lr)

    players = player.AsyncPlayersSwarm(config, env_factory, run_model)
    iter_idx = 0
    bench_samples = 0
    bench_ts = time.time()

    while True:
        if iter_idx % SYNC_MODEL_EVERY_BATCH == 0:
            players.push_model_weights(value_policy_model.get_weights())

        iter_idx += 1
        batch_ts = time.time()
        x_batch = players.get_batch()
        # stub for y
        y_stub = np.zeros(len(x_batch[0]))

        l = value_policy_model.train_on_batch(x_batch, [y_stub]*3)
        bench_samples += config.batch_size

        if iter_idx % SUMMARY_EVERY_BATCH == 0:
            l_dict = dict(zip(value_policy_model.metrics_names, l))
            done_rewards = players.get_done_rewards()

            if done_rewards:
                common.summary_value("reward_episode_mean", np.mean(done_rewards), summary_writer, iter_idx)
                common.summary_value("reward_episode_max", np.max(done_rewards), summary_writer, iter_idx)
                common.summary_value("reward_episode_min", np.min(done_rewards), summary_writer, iter_idx)

            # summary_value("rewards_norm_mean", np.mean(y_batch[0]), summary_writer, iter_idx)
            common.summary_value("speed", bench_samples / (time.time() - bench_ts), summary_writer, iter_idx)
            common.summary_value("loss", l_dict['loss'], summary_writer, iter_idx)
            summary_writer.add_summary(l_dict['value_summary'], global_step=iter_idx)
            summary_writer.flush()
            bench_samples = 0
            logger.info("Iter %d: speed %s per batch", iter_idx,
                        datetime.timedelta(seconds=(time.time() - bench_ts)/SUMMARY_EVERY_BATCH))
            bench_ts = time.time()

        if iter_idx % SAVE_MODEL_EVERY_BATCH == 0:
            value_policy_model.save(os.path.join("logs", args.name, "model-%06d.h5" % iter_idx))

        tweaker.check()
