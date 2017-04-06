#!/usr/bin/env python
# Quick-n-dirty implementation of Advantage Actor-Critic method from https://arxiv.org/abs/1602.01783
import os
import argparse
import logging
import time
import numpy as np

from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

from algo_lib.common import make_env, summarize_gradients, summary_value
from algo_lib import common
from algo_lib import atari
from algo_lib.a3c import make_run_model, make_train_model
from algo_lib.player import Player, generate_batches

HISTORY_STEPS = 4

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PLAYERS_COUNT = 50
BATCH_SIZE = 128

SUMMARY_EVERY_BATCH = 10
SYNC_MODEL_EVERY_BATCH = 1
SAVE_MODEL_EVERY_BATCH = 3000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ini", required=True, help="Ini file with configuration")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    args = parser.parse_args()

    # limit GPU memory
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    K.set_session(tf.Session(config=config))

    config = common.Configuration(args.ini)
    env_factory = atari.AtariEnvFactory(config)

    env = env_factory()
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    logger.info("Created environment %s, state: %s, actions: %s", config.env_name, state_shape, n_actions)

    tr_input_t, tr_conv_out_t = atari.net_input(env)
    value_policy_model = make_train_model(tr_input_t, tr_conv_out_t, n_actions)

    r_input_t, r_conv_out_t = atari.net_input(env)
    run_model = make_run_model(r_input_t, r_conv_out_t, n_actions)

    value_policy_model.summary()

    loss_dict = {
        'policy_loss': lambda y_true, y_pred: y_pred,
        'value_loss': lambda y_true, y_pred: y_pred,
        'entropy_loss': lambda y_true, y_pred: y_pred,
    }

    value_policy_model.compile(optimizer=Adam(lr=0.001, epsilon=1e-3, clipnorm=0.1), loss=loss_dict)

    # keras summary magic
    summary_writer = tf.summary.FileWriter("logs/" + args.name)
    summarize_gradients(value_policy_model)
    value_policy_model.metrics_names.append("value_summary")
    value_policy_model.metrics_tensors.append(tf.summary.merge_all())

    players = [Player(env_factory(), reward_steps=config.a3c_steps,
                      gamma=config.a3c_gamma, max_steps=config.max_steps, player_index=idx)
               for idx in range(PLAYERS_COUNT)]

    bench_samples = 0
    bench_ts = time.time()

    for iter_idx, x_batch in enumerate(generate_batches(run_model, players, BATCH_SIZE)):
        y_stub = np.zeros(len(x_batch[0]))
        l = value_policy_model.train_on_batch(x_batch, [y_stub]*3)
        bench_samples += BATCH_SIZE

        if iter_idx % SUMMARY_EVERY_BATCH == 0:
            l_dict = dict(zip(value_policy_model.metrics_names, l))
            done_rewards = Player.gather_done_rewards(*players)

            if done_rewards:
                summary_value("reward_episode_mean", np.mean(done_rewards), summary_writer, iter_idx)
                summary_value("reward_episode_max", np.max(done_rewards), summary_writer, iter_idx)

            summary_value("speed", bench_samples / (time.time() - bench_ts), summary_writer, iter_idx)
            summary_value("reward_batch", np.mean(x_batch[2]), summary_writer, iter_idx)
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
