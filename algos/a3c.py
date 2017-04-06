#!/usr/bin/env python
# Quick-n-dirty implementation of Advantage Actor-Critic method from https://arxiv.org/abs/1602.01783
import uuid
import os
import argparse
import logging
import numpy as np
import pickle

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import tensorflow as tf
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam

from algo_lib import common# import make_env, summarize_gradients, summary_value, HistoryWrapper
from algo_lib import a3c
from algo_lib.player import Player, generate_batches

HISTORY_STEPS = 4
SIMPLE_L1_SIZE = 50
SIMPLE_L2_SIZE = 50

SUMMARY_EVERY_BATCH = 100
SAVE_MODEL_EVERY_BATCH = 3000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default="CartPole-v0", help="Environment name to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and save data into provided dir, default=disabled")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma for reward discount, default=1.0")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    args = parser.parse_args()

    env_wrappers = (common.HistoryWrapper(HISTORY_STEPS),)
    env = common.make_env(args.env, args.monitor, wrappers=env_wrappers)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    in_t = Input(shape=state_shape, name='input')
    fl_t = Flatten(name='flat')(in_t)
    l1_t = Dense(SIMPLE_L1_SIZE, activation='relu', name='in_l1')(fl_t)
    out_t = Dense(SIMPLE_L2_SIZE, activation='relu', name='in_l2')(l1_t)

    run_model = a3c.make_run_model(in_t, out_t, n_actions)
    value_policy_model = a3c.make_train_model(in_t, out_t, n_actions, entropy_beta=0.01)
    value_policy_model.summary()

    loss_dict = {
        'policy_loss': lambda y_true, y_pred: y_pred,
        'value_loss': lambda y_true, y_pred: y_pred,
        'entropy_loss': lambda y_true, y_pred: y_pred,
    }
    value_policy_model.compile(optimizer=Adam(lr=0.0005, clipnorm=0.1), loss=loss_dict)

    # keras summary magic
    summary_writer = tf.summary.FileWriter("logs-a3c/" + args.name)
    common.summarize_gradients(value_policy_model)
    value_policy_model.metrics_names.append("value_summary")
    value_policy_model.metrics_tensors.append(tf.summary.merge_all())

    if args.env.startswith("MountainCar"):
        reward_hook = lambda reward, done, step: int(done)*10.0
    else:
        reward_hook = None

    players = [
        Player(common.make_env(args.env, args.monitor, wrappers=env_wrappers), reward_steps=20, gamma=0.999,
               max_steps=40000, player_index=idx, reward_hook=reward_hook)
        for idx in range(10)
    ]

    for iter_idx, x_batch in enumerate(generate_batches(run_model, players, 128)):
        y_stub = np.zeros(len(x_batch[0]))
        pre_weights = value_policy_model.get_weights()
        l = value_policy_model.train_on_batch(x_batch, [y_stub]*3)
        post_weights = value_policy_model.get_weights()

        # logger.info("Iteration %d, loss: %s", iter_idx, l[:-1])
        if np.isnan(l[:-1]).any():
            break

        if iter_idx % SUMMARY_EVERY_BATCH == 0:
            l_dict = dict(zip(value_policy_model.metrics_names, l))
            done_rewards = Player.gather_done_rewards(*players)

            if done_rewards:
                common.summary_value("reward_episode_mean", np.mean(done_rewards), summary_writer, iter_idx)
                common.summary_value("reward_episode_max", np.max(done_rewards), summary_writer, iter_idx)
                common.summary_value("reward_episode_min", np.min(done_rewards), summary_writer, iter_idx)

                common.summary_value("reward_batch", np.mean(x_batch[2]), summary_writer, iter_idx)
                common.summary_value("loss", l_dict['loss'], summary_writer, iter_idx)
            summary_writer.add_summary(l_dict['value_summary'], global_step=iter_idx)
            summary_writer.flush()

        if iter_idx % SAVE_MODEL_EVERY_BATCH == 0:
            value_policy_model.save(os.path.join("logs-a3c", args.name, "model-%06d.h5" % iter_idx))

        if iter_idx % 20 == 0:
            run_model.set_weights(value_policy_model.get_weights())
    pass
