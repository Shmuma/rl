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
from keras.optimizers import Adam, Adagrad, RMSprop
from keras.objectives import mean_squared_error
import keras.backend as K

from algo_lib.common import make_env, summarize_gradients, summary_value, HistoryWrapper
from algo_lib.a3c import make_models
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

    env_wrappers = (HistoryWrapper(HISTORY_STEPS),)
    env = make_env(args.env, args.monitor, wrappers=env_wrappers)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    logger.info("Created environment %s, state: %s, actions: %s", args.env, state_shape, n_actions)

    in_t = Input(shape=(HISTORY_STEPS,) + state_shape, name='input')
    fl_t = Flatten(name='flat')(in_t)
    l1_t = Dense(SIMPLE_L1_SIZE, activation='relu', name='in_l1')(fl_t)
    out_t = Dense(SIMPLE_L2_SIZE, activation='relu', name='in_l2')(l1_t)

    run_model, value_policy_model = make_models(in_t, out_t, n_actions, entropy_beta=0.1)
    value_policy_model.summary()

    loss_dict = {
        'value': lambda y_true, y_pred: K.sqrt(mean_squared_error(y_true, y_pred)),
        'policy_loss': lambda y_true, y_pred: y_pred
    }
    # Adam(lr=0.001, epsilon=1e-3, clipnorm=0.1)
    #Adam(lr=0.0001, clipnorm=0.1)
    value_policy_model.compile(optimizer=Adam(lr=0.0001, clipnorm=0.1, clipvalue=0.1), loss=loss_dict)

    # keras summary magic
    summary_writer = tf.summary.FileWriter("logs-a3c/" + args.name)
    summarize_gradients(value_policy_model)
    value_policy_model.metrics_names.append("value_summary")
    value_policy_model.metrics_tensors.append(tf.summary.merge_all())

    players = [
        Player(make_env(args.env, args.monitor, wrappers=env_wrappers), reward_steps=19, gamma=0.99,
               max_steps=500, player_index=idx)
        for idx in range(10)
    ]

    for iter_idx, (x_batch, y_batch) in enumerate(generate_batches(run_model, players, 128)):
        pre_weights = value_policy_model.get_weights()
        l = value_policy_model.train_on_batch(x_batch, y_batch)
        post_weights = value_policy_model.get_weights()

        if any(map(lambda w: np.isnan(w).any(), post_weights)):
            logging.info("NaN in weights, iter %d, loss %s", iter_idx, l)
            name = str(uuid.uuid4())
            value_policy_model.save(name + "-nan-model.h5")
            value_policy_model.set_weights(pre_weights)
            value_policy_model.save(name + "-pre-model.h5")
            with open(name + "-out-x.dat", "wb") as fd:
                pickle.dump(x_batch, fd)
            with open(name + "-out-y.dat", "wb") as fd:
                pickle.dump(y_batch, fd)
            break

        logger.info("Iteration %d, loss: %s", iter_idx, l[:-1])
        if np.isnan(l[:-1]).any():
            break

        if iter_idx % SUMMARY_EVERY_BATCH == 0:
            l_dict = dict(zip(value_policy_model.metrics_names, l))
            done_rewards = Player.gather_done_rewards(*players)

            if done_rewards:
                summary_value("reward_episode_mean", np.mean(done_rewards), summary_writer, iter_idx)
                summary_value("reward_episode_max", np.max(done_rewards), summary_writer, iter_idx)
                summary_value("reward_episode_min", np.min(done_rewards), summary_writer, iter_idx)

            summary_value("reward_batch", np.mean(y_batch[0]), summary_writer, iter_idx)
            summary_value("loss_value", l_dict['value_loss'], summary_writer, iter_idx)
            summary_value("loss", l_dict['loss'], summary_writer, iter_idx)
            summary_writer.add_summary(l_dict['value_summary'], global_step=iter_idx)
            summary_writer.flush()

        if iter_idx % SAVE_MODEL_EVERY_BATCH == 0:
            value_policy_model.save(os.path.join("logs-a3c", args.name, "model-%06d.h5" % iter_idx))

        run_model.set_weights(value_policy_model.get_weights())
        #     name = str(uuid.uuid4())
        #     value_policy_model.save(name + "-out-model.h5")
        #     with open(name + "-out-x.dat", "wb") as fd:
        #         pickle.dump(x_batch, fd)
        #     with open(name + "-out-y.dat", "wb") as fd:
        #         pickle.dump(y_batch, fd)
        #     break
    pass
