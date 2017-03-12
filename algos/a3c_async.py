#!/usr/bin/env python
import os
import argparse
import logging
import numpy as np
import multiprocessing as mp

import time
import tensorflow as tf
import keras.backend as K

from keras.optimizers import Adam

from algo_lib.common import make_env, summarize_gradients, summary_value
from algo_lib.atari_opts import net_input, HISTORY_STEPS
from algo_lib.a3c import make_train_model, make_run_model

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PLAYERS_COUNT = 50
BATCH_SIZE = 128

SUMMARY_EVERY_BATCH = 10
SYNC_MODEL_EVERY_BATCH = 10
SAVE_MODEL_EVERY_BATCH = 3000


# def player(model_queue, state_queue):
#     with tf.device("/cpu:0"):
#         run_model = make_model(2)
#         run_model.summary()
#
#         while True:
#             time.sleep(1)
#             r = run_model.predict_on_batch(np.array([[100, 100]]))
#             print(r)
#             if not model_queue.empty():
#                 weights = model_queue.get()
#                 if weights is None:
#                     print("Stop requested, exit")
#                     break
#                 run_model.set_weights(weights)
#                 print("New model received")
#     pass



class AsyncPlayersSwarm:
    def __init__(self, env_name, history_steps, gamma, reward_steps, players_count, batch_size, max_steps):
        self.batch_size = batch_size
        self.samples_queue = mp.Queue(maxsize=batch_size * 10)
        self.control_queues = []
        self.processes = []

        for _ in range(players_count):
            ctrl_queue = mp.Queue()
            args = (env_name, history_steps, gamma, reward_steps, max_steps, ctrl_queue, self.samples_queue)
            proc = mp.Process(target=AsyncPlayersSwarm.player, args=args)
            self.control_queues.append(ctrl_queue)
            self.processes.append(proc)
            proc.start()

    def push_model_weights(self, weights):
        for q in self.control_queues:
            q.put(weights)

    def get_batch(self):
        batch = []
        while len(batch) < self.batch_size:
            batch.append(self.samples_queue.get())
        states, actions, rewards, advantages = list(map(np.array, zip(*batch)))
        return [states, actions, advantages], [rewards, rewards]

    @classmethod
    def player(cls, env_name, history_steps, gamma, reword_steps, max_steps, ctrl_queue, out_queue):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        with tf.device("/cpu:0"):
            env = make_env(env_name, history_steps=history_steps)
            input_t, conv_out_t = net_input(env.observation_space.shape)
            n_actions = env.action_space.n
            model = make_run_model(input_t, conv_out_t, n_actions)
            state = env.reset()
            memory = []
            step_index = 0

            while True:
                # check ctrl queue for new model
                if not ctrl_queue.empty():
                    weights = ctrl_queue.get()
                    # stop requested
                    if weights is None:
                        break
                    model.set_weights(weights)
#                    logging.info("Models updated")

                # choose action
                probs, value = model.predict_on_batch(np.array([state]))
                probs, value = probs[0], value[0][0]
                action = np.random.choice(n_actions, p=probs)

                # do action
                new_state, reward, done, _ = env.step(action)
                memory.append((state, action, reward, value))
                state = new_state
                step_index += 1

                if done or step_index > max_steps:
                    state = env.reset()
                    step_index = 0
                    AsyncPlayersSwarm.memory_to_samples(out_queue, memory, gamma=gamma, sum_r=0.0)
                    memory = []
                elif len(memory) == reword_steps + 1:
                    last_item = memory.pop()
                    AsyncPlayersSwarm.memory_to_samples(out_queue, memory, gamma=gamma, sum_r=last_item[-1])
                    memory = [last_item]

    @classmethod
    def memory_to_samples(cls, out_queue, memory, gamma, sum_r):
        for state, action, reward, value in reversed(memory):
            sum_r = reward + sum_r * gamma
            advantage = sum_r - value
            out_queue.put((state, action, sum_r, advantage))


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

    tr_input_t, tr_conv_out_t = net_input(state_shape)
    value_policy_model = make_train_model(tr_input_t, tr_conv_out_t, n_actions)

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

    players = AsyncPlayersSwarm(args.env, HISTORY_STEPS, args.gamma, args.steps,
                                players_count=PLAYERS_COUNT, max_steps=40000, batch_size=BATCH_SIZE)
    iter_idx = 0
    bench_samples = 0
    bench_ts = time.time()

    while True:
        iter_idx += 1
        x_batch, y_batch = players.get_batch()
        l = value_policy_model.train_on_batch(x_batch, y_batch)
        bench_samples += BATCH_SIZE

        if iter_idx % SUMMARY_EVERY_BATCH == 0:
            l_dict = dict(zip(value_policy_model.metrics_names, l))
            # done_rewards = Player.gather_done_rewards(*players)
            #
            # if done_rewards:
            #     summary_value("reward_episode_mean", np.mean(done_rewards), summary_writer, iter_idx)
            #     summary_value("reward_episode_max", np.max(done_rewards), summary_writer, iter_idx)

            summary_value("speed", bench_samples / (time.time() - bench_ts), summary_writer, iter_idx)
            summary_value("reward_batch", np.mean(y_batch[0]), summary_writer, iter_idx)
            summary_value("loss_value", l_dict['value_loss'], summary_writer, iter_idx)
            summary_value("loss_full", l_dict['loss'], summary_writer, iter_idx)
            summary_writer.add_summary(l_dict['value_summary'], global_step=iter_idx)
            summary_writer.flush()
            bench_samples = 0
            bench_ts = time.time()

        if iter_idx % SYNC_MODEL_EVERY_BATCH == 0:
            players.push_model_weights(value_policy_model.get_weights())
            logger.info("Models synchronized @ iter %d", iter_idx)

        if iter_idx % SAVE_MODEL_EVERY_BATCH == 0:
            value_policy_model.save(os.path.join("logs", args.name, "model-%06d.h5" % iter_idx))
