import os
import logging
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import queue
from keras.models import model_from_json


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Player:
    """
    Simple syncronous pool of players
    """
    def __init__(self, env, reward_steps, gamma, max_steps, player_index, reward_hook=None):
        self.env = env
        self.reward_steps = reward_steps
        self.gamma = gamma
        self.reward_hook = reward_hook

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
        input = np.array([
            p.state for p in players
        ])
        probs, values = model.predict_on_batch(input)
        result = []

        for idx, player in enumerate(players):
            prb = softmax(probs[idx])
            action = np.random.choice(len(prb), p=prb)
            result.extend(player.step(action, values[idx][0]))
        return result

    def step(self, action, value):
        result = []
        new_state, reward, done, _ = self.env.step(action)
        self.episode_reward += reward
        if self.reward_hook is not None:
            reward = self.reward_hook(reward=reward, done=done, step=self.step_index)
        self.memory.append((self.state, action, reward, value))
        self.state = new_state
        self.step_index += 1

        if done or self.step_index > self.max_steps:
            self.state = self.env.reset()
            logging.info("%3d: Episode done @ step %5d, sum reward %d",
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
        sum_r, last_item = 0.0, None

        if not is_done:
            last_item = self.memory.pop()
            sum_r = last_item[-1]

        for state, action, reward, value in reversed(self.memory):
            sum_r = reward + sum_r * self.gamma
            result.append((state, action, sum_r))

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
            states, actions, rewards = list(map(np.array, zip(*samples[:batch_size])))
            # rewards -= rewards.mean()
            # rewards /= rewards.std()
            yield [states, actions, rewards], [rewards, rewards]
            samples = samples[batch_size:]


class AsyncPlayersSwarm:
    def __init__(self, config, env_factory, model):
        self.config = config
        self.batch_size = config.batch_size
        self.samples_queue = mp.Queue(maxsize=self.batch_size * 10)
        self.done_rewards_queue = mp.Queue()
        self.control_queues = []
        self.processes = []
        for _ in range(config.swarms_count):
            ctrl_queue = mp.Queue()
            self.control_queues.append(ctrl_queue)
            args = (config, env_factory, model.to_json(), ctrl_queue, self.samples_queue, self.done_rewards_queue)
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
    def player(cls, config, env_factory, model_json, ctrl_queue, out_queue, done_rewards_queue):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        with tf.device("/cpu:0"):
            model = model_from_json(model_json)
            players = [Player(env_factory(), config.a3c_steps, config.a3c_gamma, config.max_steps, idx)
                       for idx in range(config.swarm_size)]
            # input_t, conv_out_t = atari.net_input(players[0].env)
            # n_actions = players[0].env.action_space.n
            # model = make_run_model(input_t, conv_out_t, n_actions)
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
