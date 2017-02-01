import gym
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad


BATCH_SIZE = 128
NUM_FRAMES = 2


def build_model(num_frames, input_shape, actions):
    m = Sequential()
    m.add(Dense(100, input_shape=(num_frames,) + input_shape, activation='relu'))
    m.add(Dense(100, activation='relu'))
    m.add(Dense(output_dim=actions))
    m.add(Activation('softmax'))
    return m


def train_on_batch(model, activations_count, train_x, train_labels, rewards):
    train_y = to_categorical(train_labels, activations_count)
    return model.train_on_batch(np.array(train_x), train_y, sample_weight=-np.array(rewards))


def get_action(model, activactions_count, observation):
    if len(observation) != NUM_FRAMES:
        return random.randint(0, activactions_count-1)
    return model.predict_classes(np.array([observation]), batch_size=1, verbose=0)[0]


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    m = build_model(NUM_FRAMES, env.observation_space.shape, env.action_space.n)
    print(m.summary())

    m.compile(optimizer=Adagrad(lr=0.001), loss='mse')

    print(env.observation_space)
    print(env.action_space)

    train_x = []
    train_y = []
    rewards = []

    episodes = []
    for ep_idx in range(100000):
        obs = env.reset()
        frames = [obs]
        episode_len = 0

        while True:
            action = get_action(m, env.action_space.n, frames)
            print(action)
            frames.append(obs)
            frames = frames[-NUM_FRAMES:]
            if len(frames) == NUM_FRAMES:
                train_x.append(list(frames))
                train_y.append(action)
            obs, reward, done, info = env.step(action)
            episode_len += 1
            rewards.append(reward)

            if len(train_x) == BATCH_SIZE:
                losses = train_on_batch(m, env.action_space.n, train_x, train_y, rewards)
                print("Episode: {episode}, Loss: {loss:.4f}, max: {max}, avg: {avg:.3f}".format(
                    episode=ep_idx,
                    loss=losses,
                    max=np.max(episodes),
                    avg=np.mean(episodes),
                ))
                train_x = []
                train_y = []
                rewards = []

            if done:
                episodes.append(episode_len)
                episodes = episodes[-100:]
                break

    m.save_weights("model.hdf5")
    pass
