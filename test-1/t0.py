# Multi-layer perceptron inspired by this: https://gym.openai.com/evaluations/eval_P4KyYPwIQdSg6EqvHgYjiw
# https://gist.githubusercontent.com/anonymous/d829ec2f8bda088ac897aa2055dcd3a8/raw/d3fcdfdcc9038bf24385589e94939dcd3c198349/crossentropy_method.py
import gym
from gym import wrappers
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad


BATCH_SIZE = 500


def make_model(state_shape, actions_n):
    m = Sequential()
    m.add(Dense(40, input_shape=state_shape, activation='relu'))
    m.add(Dense(40))
    m.add(Dense(actions_n))
    m.add(Activation('softmax'))
    return m


def generate_session(env, model, n_actions, t_max=1000):
    states = []
    actions = []
    s = env.reset()
    total_reward = 0

    for t in range(t_max):
        probs = model.predict_proba(np.array([s]), verbose=0)[0]

        action = np.random.choice(n_actions, p=probs)
        new_s, reward, done, _ = env.step(action)
        states.append(s)
        actions.append(action)
        total_reward += reward
        s = new_s
        if done:
            break

    return states, actions, total_reward


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = wrappers.Monitor(env, "test-1")
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    m = make_model(state_shape, n_actions)
    m.summary()
    m.compile(optimizer=Adagrad(lr=0.001), loss='categorical_crossentropy')

    for idx in range(100):
        batch = [generate_session(env, m, n_actions) for _ in range(BATCH_SIZE)]
        b_states, b_actions, b_rewards = map(np.array, zip(*batch))

        threshold = np.percentile(b_rewards, 50)

        elite_states = b_states[b_rewards > threshold]
        elite_actions = b_actions[b_rewards > threshold]

        if len(elite_states) > 0:
            elite_states, elite_actions = map(np.concatenate, [elite_states, elite_actions])
            oh_actions = to_categorical(elite_actions, nb_classes=n_actions)
            m.fit(elite_states, oh_actions, verbose=0)
            print("%d: mean reward = %.5f\tthreshold = %.1f" % (idx, np.mean(b_rewards), threshold))
            m.save_weights("t0-iter=%03d-thr=%.2f.hdf5" % (idx, threshold))
        else:
            print("%d: no improvement" % idx)

    pass
