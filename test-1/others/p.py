"""
idea(and code) from Karpathy's PG Pong

Karpathy's PG Pong code : https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

Karpathy's PG blog post : http://karpathy.github.io/2016/05/31/rl/

https://gist.github.com/zzing0907/de3665f9f7bbe9329b283da90d72049e#file-cartpole_pg-py
"""
import numpy as np
import pickle
import gym, gym.wrappers

H = 10
learning_rate = 2e-3
gamma = 0.99
decay_rate = 0.99
score_queue_size = 100
resume = False
D = 3

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prepro(I):
    return I[1:]


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h = sigmoid(h)
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h


def policy_backward(eph, epdlogp, epx):
    global grad_buffer
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    eph_dot = eph * (1 - eph)
    dW1 = dh * eph_dot
    dW1 = np.dot(dW1.T, epx)

    for k in model: grad_buffer[k] += {'W1': dW1, 'W2': dW2}[k]


env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, "res-1")
#env.monitor.start('CartPole', force=True)
observation = env.reset()
reward_sum, episode_num = 0, 0
xs, hs, dlogps, drs = [], [], [], []
score_queue = []

while True:

    x = prepro(observation)

    act_prob, h = policy_forward(x)

    if np.mean(score_queue) > 180:
        action = 1 if 0.5 < act_prob else 0
    else:
        action = 1 if np.random.uniform() < act_prob else 0

    xs.append(x)
    hs.append(h)
    y = action
    dlogps.append(y - act_prob)

    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)

    if done:
        episode_num += 1

        if episode_num > score_queue_size:
            score_queue.append(reward_sum)
            score_queue.pop(0)
        else:
            score_queue.append(reward_sum)

        print("episode : " + str(episode_num) + ", reward : " + str(reward_sum) + ", reward_mean : " + str(
            np.mean(score_queue)))

        if np.mean(score_queue) >= 200:
            print("CartPole solved!!!!!")
            break

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr

        policy_backward(eph, epdlogp, epx)
        for k, v in model.items():
            g = grad_buffer[k]
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
            model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v)

        if episode_num % 1000 == 0: pickle.dump(model, open('Cart.p', 'wb'))

        reward_sum = 0
        observation = env.reset()

#env.monitor.close()
