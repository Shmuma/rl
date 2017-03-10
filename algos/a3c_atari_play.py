#!/usr/bin/env python
import argparse
import numpy as np

from keras.models import Model

from a3c_atari import make_model, preprocess

from algo_lib.common import make_env
from algo_lib.atari_opts import HISTORY_STEPS, net_input
from algo_lib.a3c import net_prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--model", help="Model file to read. If not specified, will play randomly")
    parser.add_argument("-e", "--env", required=True, help="Environment to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and write to directory, default=disabled")
    parser.add_argument("--iters", type=int, default=100, help="Episodes to play, default=100")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Show individual episode results")
    args = parser.parse_args()

    env = make_env(args.env, args.monitor, history_steps=HISTORY_STEPS)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    if args.model is not None:
        input_t, conv_out_t = net_input(state_shape)
        out_policy_t, out_value_t = net_prediction(conv_out_t, n_actions)
        model = Model(input=input_t, output=[out_policy_t, out_value_t])
        model.summary()
        model.load_weights(args.model)
    else:
        model = None

    rewards = []
    steps = []

    for iter in range(args.iters):
        state = env.reset()
        sum_reward = 0.0
        step = 0
        while True:
            if model is None:
                action = env.action_space.sample()
            else:
                probs, value = model.predict_on_batch([
                    np.array([state]),
                ])
                probs, value = probs[0], value[0][0]
                # take action
                action = np.random.choice(len(probs), p=probs)
            state, reward, done, _ = env.step(action)
            step += 1
            sum_reward += reward
            if done:
                if args.verbose:
                    print("Episode %d done in %d steps, reward %f" % (iter, step, sum_reward))
                break
        rewards.append(sum_reward)
        steps.append(step)
    print("Done %d episodes, mean reward %.3f, mean steps %.2f" % (args.iters, np.mean(rewards), np.mean(steps)))
    pass
