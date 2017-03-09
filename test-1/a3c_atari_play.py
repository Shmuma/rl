#!/usr/bin/env python
import argparse
import numpy as np

from a3c_atari import make_env, make_model, preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--model", help="Model file to read. If not specified, will play randomly")
    parser.add_argument("-e", "--env", required=True, help="Environment to use")
    parser.add_argument("-m", "--monitor", help="Enable monitor and write to directory, default=disabled")
    parser.add_argument("--iters", type=int, default=100, help="Episodes to play, default=100")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Show individual episode results")
    args = parser.parse_args()

    env = make_env(args.env, args.monitor)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    if args.model is not None:
        model = make_model(n_actions, train_mode=False)
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
                    np.array([preprocess(state)]),
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
