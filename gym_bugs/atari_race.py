import sys
import gym

from rl_lib.wrappers import HistoryWrapper

ENV_NAME = "Breakout-v0"
ENV_COUNT = 50

if __name__ == "__main__":
    envs = {}
    HistoryWrapper(4)(gym.make(ENV_NAME))

    for idx in range(ENV_COUNT):
        e = HistoryWrapper(4)(gym.make(ENV_NAME))
        envs[idx] = {
            'done': False,
            'steps': 0,
            'reward': 0.0,
            'state': e.reset(),
            'env': e
        }

    # play randomly
    while not all(map(lambda e: e['done'], envs.values())):
        for idx, e in envs.items():
            if e['done']:
                continue
            e['steps'] += 1
            e['state'], r, done, _ = e['env'].step(e['env'].action_space.sample())
            e['reward'] += r
            e['done'] = done
            if done:
                print("Env %d done after %d steps with reward %s" % (idx, e['steps'], e['reward']))
                if e['steps'] < 100:
                    sys.exit(0)
                e['steps'] = 0
                e['reward'] = 0.0
                e['done'] = False
                e['state'] = e['env'].reset()

    pass
