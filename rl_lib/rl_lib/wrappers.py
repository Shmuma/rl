import gym
import numpy as np

def HistoryWrapper(steps):
    class HistoryWrapper(gym.Wrapper):
        """
        Track history of observations for given amount of steps
        Initial steps are zero-filled
        """
        def __init__(self, env):
            super(HistoryWrapper, self).__init__(env)
            self.steps = steps
            self.history = self._make_history()

        def _make_history(self):
            return [np.zeros(shape=self.env.observation_space.shape) for _ in range(steps)]

        def _step(self, action):
            obs, reward, done, info = self.env.step(action)
            self.history.pop(0)
            self.history.append(obs)
            return np.array(self.history), reward, done, info

        def _reset(self):
            self.history = self._make_history()
            self.history.pop(0)
            self.history.append(self.env.reset())
            return np.array(self.history)

    return HistoryWrapper
