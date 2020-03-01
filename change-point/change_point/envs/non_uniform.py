import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Discrete
from gym.utils import seeding


class NonUniformPrior(gym.Env):
    def __init__(self, Ts, Tt, N, stop_error, dist=np.random.sample, seed=None):
        """
        :param stop_error:
        :param seed:
        :param dist: Distribution from which change point is to be drawn from. Defaults to uniform distribution over [0, 1).
        The distribution should have a min and a max of 1, although this is left to user to verify.
        """
        self.seed(seed)
        self._save_args(Ts, Tt, N, stop_error, seed, dist)
        self.reset()

    def _cost(self, action):
        return self.Ts + self.Tt * action

    def _reward(self, action):
        return -1 * self._cost(action)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _save_args(self, Ts, Tt, N, stop_error, seed, dist):
        self.Ts = Ts
        self.Tt = Tt
        N = int(N)
        self.N = N
        self.delta = 1
        self.stop_error = stop_error
        self.seed = seed
        self.dist = dist

    def reset(self):
        self.total_dist = 0
        self.location = 0
        self.min_loc = 0
        self.max_loc = self.N
        self.location_hist = []
        self.direction = 1
        self.change_point = self.dist(1)[0] * self.N
        self.observation_space = Discrete(self.N + 1)
        self.action_space = Discrete(self.N + 1)
        self.S = self.N
        return self.S

    def _move_agent(self, action: float):
        self.total_dist += action
        # find movement
        mvmt = action * self.direction

        self.location_hist += [self.location]
        self.location += mvmt

    def _get_info(self):
        info = {
            'change_point': self.change_point,
            'S': self.S,
            'total_dist': self.total_dist,
            'location': self.location,
            'location_hist': self.location_hist
        }
        return info

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._move_agent(action)

        # See if change point is to left or right
        relative_location = self.location < self.change_point

        # If change point is to the right of location, make direction positive, else negative
        # Also update where agent is searching
        if relative_location == 1:
            self.direction = 1
            self.min_loc = self.location
        else:
            self.direction = -1
            self.max_loc = self.location

        self.S = self.max_loc - self.min_loc

        if np.abs(self.location - self.change_point) < self.stop_error:
            done = True
            reward = 100
        else:
            done = False
            reward = self._reward(action)

            # Update action space
            self.action_space = Discrete(n=self.S / self.delta)

        return self.S, reward, done, self._get_info()

    def render(self, mode='human'):
        linex = np.linspace(0,self.N, 101)
        liney = np.zeros(linex.shape)
        hypespacex = np.linspace(self.min_loc, self.max_loc, 101)
        hypespacey = np.zeros(hypespacex.shape)
        fig, ax = plt.subplots(figsize=(5, 1))
        ax.plot(linex,liney,c='black')
        ax.plot(hypespacex, hypespacey, c='orange')
        ax.scatter(self.change_point, 0, c='red',marker='x', zorder=1000)
        ax.scatter(self.location, 0, c='m', zorder=1000)
        ax.get_yaxis().set_visible(False)
        fig.show()
        time.sleep(1)

    def close(self):
        pass
