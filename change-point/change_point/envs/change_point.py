import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import seeding
from abc import ABC, abstractmethod

# This is the base class for the change point problems

class ChangePoint(gym.Env, ABC):
    def __init__(self, sample_cost, movement_cost, delta, dist=None, seed=0, epsilon = None):
        """
        :param stop_error:
        :param seed:
        """
        self.seed(seed)
        self._set_args(sample_cost, movement_cost, delta, epsilon)
        self._initialize_distribution(dist)
        self._initialize_state()
        self.reset()

    @abstractmethod
    def _initialize_distribution(self, dist):
        pass

    @abstractmethod
    def _initialize_state(self):
        pass

    @abstractmethod
    def _set_action_space(self):
        pass

    @abstractmethod
    def _update_state(self):
        pass

    @abstractmethod
    def _discrete_state(self):
        pass

    @abstractmethod
    def _move_agent(self, action):
        pass

    def _dead_battery(self):
        return False

    def _cost(self, action):
        return self.sample_cost + self.movement_cost * action

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def _set_args(self, sample_cost, movement_cost, delta, epsilon):
        self.sample_cost = sample_cost
        self.movement_cost = movement_cost
        self.delta = delta
        if epsilon is None:
            epsilon = delta
        self.epsilon = epsilon
        self.N = round(1 / delta)
        self._set_action_space()

    def reset(self):
        self.total_dist = 0
        self.location = self.min_loc = 0
        self.opposite_bound = self.max_loc = 1
        self.location_hist = []
        self.direction = 1
        self.change_point = self.dist.rvs(1)[0]
        self._update_state()
        # This returns the state rescaled from 0 to 1
        return self._discrete_state()

    def _update_hist(self, dist):
        self.total_dist += dist
        self.location_hist += [self.location]

    def _get_info(self):
        info = {
            'change_point': self.change_point,
            'S': self.S,
            'total_dist': self.total_dist,
            'location': self.location,
            'location_hist': self.location_hist
        }
        return info

    def step(self, action):
        assert self.action_space.contains(action)
        cost_params = self._move_agent(action)

        # If change point is to the right of location, make direction positive, otherwise negative
        # Also update where agent is searching
        if self.location < self.change_point:
            self.direction = 1
            self.min_loc = self.location
            self.opposite_bound = self.max_loc
        else:
            self.direction = -1
            self.max_loc = self.location
            self.opposite_bound = self.min_loc

        self._update_state()

        # Test whether they're equal using isclose, to save from fpn errors
        if self.h_space_len < self.epsilon or np.isclose(self.h_space_len, self.epsilon) or self._dead_battery():
            done = True
        else:
            done = False

        reward = -1 * self._cost(cost_params)
        return self._discrete_state(), reward, done, self._get_info()

    def render(self, mode='human'):
        linex = np.linspace(0,1, 101)
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
        if mode == 'human':
            time.sleep(0.1)

    def close(self):
        plt.clf()
        plt.cla()
