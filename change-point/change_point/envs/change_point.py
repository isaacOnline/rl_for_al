import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Discrete
from gym.utils import seeding
from abc import ABC, abstractmethod
import tensorflow as tf

# This is the base class for the change point problems
# The only difference between the two change point environments right now
# is the distribution the change point is drawn from


class ChangePoint(gym.Env, ABC):
    def __init__(self, sample_cost, movement_cost, N, dist=None, seed=None, tf=False):
        """
        :param stop_error:
        :param seed:
        """
        self.tf = tf
        self.seed(seed)
        self._set_args(sample_cost, movement_cost, N, seed)
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
    def _update_state(self):
        pass

    def _cost(self, action):
        return self.sample_cost + self.movement_cost * action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_args(self, sample_cost, movement_cost, N, seed):
        self.sample_cost = sample_cost
        self.movement_cost = movement_cost / N
        N = int(N)
        self.N = N
        self.delta = 1
        self.seed = seed
        self.action_space = Discrete(self.N)

    def reset(self):
        self.total_dist = 0
        self.location = 0
        self.min_loc = 0
        self.max_loc = self.N
        self.location_hist = []
        self.direction = 1
        if self.tf:
            self.change_point = 0
        else:
            self.change_point = self.dist.rvs(1)[0]
        self._update_state()
        return self.S

    def get_movement(self, S, N, action):
        """
        Converts an action into the actual distance that the agent will travel. Currently,
        actions correspond to fractions of the hypothesis space that the agent will travel,
        which are then rounded to the nearest int. So, for example, if N is 1000 and the
        hypothesis space is of length 600, an agent might make an action of 829. This would
        mean the agent would travel 829/1000 * 600 = 497.4, rounded to the nearest int,
        so 497.

        :param action:
        :return:
        """
        if len(S.shape) >= 1:
            S = abs(S[0] - S[1])
        raw_travel_dist = action/N * S
        rounded_travel_dist = np.round(raw_travel_dist)

        if rounded_travel_dist == 0:
            rounded_travel_dist +=1
        if rounded_travel_dist == S:
            rounded_travel_dist -= 1

        mvmt = rounded_travel_dist * self.direction
        return rounded_travel_dist, mvmt

    def _update_hist(self, dist):
        self.total_dist += dist
        self.location_hist += [self.location]

    def _move_agent(self, action: float):
        dist, mvmt = self.get_movement(self.S, self.N, action)
        self._update_hist(dist)
        self.location += mvmt
        return dist

    def _get_info(self):
        info = {
            'change_point': self.change_point,
            'S': self.S,
            'total_dist': self.total_dist,
            'location': self.location,
            'location_hist': self.location_hist
        }
        return info

    def _correct_action(self, action:int):
        """
        Do not allow 0 or N as actions
        :param action:
        :return:
        """
        if action == 0:
            action += 1
        elif action == self.N:
            action -= 1
        return action

    def step(self, action: int):
        assert self.action_space.contains(action)
        action = self._correct_action(action)
        dist = self._move_agent(action)

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

        self._update_state()

        if np.abs(self.location - self.change_point) <= 1:
            done = True
        else:
            done = False

        reward = -1 * self._cost(dist)
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
        time.sleep(0.1)

    def close(self):
        plt.clf()
        plt.cla()
