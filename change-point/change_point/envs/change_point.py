import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import seeding
from abc import ABC, abstractmethod

class ChangePoint(gym.Env, ABC):
    """
    This is the base class for the change point problems. It has concrete methods to render, step, and reset, which
    rely on abstract methods implemented by each of the individual problem types.
    """
    def __init__(self, sample_cost, movement_cost, delta, dist=None, seed=0, epsilon = None):
        """
        Save params, set up attributes

        :param sample_cost: Ts
        :param movement_cost: Tt
        :param delta:
        :param dist:
        :param seed:
        :param epsilon:
        """
        self.seed(seed)
        self._set_args(sample_cost, movement_cost, delta, epsilon)
        self._initialize_distribution(dist)
        self._initialize_state()
        self.battery_level = np.Inf # (For uniform/non-uniform, since recharging env will overwrite this on reset)
        self.reset()

    @abstractmethod
    def _initialize_distribution(self, dist):
        """
        Create the distribution from which to draw change points.

        :param dist:
        :return:
        """
        pass

    @abstractmethod
    def _initialize_state(self):
        """
        Set the initial state

        :return:
        """
        pass

    @abstractmethod
    def _set_action_space(self):
        """
        Define possible moves for the agent.

        :return:
        """
        pass

    @abstractmethod
    def _update_state(self):
        """
        Update the internally-stored state and hypothesis space

        :return:
        """
        pass

    @abstractmethod
    def _discrete_state(self):
        """
        Returns the state, as an array of integers
        :return:
        """
        pass


    @abstractmethod
    def get_movement(self, action):
        """
        Translate the action into an actual distance to travel on the range [0,1]

        :param action:
        :return:
        """
        pass


    @abstractmethod
    def _move_agent(self, action):
        """
        Change the agent's position acording to the given action

        :param action:
        :return:
        """
        pass

    def _cost(self, action):
        """
        Determine how much an action costs

        :param action:
        :return:
        """
        return self.sample_cost + self.movement_cost * action

    def seed(self, seed=0):
        """
        Make env deterministic

        :param seed:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_args(self, sample_cost, movement_cost, delta, epsilon):
        """
        Save environment params. (Used in initialization)

        :param sample_cost:
        :param movement_cost:
        :param delta:
        :param epsilon:
        :return:
        """
        self.sample_cost = sample_cost
        self.movement_cost = movement_cost
        self.delta = delta
        if epsilon is None:
            epsilon = delta
        self.epsilon = epsilon
        self.N = round(1 / delta)
        self._set_action_space()

    def reset(self):
        """
        Return the environment to it's original state

        :return:
        """
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
        """
        Save the distance and location

        :param dist:
        :return:
        """
        self.total_dist += dist
        self.location_hist += [self.location]

    def _get_info(self):
        """
        Return a dictionary containing relevant diagnostic information. Info not to be used for training

        :return:
        """
        info = {
            'change_point': self.change_point,
            'S': self.S,
            'total_dist': self.total_dist,
            'location': self.location,
            'location_hist': self.location_hist
        }
        return info

    def step(self, action):
        """
        Perform one action on the environment

        :param action:
        :return:
        """
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
        if self.h_space_len < self.epsilon or np.isclose(self.h_space_len, self.epsilon) or self.battery_level < 0:
            done = True
        else:
            done = False

        reward = -1 * self._cost(cost_params)
        return self._discrete_state(), reward, done, self._get_info()

    def render(self, mode='human'):
        """
        Draw a pretty picture

        :param mode: If "human", wait 0.1 seconds before closing so you can actually see things
        :return:
        """
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
        """
        Method that needs to be implemented by any Env but is not really relevant here

        :return:
        """
        pass
