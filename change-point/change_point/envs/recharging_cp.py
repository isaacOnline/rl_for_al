from change_point.envs.change_point import ChangePoint
from scipy.stats import truncnorm
from gym.spaces import MultiDiscrete
import numpy as np
from gym.spaces import Discrete


# The default distribution here is truncated normal

class RechargingCP(ChangePoint):
    def __init__(self, sample_cost, movement_cost, battery_capacity, N, gamma, dist=None, seed=None):
        """
        :param stop_error:
        :param seed:
        """
        self.battery_capacity = battery_capacity
        self.battery_level = battery_capacity
        self.gamma = gamma
        ChangePoint.__init__(self, sample_cost, movement_cost, N, dist, seed)

    def set_action_space(self):
        # This isn't self.N + 1 because moving N units would not decrease the size of the hypothesis space
        # This isn't self.battery_level + 1, because we can never get to a battery level of 0, meaning you could never
        # have low enough battery to complete a full recharge
        self.action_space = MultiDiscrete([self.N, self.battery_level])

    def _cost(self, action):
        # The cost is the sampling time,
        # Plus (the time to move one unit) multiplied by (the number of units moved),
        # Plus the recharge time
        if action[1] > 0:
            return_to_origin_cost = self.location * 2 * self.movement_cost
        else:
            return_to_origin_cost = 0

        return self.sample_cost + self.movement_cost * action[0] + action[1] + return_to_origin_cost


    def _initialize_distribution(self, dist=None):
        if dist is None:
            # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            # for description of how truncnorm is being used
            min = 0
            max = self.N
            mean = self.N * 0.5
            sd = np.sqrt(self.N * 0.1)
            a = (min - mean) / sd
            b = (max - mean) / sd
            dist = truncnorm(a, b, loc=mean,scale=sd)
        self.dist = dist

    def reset(self):
        self.battery_level = self.battery_capacity
        ChangePoint.reset(self)

    def _initialize_state(self):
        self.observation_space = MultiDiscrete([self.battery_capacity+1,self.N+1, self.N+1])

    def _update_state(self):
        self.S = np.array([self.battery_level, self.location, self.opposite_bound])
        self.h_space_len = self.max_loc - self.min_loc

    def _move_agent(self, action: float):
        dist, mvmt = self.get_movement(action)
        self._update_hist(dist)
        self.location += mvmt