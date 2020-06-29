import time
from change_point.envs.change_point import ChangePoint
from scipy.stats import truncnorm
from gym.spaces import MultiDiscrete
import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Discrete
from matplotlib.gridspec import GridSpec

# The default distribution here is truncated normal
class RechargingCP(ChangePoint):
    def __init__(self, sample_cost, movement_cost, battery_capacity, delta, gamma, dist=None, seed=None, epsilon = None):
        """

        :param sample_cost:
        :param movement_cost:
        :param battery_capacity:
        :param delta:
        :param gamma: I renamed lowercase delta from the paper
        :param dist:
        :param seed:
        :param epsilon:
        """
        # battery capacity must be an integer
        assert type(battery_capacity) is int

        assert np.isclose(0, (battery_capacity / gamma) % 1)

        # make sure (movement cost x delta) is divisible by gamma, as battery level needs to be divisible by gamma
        # (since the cost of an action determines how much the battery level declines)
        assert np.isclose(0, (movement_cost * delta / gamma) % 1)

        self.battery_capacity = battery_capacity

        self.gamma = gamma
        ChangePoint.__init__(self, sample_cost, movement_cost, delta, dist, seed)

    def _set_action_space(self):
        # Calculate the number of possible recharge actions
        possible_recharge_actions = int(round(self.battery_capacity / self.gamma))

        # This isn't self.N + 1 because moving N units would not decrease the size of the hypothesis space
        # This isn't num_recharge_actions + 1, because we can never get to a battery level of 0, meaning we could never
        # have low enough battery to complete a full recharge
        self.action_space = MultiDiscrete([self.N, possible_recharge_actions])

    def _cost(self, action):
        # TODO: Right now, if the agent decides to charge for longer than the battery capacity, it gets charged for the whole recharge time. it might be best to change this
        if self._dead_battery():
            # If battery dies, return huge negative reward
            return (self.movement_cost + self.sample_cost + self.N) * 100000
        else:
            # The cost is the sampling time,
            # Plus (the time to move one unit) multiplied by (the number of units moved),
            # Plus the recharge time
            return self.sample_cost + self.movement_cost * action[0] + self._scale_recharge_action(action[1])

    def _dead_battery(self):
        return self.battery_level < 0

    def _initialize_distribution(self, dist=None):
        if dist is None:
            # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            # for description of how truncnorm is being used
            min = 0
            max = 1
            mean = 0.5
            sd = np.sqrt(0.1)
            a = (min - mean) / sd
            b = (max - mean) / sd
            dist = truncnorm(a, b, loc=mean,scale=sd)
        self.dist = dist

    def reset(self):
        self.battery_level = self.battery_capacity
        return ChangePoint.reset(self)


    def _initialize_state(self):
        battery_options = int(round(self.battery_capacity / self.gamma))
        self.observation_space = MultiDiscrete([self.N+1, self.N+1, battery_options +1])

    def _update_state(self):
        self.S = np.array([self.location, self.opposite_bound, self.battery_level])
        self.h_space_len = self.max_loc - self.min_loc

    def _move_agent(self, action):
        dist, mvmt = self.get_movement(action)
        self._update_battery(action, dist)
        self._update_hist(dist)
        self.location += mvmt
        return (dist, action[1])

    def _update_battery(self, action, dist):
        # If we're recharging, we must return to origin
        if action[1] > 0:
            dist_to_origin = self.location
            dist_from_origin = dist - dist_to_origin

            battery_at_origin = min(
                self.battery_capacity, # Can't have more charge than battery capacity
                self.battery_level \
                    - dist_to_origin * self.movement_cost \
                    + self._scale_recharge_action(action[1])
            )
            self.battery_level = battery_at_origin \
                                 - dist_from_origin * self.movement_cost \
                                 - self.sample_cost
        else:
            self.battery_level -= dist * self.movement_cost \
                                  + self.sample_cost

    def _scale_recharge_action(self, raw_action):
        return int(round(raw_action * self.gamma))

    def get_movement(self, action):
        """
        :param action:
        :return:
        """
        portion = action[0]
        k = round(portion * self.h_space_len)
        distance = k * self.delta

        if np.isclose(distance, 0):
            distance += self.delta
        if np.isclose(distance, self.h_space_len):
            distance -= self.delta

        mvmt = distance * self.direction

        # compute how many units moved, considering recharging
        if action[1] > 0:
            # If recharging, we have to travel to the origin, then from there to our new location
            new_location = self.location + mvmt
            distance = self.location + new_location
        else:
            pass

        return distance, mvmt

    def _discrete_state(self):
        return np.array([int(round(self.location * self.N)),
                         int(round(self.opposite_bound * self.N)),
                         int(round(self.battery_level))])


    def render(self, mode='human'):
        fig, ax = plt.subplots(1, 2, figsize=(7,2))
        gs = GridSpec(1, 2, width_ratios=[8, 1], figure = fig)

        ax1 = plt.subplot(gs[0])
        ax1.set_title("Location")
        ax1.plot([0,1],[0,0],c='black')
        ax1.plot([self.min_loc, self.max_loc], [0,0], c='orange')
        ax1.scatter(self.change_point, 0, c='red',marker='x', zorder=1000)
        ax1.scatter(self.location, 0, c='m', zorder=1000)
        ax1.get_yaxis().set_visible(False)

        if self.battery_level / self.battery_capacity > 0.6:
            battery_color = 'green'
        elif self.battery_level / self.battery_capacity > 0.2:
            battery_color = 'yellow'
        else:
            battery_color = 'red'


        ax2 = plt.subplot(gs[1])
        ax2.set_title("Battery Level")
        ax2.plot([0, 0], [0, self.battery_capacity], c='lightgray',linewidth = 40, solid_capstyle="butt")
        ax2.plot([0, 0], [0, self.battery_level],c=battery_color, linewidth = 40, solid_capstyle="butt")

        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().tick_right()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        fig.tight_layout()
        fig.show()
        if mode == 'human':
            time.sleep(0.1)