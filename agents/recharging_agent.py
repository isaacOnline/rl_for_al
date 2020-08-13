import numpy as np

from tqdm import trange
from agents.value_iterator import ValueIterator
from scipy.stats import uniform
from datetime import datetime
from base.scorer import RechargingScorer
from base.distributions import get_unif
import gym

class RechargingAgent(ValueIterator):
    """
    This agent is meant to calculate a policy for the recharging change point problem. The policy can then be fed to a
    NonUniformScorer to see how well the agent performs on the non_uniform-v0 gym environment.
    """
    def __init__(self, sample_cost, movement_cost, battery_capacity, delta, gamma, dist=None, seed=None, epsilon = None):
        """
        Save parameters, validate that the specified parameters work in combination.

        :param sample_cost: Ts from sps paper
        :param movement_cost: Tt from sps paper
        :param battery_capacity: T0 from sps paper
        :param delta: Upper case delta from the sps paper
        :param gamma: Lower case delta from the sps paper, since "Delta" vs "delta" seemed like a bad idea
        :param dist:
        :param seed:
        :param epsilon: Epsilon from sps paper
        """
        assert battery_capacity > (2 * movement_cost + sample_cost), "Battery capacity must be enough " \
                                                                     "to take a sample at the other end of the " \
                                                                     "hypothesis space and then return to the origin"

        # Check that battery battery capacity is divisible by gamma
        assert np.isclose(0, (battery_capacity / gamma) % 1) or \
               np.isclose(1, (battery_capacity / gamma) % 1)

        # make sure (movement cost x delta) is divisible by gamma, as battery level needs to be divisible by gamma
        # (since the cost of an action determines how much the battery level declines)
        assert np.isclose(0, (movement_cost * delta / gamma) % 1) or \
               np.isclose(1, (movement_cost * delta / gamma) % 1)


        ValueIterator.__init__(self, delta, epsilon, sample_cost, movement_cost)
        self.gamma = gamma
        self.battery_capacity = battery_capacity
        # Dist must be an object with a cdf function
        if dist is None:
            self.dist = uniform(0, 1)
        else:
            self.dist = dist

        # Make sure that terminal h space length is divisible by delta
        # (This suppresses a downstream floating point issue)
        assert np.isclose(0, (self.epsilon / self.delta) % 1) or \
               np.isclose(1, (self.epsilon / self.delta) % 1)

        # The number of possible battery levels
        battery_options = int(round(battery_capacity / gamma))
        state_shape = (self.N + 1, self.N+1, battery_options + 1)

        self.policy = np.zeros(state_shape, dtype=np.int)

        # Gym actions are tuples, (movement action, recharge action). The movement action is handled like
        # it was in the uniform case, and the recharge action is l from section 3.3 of the sps paper.
        self.gym_actions = np.zeros(state_shape, dtype = (np.float128, 2))

        self.state_values = np.zeros(state_shape,dtype=np.float128)

    def calculate_policy(self):
        start_time = datetime.now()

        # find the index where the hypothesis space is small enough (ie equal to epsilon), then only
        # search hypothesis space lengths larger than this
        terminal_index = int(round(self.epsilon * self.N))
        for h_space_len in trange(terminal_index + 1, self.N + 1):
            # loop over possible current locations
            for location in range(self.N + 1):
                # calculate possible values of xh
                opposite_bounds = []
                if location - h_space_len >= 0:
                    opposite_bounds.append(location - h_space_len)
                if location + h_space_len <= self.N:
                    opposite_bounds.append(location + h_space_len)

                for opposite_bound in opposite_bounds:
                    # Only certain battery levels are possible at certain states, so this just
                    # finds the min/max battery levels that we need to look at

                    # The current battery level must be enough so that you can return to origin
                    min_battery_level = location / self.N * self.movement_cost

                    # The battery can only be full at the origin, after which the agent had to travel to
                    # the current location, so the max possible current battery level varies by location
                    max_battery_level = self.battery_capacity - location / self.N * self.movement_cost

                    # This error can arise when battery capacity is very low
                    assert min_battery_level <= max_battery_level, \
                        f"There are no valid actions to take at position {(location, opposite_bound)} " \
                        f"for agent with Tt: {self.movement_cost}, Ts: {self.sample_cost}, " \
                        f"battery cap: {self.battery_capacity}"

                    # loop through possible battery levels
                    for battery_level in np.arange(min_battery_level, max_battery_level + self.gamma,
                                                   self.gamma):

                        # Make sure battery level is divisible by gamma
                        assert np.isclose(0, battery_level / self.gamma % 1) or \
                               np.isclose(1, battery_level / self.gamma % 1)

                        best_ev = np.inf
                        best_action = np.inf
                        best_recharge_time = np.inf

                        # Loop through possible movements
                        for movement in range(1, h_space_len):
                            if location < opposite_bound:
                                # move forward
                                direction = 1
                                new_location = location + direction * movement
                                state_prob = np.float128(self.dist.cdf(opposite_bound/self.N) - self.dist.cdf(location/self.N))
                                this_section_prob = np.float128(self.dist.cdf(new_location/self.N) - self.dist.cdf(location/self.N))
                                assert state_prob != 0
                                Pst = this_section_prob/state_prob
                            else:
                                # move backward
                                direction = -1
                                new_location = location + direction * movement
                                state_prob = np.float128(self.dist.cdf(location/self.N) - self.dist.cdf(opposite_bound/self.N))
                                this_section_prob = np.float128(self.dist.cdf(location/self.N) - self.dist.cdf(new_location/self.N))

                                assert state_prob != 0
                                Pst = this_section_prob/state_prob
                                # Pstc is the complement (theta between xx +/- aa and xh)
                            Pstc = 1 - Pst

                            # The recharge time must be enough so that the agent can return to the origin after sampling
                            dist_to_origin = location
                            dist_from_origin = new_location # <- distance that agent will move after it returns to the origin,
                                                            # to get to the new location
                            min_new_battery_level = dist_from_origin / self.N * self.movement_cost
                            min_battery_at_origin = min_new_battery_level \
                                                    + dist_from_origin / self.N * self.movement_cost \
                                                    + self.sample_cost
                            min_recharge_time = min_battery_at_origin \
                                                - battery_level \
                                                + dist_to_origin / self.N * self.movement_cost
                            min_recharge_time = max(min_recharge_time, 0)

                            # The battery at origin is capped at battery capacity. (The latter term is what the battery
                            # level will be when the agent has returned to the origin)
                            max_recharge_time = self.battery_capacity - (battery_level - dist_to_origin/self.N * self.movement_cost)

                            # Recharge times must be divisible by gamma
                            # Ceiling division is used because overcharging is allowable, but undercharging could kill
                            # the agent
                            min_recharge_time = self.gamma * np.ceil(min_recharge_time / self.gamma)
                            max_recharge_time = self.gamma * np.ceil(max_recharge_time / self.gamma)

                            # Loop through possible recharge actions
                            for recharge_time in np.arange(min_recharge_time, max_recharge_time + self.gamma, self.gamma):
                                if recharge_time > 0:
                                    battery_at_origin = battery_level \
                                        - dist_to_origin / self.N * self.movement_cost\
                                        + recharge_time
                                    new_battery_level = battery_at_origin \
                                                         - dist_from_origin / self.N * self.movement_cost\
                                                         - self.sample_cost
                                    travel_dist = movement + dist_to_origin + dist_from_origin
                                else:
                                    new_battery_level = battery_level \
                                                         - movement / self.N * self.movement_cost\
                                                         - self.sample_cost
                                    travel_dist = movement

                                # calculate state action value
                                new_battery_index = int(round(new_battery_level / self.gamma))
                                ev = self.sample_cost \
                                     + self.movement_cost * travel_dist / self.N \
                                     + recharge_time \
                                     + Pst * self.state_values[new_location, location, new_battery_index] \
                                     + Pstc * self.state_values[new_location, opposite_bound, new_battery_index]
                                if ev < best_ev:
                                    best_action = movement
                                    best_ev = ev
                                    best_recharge_time = recharge_time

                        # Save the actions for this state (a state is the tuple (location, opposite bound,
                        # battery_index), where battery_index is t from 3.3 of the sps paper)
                        battery_index = int(round(battery_level / self.gamma))
                        self.policy[location, opposite_bound, battery_index] = best_action
                        self.state_values[location, opposite_bound, battery_index] = best_ev
                        self.gym_actions[location,opposite_bound, battery_index] = (round(best_action/h_space_len * self.N),
                                                                                    best_recharge_time / self.gamma)

        self.policy = self.policy/self.N
        end_time = datetime.now()
        self.train_time = end_time - start_time
        return self.train_time

    def save(self):
        """
        Save gym policy for later use, so we don't have to recalculate
        """
        dist_name = self.dist.dist.name

        # Save as a numpy array, since it's more than 2d and a csv wouldn't really make sense
        policy_path = f"experiments/vi_vs_sb/recharging/vi_policies/" \
                      f"{int(self.movement_cost)}_" \
                      f"{self.sample_cost}_" \
                      f"{self.N}_" \
                      f"{self.gamma}_" \
                      f"{self.battery_capacity}_" \
                      f"{dist_name}.npy"
        np.save(policy_path, self.gym_actions)


if __name__ == "__main__":
    sample_cost = 1
    movement_cost = 10
    N = 600
    delta = 1/N
    dist=get_unif()
    battery = 50
    epsilon = 100
    kwargs = {
        'sample_cost': sample_cost,
        'movement_cost': movement_cost,
        'delta': delta,
        'battery_capacity': battery,
        'gamma': 1,
        'dist': dist,
        'epsilon': epsilon
    }


    agnt = RechargingAgent(**kwargs)
    train_time = agnt.calculate_policy()
    agnt.save()

    RechargingScorer().score(agnt.gym_actions, gym.make("change_point:recharging-v0", **kwargs), trials=10000)

