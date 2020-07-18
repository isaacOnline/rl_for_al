import numpy as np

from tqdm import trange
from agents.value_iterator import ValueIterator
from scipy.stats import uniform, truncnorm
from datetime import datetime
from base.scorer import RechargingScorer
import gym

class RechargingAgent(ValueIterator):
    def __init__(self, sample_cost, movement_cost, battery_capacity, delta, gamma, dist=None, seed=None, epsilon = None):

        assert battery_capacity > (2 * movement_cost + sample_cost), "Battery capacity must be enough " \
                                                                     "to take a sample at the other end of the " \
                                                                     "hypothesis space and then return to the origin"

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
        # (This supresses a downstream floating point issue)
        assert np.isclose(0, (self.epsilon / self.delta) % 1) or \
               np.isclose(1, (self.epsilon / self.delta) % 1)


        battery_options =  int(round(battery_capacity / gamma))
        state_shape = (self.N + 1, self.N+1, battery_options + 1)

        self.policy = np.zeros(state_shape, dtype=np.int)

        # Gym actions are policies
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

                # now compute value of each state
                for opposite_bound in opposite_bounds:
                    # The current battery level must be enough so that you can return to origin
                    min_battery_level = location / self.N * self.movement_cost

                    # The battery can only be full at the origin, after which the agent had to travel to
                    # the current location, so the max possible current battery level varies by location
                    max_battery_level = self.battery_capacity - location / self.N * self.movement_cost

                    assert min_battery_level <= max_battery_level, \
                        f"There are no valid actions to take at position {(location, opposite_bound)} " \
                        f"for agent with Tt: {self.movement_cost}, Ts: {self.sample_cost}, " \
                        f"battery cap: {self.battery_capacity}"

                    for battery_level in np.arange(min_battery_level, max_battery_level + self.gamma,
                                                   self.gamma):
                        assert np.isclose(0, battery_level / self.gamma % 1) or np.isclose(1,
                                                                                           battery_level / self.gamma % 1)
                        best_ev = np.inf
                        best_action = np.inf
                        best_recharge_time = np.inf

                        for movement in range(1, h_space_len):
                            if location < opposite_bound:
                                # move forward
                                direction = 1
                                new_location = location + direction * movement
                                state_prob = np.float128(self.dist.cdf(opposite_bound/self.N) - self.dist.cdf(location/self.N))
                                this_section_prob = np.float128(self.dist.cdf(new_location/self.N) - self.dist.cdf(location/self.N))
                                if state_prob == 0:
                                    Pst = 0.5
                                else:
                                    Pst = this_section_prob/state_prob
                            else:
                                # move backward
                                direction = -1
                                new_location = location + direction * movement
                                state_prob = np.float128(self.dist.cdf(location/self.N) - self.dist.cdf(opposite_bound/self.N))
                                this_section_prob = np.float128(self.dist.cdf(location/self.N) - self.dist.cdf(new_location/self.N))
                                if state_prob == 0:
                                    # if probability of change point being in this state is so small that it
                                    # can't be stored in a 64-bit float, then pretend the probability is uniform
                                    Pst = movement/h_space_len
                                else:
                                    Pst = this_section_prob/state_prob
                                # Pstc is the complement (theta between xx +/- aa and xh)
                            Pstc = 1 - Pst

                            # The recharge time must be enough so that the agent can return to the origin after sampling
                            dist_to_origin = location
                            dist_from_origin = new_location
                            min_new_battery_level = dist_from_origin / self.N * self.movement_cost
                            min_battery_at_origin = min_new_battery_level \
                                                    + dist_from_origin / self.N * self.movement_cost \
                                                    + self.sample_cost
                            min_recharge_time = min_battery_at_origin \
                                                - battery_level \
                                                + dist_to_origin / self.N * self.movement_cost
                            min_recharge_time = max(min_recharge_time, 0)

                            # The battery at origin is capped at battery capacity
                            max_recharge_time = self.battery_capacity - battery_level + dist_to_origin/self.N * self.movement_cost

                            # Recharge times must be divisible by gamma
                            # Ceiling division is used because overcharging is allowable, but undercharging could the agent
                            min_recharge_time = self.gamma * np.ceil(min_recharge_time / self.gamma)
                            max_recharge_time = self.gamma * np.ceil(max_recharge_time / self.gamma)


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
        dist_name = self.dist.dist.name

        # Save as a numpy array, since it's more than 2d and a csv wouldn't really make sense
        policy_path = f"experiments/vi_vs_rl/recharging/vi_policies/" \
                      f"{int(self.movement_cost)}_" \
                      f"{self.sample_cost}_" \
                      f"{self.N}_" \
                      f"{self.gamma}_" \
                      f"{self.battery_capacity}_" \
                      f"{dist_name}.npy"
        np.save(policy_path, self.gym_actions)

def get_dist():
    min = 0
    max = 1
    mean = 0.5
    sd = np.sqrt(0.1)
    a = (min - mean) / sd
    b = (max - mean) / sd
    dist = truncnorm(a, b, loc=mean, scale=sd)
    return dist

def get_unif():
    return uniform(0,1)

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

