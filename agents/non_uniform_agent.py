import numpy as np

from tqdm import tqdm
from agents.value_iterator import ValueIterator
from scipy.stats import uniform, truncnorm
from datetime import datetime
from base.scorer import NonUniformScorer
import gym


class NonUniformAgent(ValueIterator):
    def __init__(self, delta, epsilon = None, sample_cost = 1, movement_cost = 1000, dist = None):
        ValueIterator.__init__(self, delta, epsilon, sample_cost, movement_cost)
        # Dist must be an object with a cdf function
        if dist is None:
            self.dist = uniform(0, 1)
        else:
            self.dist = dist

        self.policy = np.zeros((self.N + 1, self.N+1), dtype=np.int)

        self.state_values = np.zeros((self.N + 1, self.N + 1),dtype=np.float128)

        self.gym_actions = np.zeros((self.N + 1, self.N + 1), np.int)

    def calculate_policy(self):
        # states are tuples (x,xh) where x is the current location and xh is
        # the opposite end of the hypothesis space
        start_time = datetime.now()

        num_iterations = (self.N - 1) * self.N * (2 * (self.N - 1) + 1) / 6
        with tqdm(total=num_iterations) as pbar:
            # find the index where the hypothesis space is small enough (ie equal to epsilon), then only
            # search hypothesis space lengths larger than this
            terminal_index = int(round(self.epsilon * self.N))
            for h_space_len in range(terminal_index  + 1, self.N + 1):
                # loop over possible current locations
                for location in range(self.N + 1):
                    # calculate possible values of xh
                    opposite_bounds = []
                    if location - h_space_len >= 0:
                        opposite_bounds.append(location - h_space_len)
                    if location + h_space_len <= self.N:
                        opposite_bounds.append(location + h_space_len)

                    # now compute value of each state
                    if h_space_len <= 1:
                        for opposite_bound in opposite_bounds:
                            self.state_values[location, opposite_bound] = 0
                    else:
                        for opposite_bound in opposite_bounds:
                            # look at all possible actions from this state
                            # actions are distances to travel
                            best_ev = np.inf
                            best_action = 1
                            for action in range(1, h_space_len):
                                pbar.update()
                                # Pst is the probability theta lies between xx and xx +/- aa
                                if location < opposite_bound:
                                    # move forward
                                    direction = 1
                                    new_location = location + direction * action
                                    state_prob = np.float128(self.dist.cdf(opposite_bound/self.N) - self.dist.cdf(location/self.N))
                                    this_section_prob = np.float128(self.dist.cdf(new_location/self.N) - self.dist.cdf(location/self.N))
                                    if state_prob == 0:
                                        Pst = 0.5
                                    else:
                                        Pst = this_section_prob/state_prob
                                else:
                                    # move backward
                                    direction = -1
                                    new_location = location + direction * action
                                    state_prob = np.float128(self.dist.cdf(location/self.N) - self.dist.cdf(opposite_bound/self.N))
                                    this_section_prob = np.float128(self.dist.cdf(location/self.N) - self.dist.cdf(new_location/self.N))
                                    if state_prob == 0:
                                        # if probability of change point being in this state is so small that it
                                        # can't be stored in a 64-bit float, then pretend the probability is uniform
                                        Pst = action/h_space_len
                                    else:
                                        Pst = this_section_prob/state_prob
                                    # Pstc is the complement (theta between xx +/- aa and xh)
                                Pstc = 1 - Pst
                                ev = self.sample_cost + self.movement_cost / self.N * action+ \
                                     Pst * self.state_values[new_location, location] + \
                                     Pstc * self.state_values[new_location, opposite_bound]
                                if ev < best_ev:
                                    best_action = action
                                    best_ev = ev

                            self.policy[location, opposite_bound] = best_action
                            self.state_values[location, opposite_bound] = best_ev
                            self.gym_actions[location,opposite_bound] = round(best_action/h_space_len * self.N)
        self.policy = self.policy/self.N
        end_time = datetime.now()
        self.train_time = end_time - start_time
        return self.train_time

    def save(self):
        dist_name = self.dist.dist.name
        policy_path = f"experiments/vi_vs_rl/non_uniform/vi_policies/{int(self.movement_cost)}_{self.N}_{dist_name}.csv"
        np.savetxt(policy_path, self.gym_actions)

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
    movement_cost = 100
    N = 100
    delta = 1/N
    dist=get_unif()
    kwargs = {
        'sample_cost': sample_cost,
        'movement_cost': movement_cost,
        'delta': delta,
        'dist': dist
    }


    agnt = NonUniformAgent(**kwargs)
    train_time = agnt.calculate_policy()
    agnt.save()


    NonUniformScorer().score(agnt.gym_actions, gym.make("change_point:non_uniform-v0", **kwargs), trials = 10000)

