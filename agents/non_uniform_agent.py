import numpy as np

from tqdm import tqdm
from agents.value_iterator import ValueIterator
from scipy.stats import uniform
from itertools import product
from base.scorer import NonUniformScorer
import gym


class NonUniformAgent(ValueIterator):
    def __init__(self, N, sample_cost = 1, movement_cost = 1000, dist = None):
        # Dist must be an object with a cdf function

        # Only let there be 1/self.stop_error + 1 number of states, since it doesn't make sense to have a state that is
        # smaller than the stop error. I'm still going to round the states, because otherwise the state space isn't
        # discrete.
        if dist is None:
            self.dist = uniform(0, N)
        else:
            self.dist = dist

        self.all_states = np.array(list(product(np.linspace(0, N, N + 1, dtype=np.int),
                                  np.linspace(0, N, N + 1, dtype=np.int))))
        self.all_states = self.all_states[np.argsort(np.abs(self.all_states[:,0] - self.all_states[:,1]))]
        self.all_states = [tuple(s) for s in self.all_states]

        ValueIterator.__init__(self, N, sample_cost, movement_cost)
        self.policy = np.zeros((N+1, N+1), dtype=np.int)

        self.state_values = np.zeros((N+1, N+1))
        self.calculate_policy()

    def calculate_policy(self):
        # states are tuples (x,xh) where x is the current location and xh is
        # the opposite end of the hypothesis space
        self.state_values = np.zeros((self.N + 1, self.N + 1))

        # loop over possible lengths of hypothesis space
        # dd = |x - xh|
        num_iterations = (self.N - 1) * self.N * (2 * (self.N - 1) + 1) / 6
        with tqdm(total=num_iterations) as pbar:
            for h_space_len in range(self.N + 1):
                # loop over possible current locations
                for location in range(self.N):
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
                            best_action = 0
                            for action in range(1, h_space_len):
                                pbar.update()
                                # Pst is the probability theta lies between xx and xx +/- aa
                                if location < opposite_bound:
                                    # move forward
                                    direction = 1
                                    new_location = location + direction * action
                                    state_prob = self.dist.cdf(opposite_bound) - self.dist.cdf(location)
                                    Pst = (self.dist.cdf(new_location) - self.dist.cdf(location)) / state_prob
                                else:
                                    # move backward
                                    direction = -1
                                    new_location = location + direction * action
                                    state_prob = self.dist.cdf(location) - self.dist.cdf(opposite_bound)
                                    Pst = (self.dist.cdf(location) - self.dist.cdf(new_location)) / state_prob
                                    # Pstc is the complement (theta between xx +/- aa and xh)
                                Pstc = 1 - Pst
                                ev = self.sample_cost + self.movement_cost * action / self.N + \
                                     Pst * self.state_values[new_location, location] + \
                                     Pstc * self.state_values[new_location, opposite_bound]
                                if ev < best_ev:
                                    best_action = action
                                    best_ev = ev
                            self.policy[location, opposite_bound] = best_action
                            self.state_values[location, opposite_bound] = best_ev

    def save(self):
        dist_name = self.dist.dist.name
        policy_path = f"experiments/vi_vs_rl/non_uniform/vi_policies/{int(self.movement_cost * self.N)}_{self.N}_{dist_name}.csv"
        np.savetxt(policy_path, self.policy)



if __name__ == "__main__":
    sample_cost = 1
    movement_cost = 1
    N = 10
    dist=uniform(0,N)
    kwargs = {
        'sample_cost': sample_cost,
        'movement_cost': movement_cost,
        'N': N,
        'seed': 5480275,
        'dist': dist
    }


    agnt = NonUniformAgent(N, movement_cost=movement_cost,dist=dist)
    agnt.save()
    NonUniformScorer().score(agnt.policy, gym.make("change_point:non_uniform-v0", **kwargs), trials = 1000000)

