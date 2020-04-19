import numpy as np

import matplotlib.pyplot as plt
from agents.value_iterator import ValueIterator
from scipy.stats import uniform
from itertools import product
from other.Scorer import scorer
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

    def _state_is_terminal(self, state):
        return abs(state[0] - state[1]) <= 1

    def _calculate_prob(self, state_prime, state):
        state_prime_prob = np.array([self.dist.cdf(max(sp)) - self.dist.cdf(min(sp)) for sp in state_prime])
        state_prob = self.dist.cdf(max(state)) - self.dist.cdf(min(state))
        state_prob = np.repeat(state_prob, state_prime_prob.shape)
        return state_prime_prob / state_prob

    def save(self):
        plt.clf()
        plt.plot(range(11), self.policy[0])
        dist_name = self.dist.dist.name
        save_path = f"visualizations/value_iterator/{self.movement_cost * self.N}_{dist_name}.png"
        self._save_image(save_path)
        policy_path = f"results/value_iterator/{int(self.movement_cost * self.N)}_{self.N}_{dist_name}.csv"
        np.savetxt(policy_path, self.policy)

    def _calculate_action_space(self, s):
        # if first bound is greater than second bound, we're moving backwards
        if s[0] > s[1]:
            action_space = range(-1, s[1] - s[0], -1)
        # otherwise we're moving forwards
        else:
            action_space = range(1, s[1] - s[0], 1)
        return action_space

    def _get_new_states(self, actions, s):
        # If changepoint is between current location and new location
        s1 = [(s[0]+a, s[1]) for a in actions]

        # If changepoint is between new location and other end of the hypothesis space
        s2 = [(s[0]+a, s[0]) for a in actions]
        return s1, s2



if __name__ == "__main__":
    stop_error = 1
    sample_cost = 1
    movement_cost = 10
    N = 1000
    kwargs = {
        'Ts': sample_cost,
        'Tt': movement_cost,
        'N': N,
    }

    agnt = NonUniformAgent(N, movement_cost=movement_cost)
    agnt.calculate_policy()
    agnt.save()
    flat_policy = np.array(agnt.policy[0]).flatten()
    scorer().score(flat_policy, gym.make("change_point:uniform-v0", **kwargs), trials = 100000)

