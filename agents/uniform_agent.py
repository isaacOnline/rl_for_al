import numpy as np
import gym
import matplotlib.pyplot as plt
from agents.value_iterator import ValueIterator
from tqdm import tqdm
from base.scorer import UniformScorer
from datetime import datetime

class UniformAgent(ValueIterator):
    def __init__(self, delta, epsilon = None, sample_cost = 1, movement_cost = 1000):
        ValueIterator.__init__(self, delta, epsilon, sample_cost, movement_cost)
        self.all_states = np.linspace(0, 1, self.N + 1, dtype=np.float64)
        self.policy = np.zeros(self.all_states.shape, dtype=np.int)
        self.state_values = np.zeros(self.all_states.shape)

    def calculate_policy(self):
        start = datetime.now()
        # initialize value, number of samples, and distance for each state
        # states are possible lengths of hypothesis space, so there are N + 1 total
        f = [0] * (self.N + 1)  # policy stored as array of where to go from each state
        val = [0] * (self.N + 1)  # value of state - total cost to termination
        num_samples = [0] * (self.N + 1)  # number of samples to termination from this state
        distance_to_term = [0] * (self.N + 1)  # distance to termination from this state
        gp = [0] * (self.N + 1) # numbers to feed to gym, which are scaled portions of the hyp space to travel

        # terminal states are those smaller/eq  stopErr
        # (Use all close to account for fpn errors)
        terminal_indexes = np.where((self.all_states < self.epsilon) | np.isclose(self.all_states, self.epsilon))[0]

        # loop over non-terminal states
        for state in range(terminal_indexes[-1] + 1, self.N + 1):
            best_value = np.inf
            # loop over all states we can go to from state state
            for s_prime in range(1, state):
                Pst = s_prime / state  # probability of transition to state s_prime
                Pstc = (state - s_prime) / state  # probability of transition to state state - s_prime
                this_value = self.sample_cost * (1 +
                                                 Pst * num_samples[s_prime] +
                                                 Pstc * num_samples[state - s_prime]) + \
                             self.movement_cost * (s_prime * self.delta +
                                                   Pst * distance_to_term[s_prime] +
                                                   Pstc * distance_to_term[state - s_prime])
                if this_value < best_value:
                    best_value = this_value
                    f[state] = s_prime
                    gp[state] = round(s_prime / state * self.N)

            # update value, ns, and dist for this state
            Psf = f[state] / state
            Psfc = (state - f[state]) / state
            num_samples[state] = 1 + Psf * num_samples[f[state]] + Psfc * num_samples[state - f[state]]
            distance_to_term[state] = f[state] * self.delta + Psf * distance_to_term[f[state]] + Psfc * distance_to_term[state - f[state]]
            val[state] = self.sample_cost * num_samples[state] + self.movement_cost * distance_to_term[state]

        self.policy = np.array([fs / self.N for fs in f])
        self.gym_actions = gp
        end = datetime.now()
        return end - start

    def save(self):
        plt.clf()
        plt.plot(self.all_states, np.array(self.policy.copy()).flatten())
        img_path = f"agents/visualizations/{int(self.movement_cost)}_uniform.png"
        self._save_image(img_path)
        # Saved as truemovementcost_N_uniform.csv
        policy_path = f"experiments/vi_vs_rl/uniform/vi_policies/{self.movement_cost}_{self.N}_uniform.csv"
        np.savetxt(policy_path, self.gym_actions)


if __name__ == "__main__":
    sample_cost = 1
    movement_cost = 100
    N = 500
    delta = 1/N
    epsilon = 0.2
    kwargs = {
        'sample_cost': sample_cost,
        'movement_cost': movement_cost,
        'delta': delta,
        'epsilon': epsilon
    }


    agnt = UniformAgent(**kwargs)
    agnt.calculate_policy()
    agnt.save()
    flat_policy = np.array(agnt.gym_actions).flatten()

    UniformScorer().score(flat_policy, gym.make("change_point:uniform-v0", **kwargs), trials = 10000)

