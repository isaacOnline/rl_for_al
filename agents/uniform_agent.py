import numpy as np
import gym
import matplotlib.pyplot as plt
from agents.value_iterator import ValueIterator
from tqdm import tqdm
from base.scorer import UniformScorer

class UniformAgent(ValueIterator):
    def __init__(self, N, sample_cost = 1, movement_cost = 1000):
        self.all_states = np.linspace(0, N, N + 1, dtype=np.int)
        ValueIterator.__init__(self, N, sample_cost, movement_cost)
        self.policy = np.zeros(self.all_states.shape, dtype=np.int)
        self.state_values = np.zeros(self.all_states.shape)
        self.calculate_policy()

    def calculate_policy(self):
        # initialize value, number of samples, and distance for each state
        # states are possible lengths of hypothesis space, so there are N+1 total
        num_samples = [0] * (self.N + 1)  # number of samples to termination from this state
        distance_traveled = [0] * (self.N + 1)  # distance to termination from this state


        with tqdm(total = self.N * (self.N - 1) / 2) as progress:
            for state in range(2, self.N + 1):
                best_value = np.inf
                # loop over all states we can go to from state ss
                for s_prime in range(1, state):
                    Pst = s_prime / state  # probability of transition to state tt
                    Pstc = (state - s_prime) / state  # probability of transition to state ss - tt
                    this_value = self.sample_cost * (1 +
                                                     Pst * num_samples[s_prime] +
                                                     Pstc * num_samples[state - s_prime]) + \
                                 self.movement_cost * (s_prime + Pst * distance_traveled[s_prime] +
                                                       Pstc * distance_traveled[state - s_prime])
                    if this_value < best_value:
                        best_value = this_value
                        self.policy[state] = s_prime

                # update value, ns, and dist for this state
                Psf = self.policy[state] / state
                Psfc = (state - self.policy[state]) / state
                num_samples[state] = 1 + Psf * num_samples[self.policy[state]] + Psfc * num_samples[state - self.policy[state]]
                distance_traveled[state] = self.policy[state] + Psf * distance_traveled[self.policy[state]] + Psfc * distance_traveled[state - self.policy[state]]
                self.state_values[state] = self.sample_cost * num_samples[state] + self.movement_cost * distance_traveled[state]
                progress.update(state-1)


    def save(self):
        plt.clf()
        plt.plot(self.all_states, np.array(self.policy.copy()).flatten())
        img_path = f"agents/visualizations/{int(self.movement_cost * self.N)}_uniform.png"
        self._save_image(img_path)
        # Saved as truemovementcost_N_uniform.csv
        policy_path = f"experiments/vi_vs_rl/uniform/vi_policies/{int(self.movement_cost * self.N)}_{self.N}_uniform.csv"
        np.savetxt(policy_path, self.policy)




if __name__ == "__main__":
    sample_cost = 1
    movement_cost = 1
    N = 5
    kwargs = {
        'sample_cost': sample_cost,
        'movement_cost': movement_cost,
        'N': N
    }


    agnt = UniformAgent(N, movement_cost=movement_cost)
    agnt.save()
    flat_policy = np.array(agnt.policy).flatten()

    UniformScorer().score(flat_policy, gym.make("change_point:uniform-v0", **kwargs), trials = 10000)

