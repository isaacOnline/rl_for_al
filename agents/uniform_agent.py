import numpy as np
import gym
import matplotlib.pyplot as plt
from agents.value_iterator import ValueIterator
from tqdm import tqdm
from base.scorer import UniformScorer

class UniformAgent(ValueIterator):
    def __init__(self, delta, epsilon = None, sample_cost = 1, movement_cost = 1000):
        ValueIterator.__init__(self, delta, epsilon, sample_cost, movement_cost)
        self.all_states = np.linspace(0, 1, self.N + 1, dtype=np.float64)
        self.policy = np.zeros(self.all_states.shape, dtype=np.int)
        self.state_values = np.zeros(self.all_states.shape)
        self.calculate_policy()

    def calculate_policy(self):
        # initialize value, number of samples, and distance for each state
        # states are possible lengths of hypothesis space, so there are N total
        # Off by one here?
        S = np.linspace(0, 1, self.N + 1)
        f = [0] * (self.N + 1)  # policy stored as array of where to go from each state
        val = [0] * (self.N + 1)  # value of state - total cost to termination
        ns = [0] * (self.N + 1)  # number of samples to termination from this state
        dist = [0] * (self.N + 1)  # distance to termination from this state
        gp = [0] * (self.N + 1) # numbers to feed to gym, which are scaled as portions of the hyp space to travel

        # terminal states are those smaller than stopErr
        T = np.where(S < self.epsilon)[0]

        # loop over non-terminal states
        for ss in range(T[-1] + 1, self.N + 1):
            minVal = np.inf
            # loop over all states we can go to from state ss
            for tt in range(1, ss):
                Pst = tt / ss  # probability of transition to state tt
                Pstc = (ss - tt) / ss  # probability of transition to state ss - tt
                tempVal = self.sample_cost * (1 + Pst * ns[tt] + Pstc * ns[ss - tt]) + self.movement_cost * (
                            tt * self.delta + Pst * dist[tt] + Pstc * dist[ss - tt])
                if tempVal < minVal:
                    minVal = tempVal
                    f[ss] = tt
                    gp[ss] = round(tt * self.N /ss)

            # update value, ns, and dist for this state
            Psf = f[ss] / ss
            Psfc = (ss - f[ss]) / ss
            ns[ss] = 1 + Psf * ns[f[ss]] + Psfc * ns[ss - f[ss]]
            dist[ss] = f[ss] * self.delta + Psf * dist[f[ss]] + Psfc * dist[ss - f[ss]]
            val[ss] = self.sample_cost * ns[ss] + self.movement_cost * dist[ss]

        self.policy = np.array([fs / self.N for fs in f])
        self.gym_policy = gp

    def save(self):
        plt.clf()
        plt.plot(range(len(ret[0])), ret[0])
        img_path = f"agents/visualizations/{int(self.movement_cost * self.N)}_uniform.png"
        self._save_image(img_path)
        # Saved as truemovementcost_N_uniform.csv
        policy_path = f"experiments/vi_vs_rl/uniform/vi_policies/{self.movement_cost}_{self.N}_uniform.csv"
        np.savetxt(policy_path, self.gym_policy)


if __name__ == "__main__":
    sample_cost = 1
    movement_cost = 1
    delta = 1/1000
    kwargs = {
        'sample_cost': sample_cost,
        'movement_cost': movement_cost,
        'delta': delta
    }


    agnt = UniformAgent(delta, movement_cost=movement_cost)
    agnt.save()
    flat_policy = np.array(agnt.gym_policy).flatten()

    UniformScorer().score(flat_policy, gym.make("change_point:uniform-v0", **kwargs), trials = 10000)

