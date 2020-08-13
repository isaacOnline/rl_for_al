import numpy as np
import gym
import matplotlib.pyplot as plt
from agents.value_iterator import ValueIterator
from base.scorer import UniformScorer
from datetime import datetime

class UniformAgent(ValueIterator):
    """
    This agent is meant to calculate a policy for the uniform change point problem. The policy can then be fed to a
    UniformScorer to see how well the agent performs on the uniform-v0 gym environment.

    This agent is based on John's code for value iteration, (for the uniform case)
    """
    def __init__(self, delta, epsilon = None, sample_cost = 1, movement_cost = 1000):
        """
        Save parameters and initialize policy

        :param delta: delta from the sps paper
        :param epsilon: epsilon from the sps paper
        :param sample_cost: Ts from the sps paper
        :param movement_cost: Tt from the sps paper
        """
        ValueIterator.__init__(self, delta, epsilon, sample_cost, movement_cost)
        self.all_states = np.linspace(0, 1, self.N + 1, dtype=np.float64)
        self.policy = np.zeros(self.all_states.shape, dtype=np.int)
        self.state_values = np.zeros(self.all_states.shape)

    def calculate_policy(self):
        """
        Calculate the optimal policy for the parameters given in initialization
        :return:
        """

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

                    # The action space can't be dynamic in gym, meaning that the gym policy can't be lengths of
                    # the hypothesis space to travel (since the lengths will change). Gym is also set up so that if you
                    # want to feed in discrete actions, they need to be integers. Which means we can't use actual
                    # fractions that say how far into the hypothesis space to travel. So instead I'm using the fraction,
                    # multiplied by the number of discrete states. This is all just logistics though, since things
                    # get scaled back to being lengths of the hypothesis space within the gym environment. As an example,
                    # if delta = 0.1, the length to travel is 0.3, and the hypothesis space is of length 0.6,
                    # then the gym action would be 0.3 / 0.6 * 10 = 5, which would then be converted back to 0.3 on the
                    # other side.
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

    def _save_image(self):
        """
        Save a visualization of the agent's calculated policy
        :return:
        """
        save_path = f"agents/visualizations/{int(self.movement_cost)}_uniform.png"
        plt.title(f"tt/ts: {self.movement_cost}/{self.sample_cost}, N: {self.N}")
        plt.xlabel("Size of Hypothesis Space")
        plt.xlim([0,1])
        plt.ylabel("Movement into Hypothesis Space")
        plt.ylim([0,1])
        plt.savefig(save_path)

    def save(self):
        """
        Save agent's policy (and a visualization of the policy) for later reference
        :return:
        """
        plt.clf()
        plt.plot(self.all_states, np.array(self.policy.copy()).flatten())
        self._save_image()
        policy_path = f"experiments/vi_vs_sb/uniform/vi_policies/{self.movement_cost}_{self.N}_uniform.csv"
        np.savetxt(policy_path, self.gym_actions)


if __name__ == "__main__":
    # Define params
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

    # Train agent
    agnt = UniformAgent(**kwargs)
    agnt.calculate_policy()
    agnt.save()

    # Get performance
    flat_policy = np.array(agnt.gym_actions).flatten()
    UniformScorer().score(flat_policy, gym.make("change_point:uniform-v0", **kwargs), trials=10000)

