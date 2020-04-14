import numpy as np

import matplotlib.pyplot as plt
from agents.value_iterator import ValueIterator
from other.Scorer import scorer
import gym


class UniformAgent(ValueIterator):
    def __init__(self, N, sample_cost = 1, movement_cost = 1000):
        # Only let there be 1/self.stop_error + 1 number of states, since it doesn't make sense to have a state that is
        # smaller than the stop error. I'm still going to round the states, because otherwise the state space isn't
        # discrete.
        self.all_states = np.linspace(0, N, N + 1, dtype=np.int)
        ValueIterator.__init__(self, N, sample_cost, movement_cost)
        self.policy = np.zeros(self.all_states.shape, dtype=np.int)
        self.state_values = np.zeros(self.all_states.shape)

    def _state_is_terminal(self, state):
        return state <= 1

    def _calculate_prob(self, state_prime, state):
        return state_prime/state

    def save(self):
        plt.clf()
        plt.plot(self.all_states, np.array(self.policy.copy()).flatten())
        save_path = f"visualizations/value_iterator/{self.movement_cost * self.N}_uniform.png"
        self._save(save_path)

    def _calculate_action_space(self, s):
        # Don't allow actions that would result in no change of the hypothesis space
        return np.linspace(1, int(s - 1), int(s - 1), dtype=np.int)

    def _get_new_states(self, actions, s):
        return actions, s - actions



if __name__ == "__main__":
    stop_error = 1
    sample_cost = 1
    movement_cost = 10
    N = 10
    kwargs = {
        'Ts': sample_cost,
        'Tt': movement_cost,
        'N': N,
    }


    agnt = UniformAgent(N, movement_cost=movement_cost)
    agnt.calculate_policy()
    agnt.save()
    flat_policy = np.array(agnt.policy).flatten()

    scorer().score(flat_policy, gym.make("change_point:uniform-v0", **kwargs), trials = 100000)
