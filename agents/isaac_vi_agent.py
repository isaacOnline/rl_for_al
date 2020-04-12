import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from other.Scorer import scorer
import gym


class ViUniformAgent(object):
    def __init__(self, N, sample_cost = 1, movement_cost = 1000):
        self.N = N

        self.sample_cost = sample_cost
        self.movement_cost = movement_cost / N

        # Only let there be 1/self.stop_error + 1 number of states, since it doesn't make sense to have a state that is
        # smaller than the stop error. I'm still going to round the states, because otherwise the state space isn't
        # discrete.
        self.all_states = np.linspace(0, N, N + 1, dtype=np.int)

        self.policy = np.zeros(self.all_states.shape,dtype=np.int)

        self.state_values = np.zeros(self.all_states.shape)

    def _calculate_prob(self, state_prime, state, action):
        if state_prime not in (action, state - action):
            return 0
        else:
            return state_prime/state

    def _save(self):
        plt.clf()
        plt.plot(self.all_states, np.array(self.policy.copy()).flatten())
        plt.title(f"tt/ts: {self.movement_cost}/{self.sample_cost}, N: {self.N}")
        plt.xlabel("Size of Hypothesis Space")
        plt.xlim([0,self.N])
        plt.ylabel("Movement into Hypothesis Space")
        plt.ylim([0,self.N])
        plt.savefig("visualizations/isaac_ideal_policy.png")

    def calculate_policy(self):
        for s in self.all_states:
            if s <= 1:
                self.state_values[s] = 0
            else:
                # Don't allow actions that would result in no change of the hypothesis space
                tuples = pd.DataFrame({
                    "action": np.linspace(1, int(s - 1), int(s - 1), dtype=np.int)
                })
                tuples['reward'] = -self.sample_cost - self.movement_cost * tuples['action']
                tuples['new_state1'] = tuples['action']
                tuples['new_state2'] = s - tuples['action']
                tuples = pd.wide_to_long(tuples, stubnames='new_state',i=['action','reward'],j="which").reset_index()

                tuples['prob'] = tuples['new_state'] / s
                tuples['new_state_value'] = self.state_values[list(tuples['new_state'])]
                tuples['value'] = tuples['prob'] * (tuples['reward'] + tuples['new_state_value'])
                action_values = tuples.groupby("action").agg(np.sum)['value'].reset_index()
                best_index = np.argmax(action_values["value"])
                self.policy[s] = action_values["action"][best_index]
                self.state_values[s] = action_values["value"][best_index]


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


    agnt = ViUniformAgent(N, movement_cost=movement_cost)
    agnt.calculate_policy()
    agnt._save()
    flat_policy = np.array(agnt.policy).flatten()

    scorer().score(flat_policy, gym.make("change_point:uniform-v0", **kwargs))