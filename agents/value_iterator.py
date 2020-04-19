from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class ValueIterator(object, metaclass=ABCMeta):
    def __init__(self, N, sample_cost = 1, movement_cost = 1000):
        self.N = N

        self.sample_cost = sample_cost
        self.movement_cost = movement_cost / N


    def calculate_policy(self):
        for s in tqdm(self.all_states):
            if self._state_is_terminal(s):
                self.state_values[s] = 0
            else:
                actions = pd.DataFrame({
                    "action": self._calculate_action_space(s)
                })
                actions['reward'] = -self.sample_cost - self.movement_cost * np.abs(actions['action'])

                possible_new_states = self._get_new_states(actions['action'], s)
                actions['new_state1'] = possible_new_states[0]
                actions['new_state2'] = possible_new_states[1]

                actions = pd.wide_to_long(actions, stubnames='new_state', i=['action', 'reward'],
                                          j="which").reset_index()

                actions['prob'] = self._calculate_prob(actions['new_state'], s)
                actions['new_state_value'] = [self.state_values[s_prime] for s_prime in actions['new_state']]
                actions['value'] = actions['prob'] * (actions['reward'] + actions['new_state_value'])
                action_values = actions.groupby("action").agg(np.sum)['value'].reset_index()
                best_index = np.argmax(action_values["value"])
                self.policy[s] = action_values["action"][best_index]
                self.state_values[s] = action_values["value"][best_index]

    def _save_image(self, save_path):
        plt.title(f"tt/ts: {int(self.movement_cost * self.N)}/{self.sample_cost}, N: {self.N}")
        plt.xlabel("Size of Hypothesis Space")
        plt.xlim([0,self.N])
        plt.ylabel("Movement into Hypothesis Space")
        plt.ylim([0,self.N])
        plt.savefig(save_path)


    @abstractmethod
    def _state_is_terminal(self, state):
        pass

    @abstractmethod
    def _calculate_prob(self, state_prime, state):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def _calculate_action_space(self, s):
        pass

    @abstractmethod
    def _get_new_states(self, actions, s):
        pass
