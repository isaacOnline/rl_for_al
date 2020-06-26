from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt


class ValueIterator(object, metaclass=ABCMeta):
    def __init__(self, delta, epsilon, sample_cost = 1, movement_cost = 1000):
        if epsilon is None:
            self.epsilon = delta
        else:
            self.epsilon = epsilon

        self.N = round(1/delta)
        self.delta = delta

        self.sample_cost = sample_cost
        self.movement_cost = movement_cost


    def _save_image(self, save_path):
        plt.title(f"tt/ts: {self.movement_cost}/{self.sample_cost}, N: {self.N}")
        plt.xlabel("Size of Hypothesis Space")
        plt.xlim([0,1])
        plt.ylabel("Movement into Hypothesis Space")
        plt.ylim([0,1])
        plt.savefig(save_path)


    @abstractmethod
    def calculate_policy(self):
        pass


    @abstractmethod
    def save(self):
        pass
