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


    def _save_image(self, save_path):
        plt.title(f"tt/ts: {int(self.movement_cost * self.N)}/{self.sample_cost}, N: {self.N}")
        plt.xlabel("Size of Hypothesis Space")
        plt.xlim([0,self.N])
        plt.ylabel("Movement into Hypothesis Space")
        plt.ylim([0,self.N])
        plt.savefig(save_path)


    @abstractmethod
    def calculate_policy(self):
        pass


    @abstractmethod
    def save(self):
        pass
