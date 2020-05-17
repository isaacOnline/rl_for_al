from scipy.stats import uniform
from change_point.envs.change_point import ChangePoint
from gym.spaces import Discrete
import numpy as np

# Change point drawn from a uniform distribution here

class UniformCP(ChangePoint):
    def _initialize_distribution(self, dist=None):
        if dist is not None:
            raise TypeError("Specifying a distribution is not possible for a uniform change point")
        self.dist = uniform(0, 1)

    def _initialize_state(self):
        self.observation_space = Discrete(self.N + 1)

    def _update_state(self):
        length = self.max_loc - self.min_loc
        self.S = np.array(length)
        self.h_space_len = length

    def _discrete_state(self):
        """
        Return i = S * N, since discrete values must be returned as ints
        """
        return np.array(int(round(self.S * self.N)))