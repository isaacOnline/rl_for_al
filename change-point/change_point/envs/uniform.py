from scipy.stats import uniform
from change_point.envs.change_point import ChangePoint
from gym.spaces import Discrete
import numpy as np

# Change point drawn from a uniform distribution here

class UniformCP(ChangePoint):
    def _initialize_distribution(self):
        self.dist = uniform(0, self.N)

    def _initialize_state(self):
        self.observation_space = Discrete(self.N + 1)

    def _update_state(self):
        length = self.max_loc - self.min_loc
        self.S = np.array(length)