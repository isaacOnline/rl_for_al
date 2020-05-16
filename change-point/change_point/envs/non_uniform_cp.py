from change_point.envs.change_point import ChangePoint
from scipy.stats import truncnorm
from gym.spaces import MultiDiscrete
import numpy as np

# The default distribution here is truncated normal

class NonUniformCP(ChangePoint):
    def _initialize_distribution(self, dist=None):
        if dist is None:
            # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            # for description of how truncnorm is being used
            min = 0
            max = 1
            mean = 0.5
            sd = np.sqrt(0.1)
            a = (min - mean) / sd
            b = (max - mean) / sd
            dist = truncnorm(a, b, loc=mean,scale=sd)
        self.dist = dist

    def _initialize_state(self, delta):
        N = round(1/delta)
        self.observation_space = MultiDiscrete([N+1,N+1])

    def _update_state(self):
        self.S = np.array([self.location, self.opposite_bound])
        self.h_space_len = self.max_loc - self.min_loc

    def _discrete_state(self):
        return np.array([int(self.location * self.N), int(self.opposite_bound * self.N)])