from change_point.envs.change_point import ChangePoint
from scipy.stats import truncnorm
from gym.spaces import Box
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
        self.observation_space = Box(low=0, high=1, shape=(1,2),dtype =np.float64)

    def _update_state(self):
        self.S = np.array([self.location, self.opposite_bound])
        self.h_space_len = self.max_loc - self.min_loc