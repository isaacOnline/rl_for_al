from change_point.envs.change_point import ChangePoint
from scipy.stats import truncnorm
from gym.spaces import MultiDiscrete
import numpy as np

# The default distribution here is truncated normal

class NonuniformCP(ChangePoint):
    def _initialize_distribution(self, dist=None):
        if dist is None:
            # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            # for description of how truncnorm is being used
            min = 0
            max = self.N
            mean = self.N * 0.5
            sd = self.N * 0.1
            a = (min - mean) / sd
            b = (max - mean) / sd
            self.dist = truncnorm(a, b, loc=mean,scale=sd)

    def _initialize_state(self):
        self.observation_space = MultiDiscrete([self.N+1, self.N+1])

    def _update_state(self):
        self.S = np.array([self.min_loc, self.max_loc])