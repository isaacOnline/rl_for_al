from change_point.envs.change_point import ChangePoint
from scipy.stats import truncnorm
from gym.spaces import MultiDiscrete
import numpy as np

# This is the class for a change point problem where the
# change point is drawn from a truncated normal distribution

# See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
# for description of how truncnorm is being used

class TruncNormCP(ChangePoint):
    def _initialize_distribution(self):
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