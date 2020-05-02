from change_point.envs.change_point import ChangePoint
from scipy.stats import truncnorm
from gym.spaces import MultiDiscrete
import numpy as np
import tensorflow_probability as tfp


# The default distribution here is truncated normal

class NonUniformCP(ChangePoint):
    def _initialize_distribution(self, dist=None):
        if dist is None:
            # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            # for description of how truncnorm is being used
            min = 0
            max = self.N
            mean = self.N * 0.5
            sd = np.sqrt(self.N * 0.1)
            a = (min - mean) / sd
            b = (max - mean) / sd
            if self.tf:
                dist = tfp.distributions.TruncatedNormal(low=0, high=self.N, scale=sd, loc=mean)
            else:
                dist = truncnorm(a, b, loc=mean,scale=sd)
        self.dist = dist

    def _initialize_state(self):
        self.observation_space = MultiDiscrete([self.N+1, self.N+1])

    def _update_state(self):
        self.S = np.array([self.min_loc, self.max_loc])