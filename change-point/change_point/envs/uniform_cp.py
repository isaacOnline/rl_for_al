from scipy.stats import uniform
from change_point.envs.change_point import ChangePoint
from gym.spaces import Box
import numpy as np

# Change point drawn from a uniform distribution here

class UniformCP(ChangePoint):
    def _initialize_distribution(self, dist=None):
        if dist is not None:
            raise TypeError("Specifying a distribution is not possible for a uniform change point")
        self.dist = uniform(0, 1)

    def _initialize_state(self, delta):
        N = round(1/delta)
        self.observation_space = Box(low=0, high=1, shape=(1,),dtype =np.float64)

    def _update_state(self):
        length = self.max_loc - self.min_loc
        self.S = np.array(length)
        self.h_space_len = length
