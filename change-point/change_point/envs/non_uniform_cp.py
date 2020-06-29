from change_point.envs.change_point import ChangePoint
from scipy.stats import truncnorm
from gym.spaces import MultiDiscrete, Discrete
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

    def _set_action_space(self):
        self.action_space = Discrete(self.N)

    def _initialize_state(self):
        self.observation_space = MultiDiscrete([self.N + 1, self.N + 1])

    def _update_state(self):
        self.S = np.array([self.location, self.opposite_bound])
        self.h_space_len = self.max_loc - self.min_loc

    def _discrete_state(self):
        return np.array([int(round(self.location * self.N)), int(round(self.opposite_bound * self.N))])

    def _move_agent(self, portion: int):
        dist, mvmt = self.get_movement(portion)
        self._update_hist(dist)
        self.location += mvmt
        return dist

    def get_movement(self, portion):
        k = round(portion * self.h_space_len)
        distance = k * self.delta

        if np.isclose(distance, 0):
            distance += self.delta
        if np.isclose(distance, self.h_space_len):
            distance -= self.delta

        mvmt = distance * self.direction
        return distance, mvmt