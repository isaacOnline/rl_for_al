from change_point.envs.change_point import ChangePoint
from gym.spaces import Discrete
import numpy as np
from base.distributions import get_unif

class UniformCP(ChangePoint):
    def _initialize_distribution(self, dist=None):
        """
        Create the distribution from which to draw change points.

        :param dist:
        :return:
        """
        if dist is not None:
            raise TypeError("Specifying a distribution is not possible for a uniform change point")
        self.dist = get_unif()

    def _initialize_state(self):
        """
        Set the observation space to have one dimensions, the length of the current hypothesis space

        :return:
        """
        self.observation_space = Discrete(self.N + 1)

    def _set_action_space(self):
        """
        Set the action space to have one dimension, which translates to the movement into the hypothesis space.

        :return:
        """
        self.action_space = Discrete(self.N)

    def _update_state(self):
        """
        Update the internally-stored state and hypothesis space

        :return:
        """
        length = self.max_loc - self.min_loc
        self.S = np.array(length)
        self.h_space_len = length

    def _discrete_state(self):
        """
        Return the state as an array with an integer inside. The integer is i = S * N, from 3.1 of the sps paper.

        :return:
        """
        return np.array(int(round(self.S * self.N)))

    def _move_agent(self, portion: int):
        """
        Change the agent's position acording to the given action

        :param action:
        :return:
        """
        dist, mvmt = self.get_movement(portion)
        self._update_hist(dist)
        self.location += mvmt
        return dist

    def get_movement(self, portion):
        """
        Translate the action into an actual distance to travel on the range [0,1]

        :param action:
        :return:
        """
        k = round(portion * self.h_space_len)
        distance = k * self.delta

        if np.isclose(distance, 0):
            distance += self.delta
        if np.isclose(distance, self.h_space_len):
            distance -= self.delta

        mvmt = distance * self.direction
        return distance, mvmt
