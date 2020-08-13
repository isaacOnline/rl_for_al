from change_point.envs.change_point import ChangePoint
from base.distributions import get_truncnorm
from gym.spaces import MultiDiscrete, Discrete
import numpy as np

class NonUniformCP(ChangePoint):
    def _initialize_distribution(self, dist=None):
        """
        Create the distribution from which to draw change points.

        :param dist:
        :return:
        """
        if dist is None:
            dist = get_truncnorm()
        self.dist = dist

    def _set_action_space(self):
        """
        Set the action space to have one dimension, which translates to the movement into the hypothesis space.

        :return:
        """
        self.action_space = Discrete(self.N)

    def _initialize_state(self):
        """
        Set the observation space to have two dimensions, one for the current location and the other for the opposite
        end.

        :return:
        """
        self.observation_space = MultiDiscrete([self.N + 1, self.N + 1])

    def _update_state(self):
        """
        Update the internally-stored state and hypothesis space

        :return:
        """
        self.S = np.array([self.location, self.opposite_bound])
        self.h_space_len = self.max_loc - self.min_loc

    def _discrete_state(self):
        """
        Return the state as an array with two integers inside. The integers are i and j, from 3.2 of the sps paper.

        :return:
        """
        return np.array([int(round(self.location * self.N)), int(round(self.opposite_bound * self.N))])

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
