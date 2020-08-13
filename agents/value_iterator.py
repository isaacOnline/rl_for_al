from abc import ABCMeta, abstractmethod


class ValueIterator(object, metaclass=ABCMeta):
    """
    This is the base class for the UniformAgent, NonUniformAgent, and RechargingAgent. It just saves the
    common params
    """
    def __init__(self, delta, epsilon, sample_cost = 1, movement_cost = 1000):
        if epsilon is None:
            self.epsilon = delta
        else:
            self.epsilon = epsilon

        self.N = round(1/delta)
        self.delta = delta

        self.sample_cost = sample_cost
        self.movement_cost = movement_cost

    @abstractmethod
    def calculate_policy(self):
        pass


    @abstractmethod
    def save(self):
        pass
