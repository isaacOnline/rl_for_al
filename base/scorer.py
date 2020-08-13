from abc import ABC, abstractmethod
from tqdm import tqdm

class Scorer(ABC):
    """
    Base class for getting performance of a policy on one of the change point environments.
    """
    def __init__(self):
        pass

    @abstractmethod
    def _standardize_observation(self, observation):
        """
        Format the state output by the gym so that it can be used to slice the policy

        :param observation: output from environment
        :return:
        """
        pass

    @abstractmethod
    def _handle_action(self, action):
        """
        Format the action so that it can be fed into the gym

        :param action: The action stored in the policy array and sliced using the state
        :return:
        """
        pass

    @abstractmethod
    def _get_hspace_len(self, observation):
        """
        Get length of the hypothesis space.

        :param observation:
        :return:
        """
        pass

    @abstractmethod
    def return_values(self):
        """
        Return a dictionary containing the performance of the model

        :return:
        """
        pass

    def score(self, policy, env, trials = 10000):
        """
        Get the average reward of a policy on the environment. The policy should be an array, whose actions can be
        accessed via the state. I.e. policy[state] should return the action for that state, to be fed into the
        gym environment

        Saves performance, which can later be accessed with the return_values method.

        :param policy: array containing the policy
        :param env: gym environment to run
        :param trials: number of times to run policy
        :return:
        """
        self.env = env
        observation = env.reset()
        total_dist = 0
        total_num_samples = 0
        total_reward = 0
        num_runs = 0
        with tqdm(total = trials) as pbar:
            while num_runs < trials:
                observation = self._standardize_observation(observation)
                action = self._handle_action(policy[observation])
                dist, _ = env.get_movement(action)

                observation, reward, done, _ = env.step(action)
                total_dist += dist
                total_num_samples += 1
                total_reward += reward
                if done:
                    num_runs += 1
                    observation = env.reset()
                    pbar.update()

        self.num_runs = num_runs
        self.avg_reward = total_reward / num_runs
        self.avg_ns = total_num_samples / num_runs
        self.avg_dist = total_dist / num_runs
        print(f'num runs: {num_runs}')
        print(f'avg reward: {self.avg_reward}')
        print(f'avg ns: {self.avg_ns}')
        print(f'avg dist: {self.avg_dist}')
        return self.return_values()


class UniformScorer(Scorer):
    def _standardize_observation(self, observation):
        """
        Format the state output by the gym so that it can be used to slice the policy

        :param observation: output from environment
        :return:
        """
        return observation

    def _handle_action(self, action):
        """
        Format the action so that it can be fed into the gym

        :param action: The action stored in the policy array and sliced using the state
        :return:
        """
        return int(action)

    def _get_hspace_len(self, observation):
        """
         Get length of the hypothesis space.

         :param observation:
         :return:
         """
        # observation is already hspace len
        return observation

    def return_values(self):
        """
        Return a dictionary containing the performance of the model

        :return:
        """
        ret = {
            'reward': self.avg_reward,
            'n_samples': self.avg_ns,
            'dist': self.avg_dist
        }
        return ret


class NonUniformScorer(Scorer):
    def _standardize_observation(self, observation):
        """
        Format the state output by the gym so that it can be used to slice the policy

        :param observation: output from environment
        :return:
        """
        observation = (int(observation[0]), int(observation[1]))
        return observation

    def _handle_action(self, action):
        """
        Format the action so that it can be fed into the gym

        :param action: The action stored in the policy array and sliced using the state
        :return:
        """
        return int(action)

    def _get_hspace_len(self, observation):
        """
         Get length of the hypothesis space.

         :param observation:
         :return:
         """
        # hspace is between min and max search point
        return int(abs(observation[1] - observation[0]))

    def return_values(self):
        """
        Return a dictionary containing the performance of the model

        :return:
        """
        ret = {
            'reward': self.avg_reward,
            'n_samples': self.avg_ns,
            'dist': self.avg_dist
        }
        return ret


class RechargingScorer(Scorer):
    def __init__(self):
        """
        Save the total number of recharges.
        """
        self.total_num_recharges = 0
        Scorer.__init__(self)

    def _standardize_observation(self, observation):
        """
        Format the state output by the gym so that it can be used to slice the policy.

        The battery level is in the range [0, battery_capacity]

        :param observation: output from environment
        :return:
        """
        battery_level = observation[2]
        battery_index = int(round(battery_level / self.env.gamma)) # TODO: Is this being handled correctly?
        observation = (int(observation[0]), int(observation[1]), battery_index)
        return observation

    def _handle_action(self, action):
        """
        Format the action so that it can be fed into the gym.

        Also, increment the number of recharges, if a recharge has occured

        :param action: The action stored in the policy array and sliced using the state
        :return:
        """
        if action[1] > 0:
            self.total_num_recharges += 1
        return [int(a) for a in action]

    def _get_hspace_len(self, observation):
        """
         Get length of the hypothesis space.

         :param observation:
         :return:
         """
        # hspace is between min and max search point
        return int(abs(observation[1] - observation[0]))

    def return_values(self):
        """
        Return a dictionary containing the performance of the model

        :return:
        """
        self.avg_recharges = self.total_num_recharges / self.num_runs
        print(f'avg num_recharges: {self.avg_recharges}')
        ret = {
            'reward': self.avg_reward,
            'n_samples': self.avg_ns,
            'dist': self.avg_dist,
            'recharges': self.avg_recharges
        }
        return ret
