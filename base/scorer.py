from abc import ABC, abstractmethod
from tqdm import tqdm

class Scorer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _standardize_observation(self, observation):
        pass

    @abstractmethod
    def _handle_action(self, action):
        pass

    @abstractmethod
    def _get_hspace_len(self, observation):
        pass

    @abstractmethod
    def return_values(self):
        pass

    def score(self, policy, env, trials = 10000):
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
        return observation

    def _handle_action(self, action):
        return int(action)

    def _get_hspace_len(self, observation):
        # observation is already hspace len
        return observation

    def return_values(self):
        ret = {
            'reward': self.avg_reward,
            'n_samples': self.avg_ns,
            'dist': self.avg_dist
        }
        return ret


class NonUniformScorer(Scorer):
    def _standardize_observation(self, observation):
        observation = (int(observation[0]), int(observation[1]))
        return observation

    def _handle_action(self, action):
        return int(action)

    def _get_hspace_len(self, observation):
        # hspace is between min and max search point
        return int(abs(observation[1] - observation[0]))

    def return_values(self):
        ret = {
            'reward': self.avg_reward,
            'n_samples': self.avg_ns,
            'dist': self.avg_dist
        }
        return ret


class RechargingScorer(Scorer):
    def __init__(self):
        self.total_num_recharges = 0
        Scorer.__init__(self)

    def _standardize_observation(self, observation):
        gamma = self.env.gamma
        battery_level = observation[2]
        battery_index = int(round(battery_level / gamma))
        observation = (int(observation[0]), int(observation[1]), battery_index)
        return observation

    def _handle_action(self, action):
        if action[1] > 0:
            self.total_num_recharges += 1
        return [int(a) for a in action]

    def _get_hspace_len(self, observation):
        # hspace is between min and max search point
        return int(abs(observation[1] - observation[0]))

    def return_values(self):
        self.avg_recharges = self.total_num_recharges / self.num_runs
        print(f'avg num_recharges: {self.avg_recharges}')
        ret = {
            'reward': self.avg_reward,
            'n_samples': self.avg_ns,
            'dist': self.avg_dist,
            'recharges': self.avg_recharges
        }
        return ret
