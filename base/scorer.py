import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
import tensorflow as tf

class Scorer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _standardize_observation(self, observation):
        pass

    @abstractmethod
    def _get_hspace_len(self, observation):
        pass

    def score(self, policy, env, trials = 10000):
        observation = env.reset()
        total_dist = this_round_dist = 0
        total_num_samples = this_round_num_samples = 0
        total_reward = this_round_reward = 0
        num_runs = 0
        N = len(policy) - 1
        delta = 1/N
        with tqdm(total = trials) as pbar:
            while num_runs < trials:
                observation = self._standardize_observation(observation)

                index = int(round(observation/delta)) # the policy is stored sequentially, so just finding how many deltas make up the current obs
                action = int(policy[index])
                true_mvmt = env.get_movement(action)[0]
                #assert np.close(true_mvmt env.get_movement(mvmt)[0])
                observation, reward, done, _ = env.step(action)
                this_round_dist += true_mvmt
                this_round_num_samples += 1
                this_round_reward += reward
                if done:
                    num_runs += 1
                    total_dist += this_round_dist
                    total_num_samples += this_round_num_samples
                    total_reward += this_round_reward
                    observation = env.reset()

                    # reset things we're keeping track of
                    this_round_dist = 0
                    this_round_num_samples = 0
                    this_round_reward = 0
                    pbar.update()

        avg_reward = total_reward / num_runs
        avg_ns = total_num_samples / num_runs
        avg_dist = total_dist / num_runs
        print(f'num runs: {num_runs}')
        print(f'avg reward: {avg_reward}')
        print(f'avg ns: {avg_ns}')
        print(f'avg dist: {avg_dist}')
        return avg_reward, avg_ns, avg_dist


class UniformScorer(Scorer):
    def _standardize_observation(self, observation):
        return observation

    def _get_hspace_len(self, observation):
        # observation is already hspace len
        return observation


class NonUniformScorer(Scorer):
    def _standardize_observation(self, observation):
        observation = (int(observation[0]), int(observation[1]))
        return observation

    def _get_hspace_len(self, observation):
        # hspace is between min and max search point
        return int(abs(observation[1] - observation[0]))
