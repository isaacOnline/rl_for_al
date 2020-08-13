import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from agents import UniformAgent
from base.scorer import UniformScorer
from experiments.vi_vs_sb.model_runner import ModelRunner


class UniformRunner(ModelRunner):
    def __init__(self, model_name, nsteps, env_params=None):
        self.vi_model = UniformAgent
        self.scorer = UniformScorer
        self.env_name = "uniform"
        self.dist_name = "uniform"
        ModelRunner.__init__(self, model_name, nsteps, env_params)

    def _set_policy_path(self):
        """
        Save the path where the value iteration policy will be saved to or loaded from

        :return:
        """
        self.policy_path = f"experiments/vi_vs_sb/uniform/vi_policies/{self.params['movement_cost']}_" \
                           f"{round(1/self.params['delta'])}_" \
                           f"{self.env_name}.csv"
        self._vi_policy_load = np.genfromtxt

    def get_sb_policy(self):
        """
        Harvest the policy from the stable baselines agent.

        :return:
        """

        policy = []
        for i in range(self.N + 1):
            obs = np.array(i)
            obs.shape = (1,)
            action = np.argmax(self.sb_model.action_probability(obs))
            policy.append(action)
        policy = np.array(policy)
        return policy

    def plot(self):
        """
        Create a plot comparing performance/policy between stable baselines and the value iterator

        :return:
        """
        plt.clf()

        plt.plot(np.linspace(0, 1, self.N+1), self.sb_policy, c='tab:orange', label=f"{self.model_name} Learner")
        plt.plot(np.linspace(0, 1, self.N+1), self.vi_policy, c='B', label="Value Iteration")

        plt.title(
            f"ACER Train Time: {self.sb_train_time}, VI Train Time: {self.vi_train_time}\n"
            f"tt/ts: {self.params['movement_cost']}/{self.params['sample_cost']}, "
            f"N: {int(1/self.params['delta'])}")
        plt.xlabel("Size of Hypothesis Space")
        plt.ylabel("Gym Policy")
        plt.legend()

        plt.xlim((0, 1))
        plt.ylim((0, self.N))

        plt.savefig(self.img_path)

    def _save_performance(self):
        """
        Save the results of the experiment to the performance log

        :return:
        """

        line = pd.DataFrame({
            "id": self.id,
            "time_steps": [self.nsteps],
            "vi_train_time": [self.vi_train_time],
            "sb_train_time": [self.sb_train_time],
            "model": [self.model_name],
            'Ts': self.params['sample_cost'],
            'Tt': self.params['movement_cost'],
            'vi_reward': [self.vi_performance['reward']],
            'rl_reward': [self.sb_performance['reward']],
            'vi_ns': [self.vi_performance['n_samples']],
            'rl_ns': [self.sb_performance['n_samples']],
            'vi_distance': [self.vi_performance['dist']],
            'rl_distance': [self.sb_performance['dist']],
            'N': self.N
        })
        line.to_csv(self.performance_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    model = "ACER"
    nsteps = 10000
    N = 1000
    Tts = np.array([10, 750, 500, 400, 300, 200, 50, 1])
    for Tt in Tts:
        kwargs = {
            'sample_cost': 1,
            'movement_cost': Tt,
            'delta': 1/N
        }

        runner = UniformRunner(model, nsteps, kwargs)
        runner.train_vi()
        runner.score_vi()

        runner.train_sb()
        print(f"N: {N} NSTEPS: {nsteps} Tt: {kwargs['movement_cost']}")
        print("{}:".format(model))
        runner.score_sb()

        runner.save()