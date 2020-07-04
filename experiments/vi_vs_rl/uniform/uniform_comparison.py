import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from agents import UniformAgent
from base.scorer import UniformScorer
from experiments.vi_vs_rl.model_runner import ModelRunner


class UniformRunner(ModelRunner):
    def __init__(self, model_name, nsteps, recalculate_vi=False, env_params=None):
        self.agent = UniformAgent
        self.scorer = UniformScorer
        self.env_name = "uniform"
        self.dist_name = "uniform"
        ModelRunner.__init__(self, model_name, nsteps, recalculate_vi, env_params)

    def _set_policy_path(self):
        self.policy_path = f"experiments/vi_vs_rl/uniform/vi_policies/{self.params['movement_cost']}_" \
                           f"{round(1/self.params['delta'])}_" \
                           f"{self.env_name}.csv"
        self._vi_policy_load = np.genfromtxt

    def get_rl_policy(self):
        policy = []
        for i in range(self.N + 1):
            obs = np.array(i)
            obs.shape = (1,)
            action = np.argmax(self.model.action_probability(obs))
            policy.append(action)
        policy = np.array(policy)
        return policy

    def plot(self, rl_policy, vi_policy):
        plt.clf()

        plt.plot(np.linspace(0, 1, self.N+1), rl_policy, c='tab:orange', label=f"{self.model_name} Learner")
        plt.plot(np.linspace(0, 1, self.N+1), vi_policy, c='B', label="Value Iteration")

        plt.title(
            f"ACER Train Time: {self.rl_train_time}, VI Train Time: {self.vi_train_time}\n"
            f"tt/ts: {self.params['movement_cost']}/{self.params['sample_cost']}, "
            f"N: {int(1/self.params['delta'])}")
        plt.xlabel("Size of Hypothesis Space")
        plt.ylabel("Gym Policy")
        plt.legend()

        plt.xlim((0, 1))
        plt.ylim((0, self.N))

        plt.savefig(self.img_path)

    def save_performance(self, rl_policy, vi_policy):
        print(f"N: {self.N} NSTEPS: {self.nsteps} Tt: {self.params['movement_cost']}")
        print("{}:".format(self.model_name))
        self.rl_performance = self.scorer().score(rl_policy, self.env)

        line = pd.DataFrame({
            "id": self.id,
            "time_steps": [self.nsteps],
            "vi_train_time": [self.vi_train_time],
            "rl_train_time": [self.rl_train_time],
            "model": [self.model_name],
            'Ts': self.params['sample_cost'],
            'Tt': self.params['movement_cost'],
            'vi_reward': [self.vi_performance['reward']],
            'rl_reward': [self.rl_performance['reward']],
            'vi_ns': [self.vi_performance['n_samples']],
            'rl_ns': [self.rl_performance['n_samples']],
            'vi_distance': [self.vi_performance['dist']],
            'rl_distance': [self.rl_performance['dist']],
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

        runner = UniformRunner(model, nsteps, False, kwargs)
        runner.train(use_callback=True)
        runner.save()