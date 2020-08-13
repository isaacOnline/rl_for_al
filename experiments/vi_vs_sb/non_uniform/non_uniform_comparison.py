import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from base.distributions import get_unif, get_truncnorm

from agents import NonUniformAgent
from base.scorer import NonUniformScorer
from experiments.vi_vs_sb.model_runner import ModelRunner

class NonUniformRunner(ModelRunner):
    def __init__(self, model_name, nsteps, env_params=None):
        self.rl_model = NonUniformAgent
        self.scorer = NonUniformScorer
        self.env_name = "non_uniform"
        self.dist = env_params['dist']
        self.dist_name = self.dist.dist.name
        ModelRunner.__init__(self, model_name, nsteps, env_params)

    def _set_policy_path(self):
        """
        Save the path where the value iteration policy will be saved to or loaded from

        :return:
        """
        self.policy_path = f"experiments/vi_vs_sb/non_uniform/vi_policies/{int(self.params['movement_cost'])}_{self.N}_{self.dist_name}.csv"
        self._vi_policy_load = np.genfromtxt

    def get_sb_policy(self):
        """
        Harvest the policy from the stable baselines agent.

        :return:
        """
        policy = np.zeros((self.N+1, self.N+1), dtype=np.int)
        for first_end in range(0, self.N + 1):
            for second_end in range(0, self.N + 1):
                obs = np.array([first_end, second_end])
                # self.sb_model.action_probability returns the stochastic policy, so we just find the most likely action
                action = np.argmax(self.sb_model.action_probability(obs))
                policy[first_end, second_end] = action
        return policy

    def plot(self):
        """
        Create a plot comparing performance/policy between stable baselines and the value iterator

        :return:
        """
        plt.clf()

        fig, (vi_ax, rl_ax,cbar_ax) = plt.subplots(1,3,gridspec_kw={'width_ratios': [20,20, 1]})
        vi_im = vi_ax.imshow(self.vi_policy, vmin = 0, vmax = self.N) #TODO: Does this need to be changed from self.N to 1?
        rl_im = rl_ax.imshow(self.sb_policy, vmin = 0, vmax = self.N)
        cbar_im = fig.colorbar(rl_im, ax=rl_ax, cax=cbar_ax)
        cbar_im.ax.set_ylabel("Movement On Line", rotation=-90, va="bottom")

        vi_ax.set_title(f"VI Policy\nTrain Time: {self.vi_train_time}\nAvg Reward: {round(self.vi_performance['reward'],2)}")
        rl_ax.set_title(f"ACER Policy\nTrain Time: {self.sb_train_time}\nAvg Reward: {round(self.sb_performance['reward'], 2)}")

        plt.suptitle(
            f"tt/ts: {self.params['movement_cost']}/{self.params['sample_cost']}\n"
            f"N: {self.N}\n"
            f"Dist: {self.dist_name}")

        plt.savefig(self.img_path)

    def _save_performance(self):
        """
        Save the results of the experiment to the performance log

        :return:
        """
        line = pd.DataFrame({
            "id": self.id,
            'N': self.N,
            "time_steps": [self.nsteps],
            "vi_train_time":[self.vi_train_time],
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
            'distribution': [self.dist_name]
        })
        line.to_csv(self.performance_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    model = "ACER"
    nsteps = 500000
    N = 300
    delta = 1/N
    for Tt in [1, 1000, 750, 500, 250, 100, 50, 10]:

        kwargs = {
            'sample_cost': 1,
            'movement_cost': Tt,
            'delta': delta,
            'dist': get_truncnorm()
        }

        runner = NonUniformRunner(model, nsteps, kwargs)
        runner.load_vi()
        runner.score_vi()
        runner.train_sb(use_callback=True)

        print(f"N: {N} NSTEPS: {nsteps} Tt: {kwargs['movement_cost']}")
        print("{}:".format(model))
        runner.score_sb()

        runner.save()