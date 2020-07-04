import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import truncnorm, uniform

from agents import NonUniformAgent
from base.scorer import NonUniformScorer
from experiments.vi_vs_rl.model_runner import ModelRunner

class NonUniformRunner(ModelRunner):
    def __init__(self, model_name, nsteps, recalculate_vi=False, env_params=None):
        self.agent = NonUniformAgent
        self.scorer = NonUniformScorer
        self.env_name = "non_uniform"
        self.dist = env_params['dist']
        self.dist_name = self.dist.dist.name
        ModelRunner.__init__(self, model_name, nsteps, recalculate_vi, env_params)

    def _set_policy_path(self):
        self.policy_path = f"experiments/vi_vs_rl/non_uniform/vi_policies/{int(self.params['movement_cost'])}_{self.N}_{self.dist_name}.csv"
        self._vi_policy_load = np.genfromtxt

    def get_rl_policy(self):
        policy = np.zeros((self.N+1, self.N+1), dtype=np.int)
        for first_end in range(0, self.N + 1):
            for second_end in range(0, self.N + 1):
                obs = np.array([first_end, second_end])
                # self.model.action_probability returns the stochastic policy, so we just find the most likely action
                action = np.argmax(self.model.action_probability(obs))
                policy[first_end, second_end] = action
        return policy

    def plot(self, rl_policy, vi_policy):
        plt.clf()

        fig, (vi_ax, rl_ax,cbar_ax) = plt.subplots(1,3,gridspec_kw={'width_ratios': [20,20, 1]})
        vi_im = vi_ax.imshow(vi_policy, vmin = 0, vmax = self.N) #TODO: Does this need to be changed from self.N to 1?
        rl_im = rl_ax.imshow(rl_policy, vmin = 0, vmax = self.N)
        cbar_im = fig.colorbar(rl_im, ax=rl_ax, cax=cbar_ax)
        cbar_im.ax.set_ylabel("Movement On Line", rotation=-90, va="bottom")

        vi_ax.set_title(f"VI Policy\nTrain Time: {self.vi_train_time}\nAvg Reward: {round(self.vi_performance['reward'],2)}")
        rl_ax.set_title(f"ACER Policy\nTrain Time: {self.rl_train_time}\nAvg Reward: {round(self.rl_performance['reward'],2)}")

        plt.suptitle(
            f"tt/ts: {self.params['movement_cost']}/{self.params['sample_cost']}\n"
            f"N: {self.N}\n"
            f"Dist: {self.dist_name}")

        plt.savefig(self.img_path)

    def save_performance(self, rl_policy, vi_policy):
        print(f"N: {self.N} NSTEPS: {self.nsteps} Tt: {self.params['movement_cost']}")
        print("{}:".format(self.model_name))
        self.rl_performance = self.scorer().score(rl_policy, self.env)

        line = pd.DataFrame({
            "id": self.id,
            'N': self.N,
            "time_steps": [self.nsteps],
            "vi_train_time":[self.vi_train_time],
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
            'distribution': [self.dist_name]
        })
        line.to_csv(self.performance_path, mode='a', header=False, index=False)

def get_truncnorm():
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    # for description of how truncnorm is being used
    min = 0
    max = 1
    mean = 0.5
    sd = np.sqrt(0.1)
    a = (min - mean) / sd
    b = (max - mean) / sd
    dist = truncnorm(a, b, loc=mean, scale=sd)
    return dist

def get_unif():
    return uniform(0, 1)

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

        runner = NonUniformRunner(model, nsteps, False, env_params=kwargs)
        runner.train(use_callback=True)
        runner.save()