import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import truncnorm, uniform

from agents import NonUniformAgent
from base.scorer import NonUniformScorer
from experiments.vi_vs_rl.model_runner import ModelRunner


class NonUniformRunner(ModelRunner):
    def __init__(self, model_name, nsteps, env_params):
        self.env_name = "non_uniform"
        self.dist = env_params['dist']
        self.dist_name = self.dist.dist.name
        ModelRunner.__init__(self, model_name, nsteps, env_params)

    def get_vi_policy(self):
        try:
            policy = np.genfromtxt(
                f"experiments/vi_vs_rl/non_uniform/vi_policies/{int(self.params['movement_cost'])}_{self.params['N']}_{self.dist_name}.csv")
        except:
            agnt = NonUniformAgent(sample_cost=self.params['sample_cost'],
                                   movement_cost=self.params['movement_cost'],
                                   N=self.params['N'],
                                   dist=self.dist)
            agnt.save()
            policy = agnt.policy
        return policy

    def get_rl_policy(self):
        policy = np.zeros((N+1, N+1), dtype=np.int)
        for first_end in range(0, self.params['N'] + 1):
            for second_end in range(0, self.params['N'] + 1):
                obs = np.array([first_end, second_end])
                # self.model.action_probability returns the stochastic policy, so we just find the most likely action
                action = np.argmax(self.model.action_probability(obs))
                mvmt, _ = self.env.get_movement(obs, self.params['N'], action)
                policy[first_end, second_end] = mvmt
        return policy

    def plot(self, rl_policy, vi_policy, run_time):
        plt.clf()

        difference = rl_policy - vi_policy

        fig, ax = plt.subplots()
        im = ax.imshow(difference)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(f"{self.model_name} Policy - Value Iteration Policy", rotation=-90, va="bottom")


        plt.suptitle(
            f"Train Time: {run_time} "
            f"tt/ts: {self.params['movement_cost']}/{self.params['sample_cost']} "
            f"N: {self.params['N']} "
            f"Dist: {self.dist_name}")

        plt.savefig(self.img_path)

    def save_performance(self, rl_policy, vi_policy, run_time):
        print("{}:".format(self.model_name))
        rl_reward, rl_ns, rl_dist = NonUniformScorer().score(rl_policy, self.env)

        print("\nValue Iteration:")
        vi_reward, vi_ns, vi_dist = NonUniformScorer().score(vi_policy, self.env)

        line = pd.DataFrame({
            "id": self.id,
            'N': self.params['N'],
            "time_steps": [self.nsteps],
            "train_time": [run_time],
            "model": [self.model_name],
            'Ts': self.params['sample_cost'],
            'Tt': self.params['movement_cost'],
            'vi_reward': [vi_reward],
            'rl_reward': [rl_reward],
            'vi_ns': [vi_ns],
            'rl_ns': [rl_ns],
            'vi_distance': [vi_dist],
            'rl_distance': [rl_dist],
            'distribution': [self.dist_name]
        })
        line.to_csv(f"experiments/vi_vs_rl/non_uniform/non_uniform_performance.csv", mode='a', header=False, index=False)

def get_truncnorm(N):
    min = 0
    max = N
    mean = N * 0.5
    sd = N * 0.1
    a = (min - mean) / sd
    b = (max - mean) / sd
    dist = truncnorm(a, b, loc=mean, scale=sd)
    return dist

def get_unif(N):
    return uniform(0, N)

if __name__ == "__main__":
    model = "ACER"
    nsteps = 15000
    N = 30
    for Tt in [1]:
        kwargs = {
            'sample_cost': 1,
            'movement_cost': Tt,
            'N': N,
            'dist': get_truncnorm(N)
        }

        runner = NonUniformRunner(model, nsteps, kwargs)
        runner.train()
        runner.save()
    # nsteps *= 2
