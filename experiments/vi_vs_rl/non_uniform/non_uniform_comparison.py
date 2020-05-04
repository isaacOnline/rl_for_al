import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import truncnorm, uniform

from agents import NonUniformAgent
from base.scorer import NonUniformScorer
from experiments.vi_vs_rl.model_runner import ModelRunner
import tensorflow_probability as tfp

class NonUniformRunner(ModelRunner):
    def __init__(self, model_name, nsteps, env_params):
        self.env_name = "non_uniform"
        self.dist = env_params['dist']
        self.dist_name = self.dist.dist.name
        ModelRunner.__init__(self, model_name, nsteps, env_params)

    def get_vi_policy(self, recalculate):
        # if we want to recalculate, do so, otherwise try to access a stored policy
        if recalculate:
            agnt = NonUniformAgent(sample_cost=self.params['sample_cost'],
                                   movement_cost=self.params['movement_cost'],
                                   N=self.params['N'],
                                   dist=self.dist)
            self.vi_train_time = agnt.calculate_policy()
            agnt.save()
            policy = agnt.policy
        else:
            try:
                policy = np.genfromtxt(
                    f"experiments/vi_vs_rl/non_uniform/vi_policies/{int(self.params['movement_cost'])}_{self.params['N']}_{self.dist_name}.csv")
                self.vi_train_time = "Not Calculated"
            except:
                agnt = NonUniformAgent(**self.params)
                self.vi_train_time = agnt.calculate_policy()
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

    def plot(self, rl_policy, vi_policy):
        plt.clf()

        fig, (vi_ax, rl_ax,cbar_ax) = plt.subplots(1,3,gridspec_kw={'width_ratios': [20,20, 1]})
        vi_im = vi_ax.imshow(vi_policy, vmin = 0, vmax = self.params['N'])
        rl_im = rl_ax.imshow(rl_policy, vmin = 0, vmax = self.params['N'])
        cbar_im = fig.colorbar(rl_im, ax=rl_ax, cax=cbar_ax)
        cbar_im.ax.set_ylabel("Movement On Line", rotation=-90, va="bottom")

        vi_ax.set_title(f"VI Policy\nTrain Time: {self.vi_train_time}\nAvg Reward: {round(self.vi_reward,2)}")
        rl_ax.set_title(f"ACER Policy\nTrain Time: {self.rl_train_time}\nAvg Reward: {round(self.rl_reward,2)}")

        plt.suptitle(
            f"tt/ts: {self.params['movement_cost']}/{self.params['sample_cost']}\n"
            f"N: {self.params['N']}\n"
            f"Dist: {self.dist_name}")

        plt.savefig(self.img_path)

    def save_performance(self, rl_policy, vi_policy):
        print(f"N: {self.params['N']} NSTEPS: {self.nsteps} Tt: {self.params['movement_cost']}")
        print("{}:".format(self.model_name))
        dist = tfp.distributions.TruncatedNormal(low=0, high=self.params['N'], scale=np.sqrt(self.params['N']*0.1), loc=self.params['N']/2)
        test_params = self.params
        test_params['dist']=dist
        test_params['tf']=True
        env = gym.make( f"change_point:{self.env_name}-v0", **test_params)
        self.rl_reward, rl_ns, rl_dist = NonUniformScorer().score(rl_policy, env)

        print("\nValue Iteration:")
        self.vi_reward, vi_ns, vi_dist = NonUniformScorer().score(vi_policy, env)

        line = pd.DataFrame({
            "id": self.id,
            'N': self.params['N'],
            "time_steps": [self.nsteps],
            "vi_train_time":[self.vi_train_time],
            "rl_train_time": [self.rl_train_time],
            "model": [self.model_name],
            'Ts': self.params['sample_cost'],
            'Tt': self.params['movement_cost'],
            'vi_reward': [self.vi_reward],
            'rl_reward': [self.rl_reward],
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
    paper_var = 0.1
    paper_sd = np.sqrt(paper_var)
    this_dist_sd = paper_sd * N
    a = (min - mean) / this_dist_sd
    b = (max - mean) / this_dist_sd
    dist = truncnorm(a, b, loc=mean, scale=this_dist_sd)
    return dist

def get_unif(N):
    return uniform(0, N)

if __name__ == "__main__":
    model = "ACER"
    nsteps = 1000000
    N = 300
    for Tt in [1, 10, 50, 100, 250, 500, 750, 1000]:
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
