import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from agents import UniformAgent
from base.scorer import UniformScorer
from experiments.vi_vs_rl.model_runner import ModelRunner


class UniformRunner(ModelRunner):
    def __init__(self, model_name, nsteps, env_params):
        self.env_name = "uniform"
        ModelRunner.__init__(self, model_name, nsteps, env_params)

    def get_vi_policy(self):
        try:
            policy = np.genfromtxt(
                f"experiments/vi_vs_rl/uniform/vi_policies/{self.params['movement_cost']}_{self.params['N']}_{self.env_name}.csv")
        except:
            agnt = UniformAgent(sample_cost=self.params['sample_cost'],
                                movement_cost=self.params['movement_cost'],
                                N=self.params['N'])
            agnt.save()
            policy = agnt.policy
        return policy

    def get_rl_policy(self):
        policy = []
        for i in range(0, self.params['N'] + 1):
            obs = np.array(i)
            action = np.argmax(self.model.action_probability(obs))
            mvmt, _ = self.env.get_movement(obs, self.params['N'], action)
            policy.append([obs, mvmt])
        policy = np.array(policy)
        return policy

    def plot(self, rl_policy, vi_policy, run_time):
        plt.clf()

        plt.plot(rl_policy[:, 0], rl_policy[:, 1], c='tab:orange', label=f"{self.model_name} Learner")
        plt.plot(rl_policy[:, 0], vi_policy, c='B', label="Value Iteration")

        plt.title(
            f"Train Time: {run_time}, tt/ts: {self.params['movement_cost']}/{self.params['sample_cost']}, N: {self.params['N']}")
        plt.xlabel("Size of Hypothesis Space")
        plt.ylabel("Movement into Hypothesis Space")
        plt.legend()

        plt.ylim((0, self.params['N']))
        plt.xlim((0, self.params['N']))

        plt.savefig(self.img_path)

    def save_performance(self, rl_policy, vi_policy, run_time):
        rl_fout = rl_policy[:, 1]
        print("{}:".format(self.model_name))
        rl_reward, rl_ns, rl_dist = UniformScorer().score(rl_fout, self.env)

        print("\nValue Iteration:")
        vi_reward, vi_ns, vi_dist = UniformScorer().score(vi_policy, self.env)

        line = pd.DataFrame({
            "id": self.id,
            "time_steps": [self.nsteps],
            "train_time": [run_time],
            "model": [self.model_name],
            'Ts': self.params['sample_cost'],
            'Tt': self.params['movement_cost'],
            'vi_reward': [vi_reward],
            'rl_reward': [rl_reward],
            'vi_ns': [vi_ns],
            'rl_ns': [rl_ns],
            'vi_dist': [vi_dist],
            'rl_dist': [rl_dist],
            'N': self.params['N']
        })
        line.to_csv(f"experiments/vi_vs_rl/uniform/uniform_performance.csv", mode='a', header=False, index=False)


if __name__ == "__main__":
    model = "ACER"
    nsteps = 300000
    for Tt in [1000, 750, 500, 400, 300, 200, 100, 50, 1]:
        kwargs = {
            'sample_cost': 1,
            'movement_cost': Tt,
            'N': 1000
        }

        runner = UniformRunner(model, nsteps, kwargs)
        runner.train()
        runner.save()
    # nsteps *= 2

