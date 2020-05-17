import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from agents import UniformAgent
from base.scorer import UniformScorer
from experiments.vi_vs_rl.model_runner import ModelRunner


class UniformRunner(ModelRunner):
    def __init__(self, model_name, nsteps, recalculate_vi=False, env_params=None):
        self.env_name = "uniform"
        self.dist_name = "uniform"
        ModelRunner.__init__(self, model_name, nsteps, recalculate_vi, env_params)

    def get_vi_policy(self, recalculate):
        if recalculate:
            policy = self._calc_vi()
        else:
            try:
                policy = np.genfromtxt(
                    f"experiments/vi_vs_rl/uniform/vi_policies/{self.params['movement_cost']}_"
                    f"{round(1/self.params['delta'])}_"
                    f"{self.env_name}.csv"
                )
                self.vi_train_time = "Not Calculated"
            except:
                policy = self._calc_vi()
        return policy

    def _calc_vi(self):
        agnt = UniformAgent(sample_cost=self.params['sample_cost'],
                            movement_cost=self.params['movement_cost'],
                            delta=self.params['delta'])
        self.vi_train_time = agnt.calculate_policy()
        agnt.save()
        policy = agnt.gym_policy
        return policy

    def get_rl_policy(self):
        policy = []
        N = round(1/(self.params['delta']))
        for i in range(N + 1):
            obs = np.array(i)
            obs.shape = (1,)
            action = np.argmax(self.model.action_probability(obs))
            policy.append(action)
        policy = np.array(policy)
        return policy

    def plot(self, rl_policy, vi_policy):
        plt.clf()
        N = round(1 / (self.params['delta']))

        plt.plot(np.linspace(0, 1, N+1), rl_policy, c='tab:orange', label=f"{self.model_name} Learner")
        plt.plot(np.linspace(0, 1, N+1), vi_policy, c='B', label="Value Iteration")

        plt.title(
            f"ACER Train Time: {self.rl_train_time}, VI Train Time: {self.vi_train_time}\n"
            f"tt/ts: {self.params['movement_cost']}/{self.params['sample_cost']}, "
            f"N: {int(1/self.params['delta'])}")
        plt.xlabel("Size of Hypothesis Space")
        plt.ylabel("Gym Policy")
        plt.legend()

        plt.xlim((0, 1))
        plt.ylim((0, N))

        plt.savefig(self.img_path)

    def score_vi(self, recalculate):
        self.vi_policy = self.get_vi_policy(recalculate)
        self.vi_reward, vi_ns, vi_distance = UniformScorer().score(self.vi_policy, self.env)

    def save_performance(self, rl_policy, vi_policy):
        print("{}:".format(self.model_name))
        self.rl_reward, rl_ns, rl_distance = UniformScorer().score(rl_policy, self.env)

        print("\nValue Iteration:")
        self.vi_reward, vi_ns, vi_distance = UniformScorer().score(vi_policy, self.env)

        line = pd.DataFrame({
            "id": self.id,
            "time_steps": [self.nsteps],
            "vi_train_time": [self.vi_train_time],
            "rl_train_time": [self.rl_train_time],
            "model": [self.model_name],
            'Ts': self.params['sample_cost'],
            'Tt': self.params['movement_cost'],
            'vi_reward': [self.vi_reward],
            'rl_reward': [self.rl_reward],
            'vi_ns': [vi_ns],
            'rl_ns': [rl_ns],
            'vi_distance': [vi_distance],
            'rl_distance': [rl_distance],
            'N': int(1/self.params['delta'])
        })
        line.to_csv(f"experiments/vi_vs_rl/uniform/uniform_performance.csv", mode='a', header=False, index=False)


if __name__ == "__main__":
    model = "ACER"
    nsteps = 2000
    N = 10
    Tts = np.array([1, 1, 1, 1, 50, 200, 300, 400, 500, 750, 1000])
    for Tt in Tts:
        kwargs = {
            'sample_cost': 1,
            'movement_cost': Tt,
            'delta': 1/N
        }

        runner = UniformRunner(model, nsteps, True, kwargs)
        runner.train()
        runner.save()
        # if score is close enough, end
