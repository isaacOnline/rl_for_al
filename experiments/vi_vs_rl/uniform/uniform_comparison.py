import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from agents import UniformAgent
from base.scorer import UniformScorer
from experiments.vi_vs_rl.model_runner import ModelRunner


class UniformRunner(ModelRunner):
    def __init__(self, model_name, nsteps, env_params):
        self.env_name = "uniform"
        self.dist_name = "uniform"
        ModelRunner.__init__(self, model_name, nsteps, env_params)

    def get_vi_policy(self, recalculate):
        if recalculate:
            policy = self._calc_vi()
        else:
            try:
                policy = np.genfromtxt(
                    f"experiments/vi_vs_rl/uniform/vi_policies/{self.params['movement_cost']}_"
                    f"{self.params['N']}_"
                    f"{self.env_name}.csv"
                )
                self.vi_train_time = "Not Calculated"
            except:
                policy = self._calc_vi()
        return policy

    def _calc_vi(self):
        agnt = UniformAgent(sample_cost=self.params['sample_cost'],
                            movement_cost=self.params['movement_cost'],
                            N=self.params['N'])
        self.vi_train_time = agnt.calculate_policy()
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

    def plot(self, rl_policy, vi_policy):
        plt.clf()

        plt.plot(rl_policy[:, 0], rl_policy[:, 1], c='tab:orange', label=f"{self.model_name} Learner")
        plt.plot(rl_policy[:, 0], vi_policy, c='B', label="Value Iteration")

        plt.title(
            f"ACER Train Time: {self.rl_train_time}, VI Train Time: {self.vi_train_time}\n"
            f"tt/ts: {self.params['movement_cost']}/{self.params['sample_cost']}, "
            f"N: {self.params['N']}")
        plt.xlabel("Size of Hypothesis Space")
        plt.ylabel("Movement on Line")
        plt.legend()

        plt.ylim((0, self.params['N']))
        plt.xlim((0, self.params['N']))

        plt.savefig(self.img_path)

    def save_performance(self, rl_policy, vi_policy):
        rl_fout = rl_policy[:, 1]
        print("{}:".format(self.model_name))
        self.rl_reward, rl_ns, rl_distance = UniformScorer().score(rl_fout, self.env)

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
            'N': self.params['N']
        })
        line.to_csv(f"experiments/vi_vs_rl/uniform/uniform_performance.csv", mode='a', header=False, index=False)


if __name__ == "__main__":
    model = "ACER"
    nsteps = 50000
    N = 1000
    todo = np.array([1000,  750,  500,  400,  300,  200,  100,   50,    1])
    while len(todo) > 0:
        for Tt in todo:
            kwargs = {
                'sample_cost': 1,
                'movement_cost': Tt,
                'N': N
            }

            runner = UniformRunner(model, nsteps, kwargs)
            runner.train()
            runner.save()
            # if score is close enough, end
            if runner.rl_reward > 1.05 * runner.vi_reward:
                todo = todo[todo != Tt]
        nsteps *= 1.25
