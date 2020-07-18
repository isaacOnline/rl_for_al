import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import truncnorm, uniform

from agents import RechargingAgent
from base.scorer import RechargingScorer
from experiments.vi_vs_rl.model_runner import ModelRunner


class RechargingRunner(ModelRunner):
    def __init__(self, model_name, nsteps, recalculate_vi=False, env_params=None):
        assert np.isclose(0, (env_params['battery_capacity'] / env_params['gamma']) % 1)

        # make sure (movement cost x delta) is divisible by gamma, as battery level needs to be divisible by gamma
        # (since the cost of an action determines how much the battery level declines)
        assert np.isclose(0, (env_params['movement_cost'] * env_params['delta'] / env_params['gamma']) % 1)
        self.agent = RechargingAgent
        self.scorer = RechargingScorer
        self.env_name = "recharging"
        self.dist = env_params['dist']
        self.dist_name = self.dist.dist.name
        ModelRunner.__init__(self, model_name, nsteps, recalculate_vi, env_params)

    def _set_policy_path(self):
        self.policy_path = f"experiments/vi_vs_rl/recharging/vi_policies/" \
                           f"{int(self.params['movement_cost'])}_" \
                           f"{self.params['sample_cost']}_" \
                           f"{self.N}_" \
                           f"{self.params['gamma']}_" \
                           f"{self.params['battery_capacity']}_" \
                           f"{self.dist_name}.npy"
        self._vi_policy_load = np.load

    def get_rl_policy(self):
        # Number of different possible battery levels
        battery_options = int(round(self.params['battery_capacity'] / self.params['gamma']))
        policy = np.zeros((self.N+1, self.N+1, battery_options + 1), dtype = (np.float128, 2))
        for first_end in range(0, self.N + 1):
            for second_end in range(0, self.N + 1):
                for battery_level in range(0, battery_options + 1):
                    obs = np.array([first_end, second_end, battery_level])
                    # self.model.action_probability returns the stochastic policy, so we just find the most likely action
                    action = self.model.predict(obs, deterministic = True)[0]
                    policy[first_end, second_end, battery_level] = action
        return policy

    def plot(self, rl_policy, vi_policy):

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:2]
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

        ax1.set_title("Avg Cost")
        ax1.bar(['RL','VI'], [-self.rl_performance['reward'], -self.vi_performance['reward']], color = colors)
        ax1.set_ylabel("Cost")

        ax2.set_title("Avg Distance Traveled")
        ax2.bar(['RL','VI'], [self.rl_performance['dist'], self.vi_performance['dist']], color = colors)
        ax2.set_ylabel("Distance")
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.label.set_rotation(270)
        ax2.yaxis.labelpad = 10

        ax3.set_title("Avg Number of Samples")
        ax3.bar(['RL','VI'], [self.rl_performance['n_samples'], self.vi_performance['n_samples']], color = colors)
        ax3.set_ylabel("Number of Samples")

        ax4.set_title("Avg Number of Recharges")
        ax4.bar(['RL','VI'], [self.rl_performance['recharges'], self.vi_performance['recharges']], color = colors)
        ax4.set_ylabel("Number of Recharges")
        ax4.yaxis.set_label_position('right')
        ax4.yaxis.label.set_rotation(270)
        ax4.yaxis.labelpad = 10

        fig.suptitle(f'Ts: {self.params["sample_cost"]}; '
                     f'Tt: {self.params["movement_cost"]}; '
                     f'N: {self.N}; '
                     f'Battery Capacity: {self.params["battery_capacity"]}; '
                     f'Time Increment: {self.params["gamma"]}\n'
                     f'RL train time: {self.rl_train_time}, VI train time: {self.vi_train_time}')
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.8)

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
            'vi_recharges': [self.vi_performance['recharges']],
            'rl_recharges': [self.rl_performance['recharges']],
            'distribution': [self.dist_name],
            'gamma': [self.params['gamma']],
            'battery_capacity': [self.params['battery_capacity']]
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
    model = "PPO2"
    nsteps = 1000
    N = 25
    delta = 1/N
    gamma = 5
    sample_cost = 1
    battery_cap = 2500
    dist = get_truncnorm()
    for Tt in [750]: #,50, 100, 500, 250, 100, 50, 1000]:

        kwargs = {
            'sample_cost': sample_cost,
            'movement_cost': Tt,
            'delta': delta,
            'battery_capacity': battery_cap,
            'gamma': gamma,
            'dist': dist
        }

        runner = RechargingRunner(model, nsteps, recalculate_vi=False, env_params=kwargs)
        runner.load("rl-baselines-zoo-master/logs/ppo2/recharging-v0_9/best_model.zip")
        #runner.train(use_callback=True)
        runner.save()