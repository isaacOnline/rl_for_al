import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from agents import QLearner, UniformAgent

sample_cost = 1
movement_cost = 0.1
N = 1000


def fit_agent():
    kwargs = {
        "sample_cost": sample_cost,
        "movement_cost": movement_cost,
        "N": N
    }
    env = gym.make("change_point:uniform-v0", **kwargs)
    agent = QLearner(discount=0.9)
    state_action_values, rewards, counts = agent.fit(env, steps=100000000)
    policy = np.array([np.arange(N + 1), agent.get_policy()]).transpose()
    sp = "results/q_learning_policy.csv"
    agent.plot_policy()
    np.savetxt(sp, policy, delimiter=",")
    np.savetxt("experiments/plot_policies/state_action_values.csv", state_action_values, delimiter=",")
    np.savetxt("experiments/plot_policies/rewards.csv", rewards, delimiter=",")
    np.savetxt("experiments/plot_policies/counts.csv", counts, delimiter=",")


def plot_q_policy():
    plt.clf()
    policy = pd.read_csv("results/q_learning_policy.csv", names=['S', 'fout'])
    plt.plot(policy.S, policy.fout)
    plt.title(f"tt/ts: {movement_cost}/{sample_cost}, N: {N}, Stop Error: {1}")
    plt.xlabel("Size of Hypothesis Space")
    plt.xlim([0, N])
    plt.ylabel("Movement into Hypothesis Space")
    plt.ylim([0, N])
    plt.savefig("visualizations/q_learning_policy.png")


def plot_q_and_optimal_policy():
    agnt = UniformAgent(Ts=sample_cost, Tt=movement_cost, N=N, stop_error=stop_error)
    agnt.fit()
    plt.clf()
    q_policy = pd.read_csv("results/q_learning_policy.csv", names=['S', 'fout'])
    plt.plot(q_policy.S, q_policy.fout, c= 'tab:orange', label="Q-Learner")
    plt.plot(agnt.S * agnt.N, agnt.fout, c='B', label="Value Iteration")
    plt.title(f"tt/ts: {movement_cost}/{sample_cost}, N: {N}, Stop Error: {1}")
    plt.xlabel("Size of Hypothesis Space")
    plt.xlim([0, N])
    plt.ylabel("Movement into Hypothesis Space")
    plt.ylim([0, N])
    plt.legend()
    plt.savefig("visualizations/q_learning_vs_optimal.png")

if __name__ == "__main__":
    fit_agent()
    plot_q_policy()