from src.optimal_agent import OptimalAgent
from src.tracker import Tracker
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def changing_tt(num_samples):
    """
        How number of samples and distance travelled change with different values of Tt
    :return:
    """
    stopErr = 1e-3
    Ts = 1
    N = 1000
    for Tt in range(1, 1000):
        agent = OptimalAgent(Ts, Tt, N, stopErr)
        tracker = Tracker(agent)
        for sample in range(num_samples):
            agent.fit()
            tracker.log_uniform()
        if Tt % 100 == 99:
            print("{0}: {1} values complete, {2} remaining".format(time.strftime("%-I:%M %p"),Tt+1,1000-Tt-1))


def changing_ts(num_samples):
    """
    How number of samples and distance travelled change with different values of Tt
    :return:
    """
    stopErr = 1e-3
    Tt = 1
    N = 1000
    for Ts in range(1, 1000):
        agent = OptimalAgent(Ts, Tt, N, stopErr)
        tracker = Tracker(agent)
        for sample in range(num_samples):
            agent.fit()
            tracker.log_uniform()
        if Ts % 100 == 99:
            print("{0}: {1} values complete, {2} remaining".format(time.strftime("%-I:%M %p"),Ts+1,1000-Tt-1))


def plot_distance():
    data = pd.read_csv("results/uniform.csv")
    data['ratio'] = data.Tt / data.Ts
    consolidated = data.groupby(["Ts", "Tt"])
    consolidated = consolidated.agg(np.mean)
    consolidated = consolidated.reset_index()
    plt.scatter(x=consolidated.ratio, y = consolidated.total_dist)
    plt.xlabel("Tt/Ts")
    plt.ylabel("Total Distance Traveled")
    plt.savefig("visualizations/time_ratio_to_dist.png")



if __name__ == "__main__":
    changing_tt(1000)