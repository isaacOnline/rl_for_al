import gym
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from other.Scorer import scorer
from datetime import datetime
from agents import UniformAgent
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1, ACER, ACKTR, GAIL, DQN, TRPO, A2C, HER

MODEL = "ACER"
NSTEPS = 1000000
models = {
    "A2C": A2C,
    "ACER": ACER,
    "ACKTR": ACKTR,     # Appeared to get stuck on first training round
    "DQN": DQN,
    "GAIL": GAIL,       # Not working, would need to spend more time reading to get it working
    "HER": HER,
    "PPO1": PPO1,
    "TRPO": TRPO
}


def get_vi_policy(params):
    agnt = UniformAgent(Ts=params['Ts'],
                        Tt=params['Tt'],
                        N=params['N'],
                        stop_error=params['stop_error'],
                        prize=params['prize'])
    policy = np.array([agnt.S * agnt.N, agnt.fout])
    return policy.transpose()


def get_rl_policy(model, N):
    policy = []
    for i in range(N + 1):
        obs = np.array(i)
        action, _ = model.predict(obs, deterministic=True)
        if action == 0:
            action += 1
        elif action == N:
            action -= 1
        action = action / N * i
        action = np.ceil(action)
        policy.append([obs, action])
    policy = np.array(policy)
    return policy


def plot_policy_vs_optimal(rl_policy, vi_policy, params, dif):
    plt.clf()

    plt.plot(rl_policy[:, 0], rl_policy[:, 1], c='tab:orange', label=f"{MODEL} Learner")
    plt.plot(vi_policy[:,0], vi_policy[:,1], c='B', label="Value Iteration")

    plt.title(f"Train Time: {dif}, tt/ts: {params['Tt']}/{params['Ts']}, N: {params['N']}")
    plt.xlabel("Size of Hypothesis Space")
    plt.ylabel("Movement into Hypothesis Space")
    plt.legend()

    plt.ylim((0, params['N']))
    plt.xlim((0, params['N']))

    i = 0
    save_path = f"visualizations/{MODEL}/{MODEL}_vs_optimal_{i}.png"
    while os.path.exists(save_path):
        i+=1
        save_path = f"visualizations/{MODEL}/{MODEL}_vs_optimal_{i}.png"
    plt.savefig(save_path)


def save_scores(rl_policy, vi_policy, run_time):
    rl_fout = rl_policy[:,1]
    env = gym.make('change_point:uniform-v0', **kwargs)
    print("{}:".format(MODEL))
    rl_reward, rl_ns, rl_dist = scorer().score(rl_fout,env)

    vi_fout = vi_policy[:,1]
    env = gym.make('change_point:uniform-v0', **kwargs)
    print("\nValue Iteration:")
    vi_reward, vi_ns, vi_dist = scorer().score(vi_fout,env)

    line = pd.DataFrame({
        "time_steps":[NSTEPS],
        "train_time":[run_time],
        "model":[MODEL],
        'Ts': kwargs['Ts'],
        'Tt': kwargs['Tt'],
        'vi_reward': [vi_reward],
        'rl_reward': [rl_reward],
        'vi_ns': [vi_ns],
        'rl_ns': [rl_ns],
        'vi_dist': [vi_dist],
        'rl_dist': [rl_dist]
    })
    line.to_csv("results/varying_tt.csv",  mode='a', header=False, index=False)

while True:
    for Tt in [1000,100,1,50,200,300,400,500,700]:
        kwargs = {
            'Ts': 1,
            'Tt': Tt,
            'N': 1000,
            'stop_error': 1,
            'prize': 0
        }

        env = gym.make('change_point:uniform-v0', **kwargs)
        # Optional: PPO1 requires a vectorized environment to run
        # the env is now wrapped automatically when passing it to the constructor
        # env = DummyVecEnv([lambda: env])

        model = models[MODEL]('MlpPolicy', env, verbose=1)

        start_time = datetime.now()
        for i in range(NSTEPS//2000):
            model.learn(total_timesteps=2000)
        end_time = datetime.now()
        dif = end_time-start_time
        model.save(f"other/{MODEL}_Model")

        rl_policy = get_rl_policy(model, 1000)
        vi_policy = get_vi_policy(kwargs)
        np.savetxt(f"results/{MODEL}_policy.csv", rl_policy)
        plot_policy_vs_optimal(rl_policy, vi_policy, kwargs, dif)

        save_scores(rl_policy, vi_policy, dif)

        env.close()
    NSTEPS *= 2
