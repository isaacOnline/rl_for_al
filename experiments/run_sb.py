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
NSTEPS = 50000
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


def get_vi_policy(type, params):
    try:
        policy = np.genfromtxt(f"results/value_iterator/{params['movement_cost']}_{params['N']}_{type}.csv")
    except:
        agnt = UniformAgent(sample_cost=params['sample_cost'],
                            movement_cost=params['movement_cost'],
                            N=params['N'])
        agnt.save()
        policy = agnt.policy
    return policy


def get_rl_policy(env, model, N):
    policy = []
    for i in range(0, N + 1):
        obs = np.array(i)
        action = np.argmax(model.action_probability(obs))
        mvmt, _ = env.get_movement(obs, N, action)
        policy.append([obs, mvmt])
    policy = np.array(policy)
    return policy


def plot_policy_vs_optimal(rl_policy, vi_policy, params, dif):
    plt.clf()

    plt.plot(rl_policy[:, 0], rl_policy[:, 1], c='tab:orange', label=f"{MODEL} Learner")
    plt.plot(rl_policy[:,0], vi_policy, c='B', label="Value Iteration")

    plt.title(f"Train Time: {dif}, tt/ts: {params['movement_cost']}/{params['sample_cost']}, N: {params['N']}")
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
    return i


def save_scores(rl_policy, vi_policy, run_time, id):
    rl_fout = rl_policy[:,1]
    env = gym.make('change_point:uniform-v0', **kwargs)
    print("{}:".format(MODEL))
    rl_reward, rl_ns, rl_dist = scorer().score(rl_fout,env)

    env = gym.make('change_point:uniform-v0', **kwargs)
    print("\nValue Iteration:")
    vi_reward, vi_ns, vi_dist = scorer().score(vi_policy, env)

    line = pd.DataFrame({
        "id": i,
        "time_steps":[NSTEPS],
        "train_time":[run_time],
        "model":[MODEL],
        'Ts': kwargs['sample_cost'],
        'Tt': kwargs['movement_cost'],
        'vi_reward': [vi_reward],
        'rl_reward': [rl_reward],
        'vi_ns': [vi_ns],
        'rl_ns': [rl_ns],
        'vi_dist': [vi_dist],
        'rl_dist': [rl_dist]
    })
    line.to_csv(f"results/{MODEL}/varying_tt.csv",  mode='a', header=False, index=False)


while True:
    for Tt in [1000,100,1,50,200,300,400,500,700]:
        kwargs = {
            'sample_cost': 1,
            'movement_cost': Tt,
            'N': 300
        }

        env = gym.make('change_point:uniform-v0', **kwargs)


        model = models[MODEL]('MlpPolicy', env, verbose=1, gamma = 1)

        start_time = datetime.now()
        model = model.learn(total_timesteps=NSTEPS)
        end_time = datetime.now()
        dif = end_time-start_time
        model.save(f"other/{MODEL}_Model")

        rl_policy = get_rl_policy(env, model, kwargs['N'])
        vi_policy = get_vi_policy("uniform",kwargs)
        i = plot_policy_vs_optimal(rl_policy, vi_policy, kwargs, dif)

        save_scores(rl_policy, vi_policy, dif, i)

        env.close()
    NSTEPS *= 2
