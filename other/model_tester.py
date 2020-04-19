import os
from datetime import datetime

import gym

from abc import ABC, abstractmethod
from stable_baselines import PPO1, ACER, ACKTR, GAIL, DQN, TRPO, A2C, HER

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

class ModelRunner(ABC):
    def __init__(self, model_name, nsteps, env_params):
        self.model_name = model_name
        self.params = env_params
        self.nsteps = nsteps
        self.get_id()
        self.start_logging()

        self.env = gym.make( f"change_point:{self.env_name}-v0", **env_params)
        self.model = models[self.model_name]('MlpPolicy', self.env, verbose=1, gamma = 1, tensorboard_log=self.log_path)

    def get_id(self):
        self.id = 0
        self.img_path = f"visualizations/{self.model_name}/{self.model_name}_vs_optimal_{self.id}.png"
        while os.path.exists(self.img_path):
            self.id+=1
            self.img_path = f"visualizations/{self.model_name}/{self.model_name}_vs_optimal_{self.id}.png"

    def start_logging(self):
        self.log_path = f"logging/{self.model_name}_{self.env_name}/"
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

    def train(self):
        self.start_time = datetime.now()
        self.model = self.model.learn(total_timesteps=self.nsteps, tb_log_name=str(self.id))
        self.end_time = datetime.now()

    def save(self):
        self.model.save(f"other/{self.model_name}_Model")
        rl_policy = self.get_rl_policy()
        vi_policy = self.get_vi_policy()
        time_elapsed = self.end_time-self.start_time
        self.plot(rl_policy, vi_policy, time_elapsed)
        self.save_performance(rl_policy, vi_policy, time_elapsed)


    @abstractmethod
    def get_vi_policy(self):
        pass

    @abstractmethod
    def get_rl_policy(self):
        pass

    @abstractmethod
    def plot(self, rl_policy, vi_policy, dif):
        pass

    @abstractmethod
    def save_performance(self, rl_policy, vi_policy, dif):
        pass
