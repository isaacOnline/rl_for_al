import os
from datetime import datetime

import gym

from abc import ABC, abstractmethod
from stable_baselines import PPO1, ACER, ACKTR, GAIL, DQN, TRPO, A2C, HER
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

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
    def __init__(self, model_name, nsteps, recalculate_vi=False, env_params=None):
        self.model_name = model_name
        self.params = env_params
        self.nsteps = nsteps
        self.get_id()
        self.start_logging()

        self.env = gym.make( f"change_point:{self.env_name}-v0", **env_params)
        self.model = models[self.model_name]('MlpPolicy', self.env, verbose=1, gamma = 1, tensorboard_log=self.log_path)
        self.score_vi(recalculate_vi)

    def get_id(self):
        self.id = 0

        self.img_path = f"experiments/vi_vs_rl/{self.env_name}/visualizations/{self.dist_name}_{self.id}.png"
        while os.path.exists(self.img_path):
            self.id+=1
            self.img_path = f"experiments/vi_vs_rl/{self.env_name}/visualizations/{self.dist_name}_{self.id}.png"
        open(self.img_path, "a").close()

    def start_logging(self):
        self.log_path = f"experiments/vi_vs_rl/{self.env_name}/logging/{self.dist_name}_{self.id}/"
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

    def train(self):
        start_time = datetime.now()

        # #callback to stop training when vi reward is reached
        # eval_env = self.env = gym.make( f"change_point:{self.env_name}-v0", **self.params)
        # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=self.vi_reward * 1.05, verbose=1)
        # eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best,
        #                              verbose=1,
        #                              n_eval_episodes = 200)

        self.model = self.model.learn(total_timesteps=self.nsteps) #,callback=eval_callback
        end_time = datetime.now()
        self.rl_train_time = end_time - start_time

    def save(self, message = None):
        rl_policy = self.get_rl_policy()
        vi_policy = self.vi_policy
        self.model.save(f"experiments/vi_vs_rl/{self.env_name}/model_objects/{self.dist_name}_{self.id}")
        if message:
            print(message)
        self.save_performance(rl_policy, vi_policy)
        self.plot(rl_policy, vi_policy)


    @abstractmethod
    def score_vi(self, recalculate):
        pass

    @abstractmethod
    def get_vi_policy(self, recalculate):
        pass

    @abstractmethod
    def get_rl_policy(self):
        pass

    @abstractmethod
    def plot(self, rl_policy, vi_policy):
        pass

    @abstractmethod
    def save_performance(self, rl_policy, vi_policy):
        pass
