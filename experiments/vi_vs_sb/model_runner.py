import os
import shutil
from datetime import datetime
import signal
import gym
import numpy as np

from abc import ABC, abstractmethod
from stable_baselines import PPO1, PPO2, ACER, ACKTR, GAIL, DQN, TRPO, A2C, HER
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Model dictionary so they can be specified by a string
models = {
    "A2C": A2C,
    "ACER": ACER,
    "ACKTR": ACKTR,
    "DQN": DQN,
    "GAIL": GAIL,
    "HER": HER,
    "PPO1": PPO1,
    "PPO2": PPO2,
    "TRPO": TRPO
}

class ModelRunner(ABC):
    def __init__(self, model_name, nsteps, env_params=None):
        """
        Set up run by saving params, env, and model + beginning logging and reserving space for results to be
        written to

        :param model_name:
        :param nsteps:
        :param env_params:
        """
        self._add_exit_handling()

        self.model_name = model_name
        self.params = env_params
        self.N = round(1/self.params['delta'])
        self.nsteps = nsteps
        self.get_id()
        self.start_logging()

        self.env = gym.make( f"change_point:{self.env_name}-v0", **env_params)
        self.sb_model = models[self.model_name]('MlpPolicy', self.env, verbose=1, gamma = 1, tensorboard_log=self.log_path)
        self._set_policy_path()
        self.performance_path = f"experiments/vi_vs_sb/{self.env_name}/{self.env_name}_performance.csv"

    def _add_exit_handling(self):
        """
        If signal is interrupted and run does not complete, then delete the reserved image space and log

        :return:
        """
        def free_up_save_space(a, b):
            # this gets fed two inputs when called, neither of which are needed
            os.remove(self.img_path)
            shutil.rmtree(self.log_path)
        signal.signal(signal.SIGINT, free_up_save_space)

    def start_logging(self):
        """
        Create a directory where model can log to when training

        :return:
        """
        self.log_path = f"experiments/vi_vs_sb/{self.env_name}/logging/{self.dist_name}_{self.id}/"
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

    def get_id(self):
        """
        Create an id for this run of the experiment by looking at the files in the visualizations library.
        Then create an empty image based on the id, so if multiple experiments are being run at the same time
        they won't use the same id.

        :return:
        """

        self.id = 0

        self.img_path = f"experiments/vi_vs_sb/{self.env_name}/visualizations/{self.dist_name}_{self.id}.png"
        while os.path.exists(self.img_path):
            self.id+=1
            self.img_path = f"experiments/vi_vs_sb/{self.env_name}/visualizations/{self.dist_name}_{self.id}.png"
        # Reserve this image path, in case multiple runners are being run at the same time
        open(self.img_path, "a").close()

    def train_sb(self, use_callback=False):
        """
        Train a stable baselines model.

        :param use_callback: If True, this will add a callback so that the training will terminate early if
                             the training performance reaches that of the value iterator.
        :return:
        """
        start_time = datetime.now()

        if use_callback:
            eval_env = self.env

            #callback to stop training when vi reward is reached
            try:
                callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=self.vi_performance['reward'], verbose=1)
                eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best,
                                             verbose=1,
                                             n_eval_episodes = 250,
                                             eval_freq=10000)
                self.sb_model = self.sb_model.learn(total_timesteps=self.nsteps, callback=eval_callback)
            except AttributeError:
                # If value iterator hasn't been evaluated yet, just train w/o a callback
                print("Value iteration not yet tested. Training without a callback.")
                self.sb_model = self.sb_model.learn(total_timesteps=self.nsteps)
        else:
            self.sb_model = self.sb_model.learn(total_timesteps=self.nsteps)

        end_time = datetime.now()
        self.sb_train_time = end_time - start_time
        self.sb_policy = self.get_sb_policy()

    def load_sb(self, path):
        """
        Load a stable baselines agent from a saved model object instead of training

        :param path:
        :return:
        """
        self.sb_model = models[self.model_name].load(path)
        self.sb_policy = self.get_sb_policy()
        self.sb_train_time = "Not Calculated"

    def score_sb(self):
        """
        Score and save the performance of the stable baselines model

        :return:
        """
        self.sb_performance = self.scorer().score(self.sb_policy, self.env)

    def train_vi(self):
        """
        Train a value iteration model

        :return:
        """
        agnt = self.vi_model(**self.params)
        self.vi_train_time = agnt.calculate_policy()
        agnt.save()
        policy = agnt.gym_actions
        self.vi_policy = policy

    def load_vi(self):
        """
        Load a value iteration model from storage

        :return:
        """
        try:
            # Some policies are saved as csvs and some are saved as np arrays, so the loading functions are
            # specific to each problem
            policy = self._vi_policy_load(self.policy_path)
            self.vi_train_time = "Not Calculated"
            self.vi_policy = policy
        except:
            print("VI policy could not be loaded; recalculating")
            self.train_vi()

    def score_vi(self):
        """
        Evaluate the performance of the value iteration model

        :return:
        """
        self.vi_performance = self.scorer().score(self.vi_policy, self.env)

    def save(self, message = None):
        """
        Save an image comparing the policies/performance of the two models, as well as line of information
        about the run the the performance log.

        :return:
        """
        model_path = f"experiments/vi_vs_sb/{self.env_name}/model_objects/{self.dist_name}_{self.id}"
        self.sb_model.save(model_path)
        self._save_performance()
        self.plot()

    @abstractmethod
    def get_sb_policy(self):
        """
        Harvest the policy from the stable baselines agent.

        :return:
        """
        pass

    @abstractmethod
    def plot(self):
        """
        Create a plot comparing performance/policy between stable baselines and the value iterator

        :return:
        """
        pass

    @abstractmethod
    def _save_performance(self):
        """
        Save the results of the experiment to the performance log

        :return:
        """
        pass

    @abstractmethod
    def _set_policy_path(self):
        """
        Save the path where the value iteration policy will be saved to or loaded from

        :return:
        """
        pass
