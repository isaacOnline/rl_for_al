import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tqdm

class QLearner:
    """
    QLearning reinforcement learning agent.
    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      discount - (float) The discount factor. Controls the perceived value of
        future reward relative to short-term reward.
      adaptive - (bool) Whether to use an adaptive policy for setting
        values of epsilon during training
    """

    def __init__(self, epsilon=0.2, discount=0.95, adaptive=False):
        self.epsilon = epsilon
        self.discount = discount
        self.adaptive = adaptive

    def random_argmax(self, results_array):
        """
        Finds the maximum of an array, but breaks any ties via random.choice

        Args:
            results_array (np.array): (N,) numpy array,

        Returns:
            random_max: An index referring to the maximum value in results_array
        """
        max_index = np.argmax(results_array)
        maximum = results_array[max_index]
        are_max = results_array == maximum
        max_indeces = np.nonzero(are_max)[0]
        random_max = np.random.choice(max_indeces)
        return random_max

    def fit(self, env, steps=1000):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length 100.
            Let s = np.floor(steps / 100), then rewards[0] should contain the
            average reward over the first s steps, rewards[1] should contain
            the average reward over the next s steps, etc.
        """
        self.env = env
        num_actions = env.action_space.n
        num_states = env.observation_space.n

        # Why does optimal agent not work for change points > 800? create rewards array
        steps_per_bin = int(np.floor(steps/100))
        total_rewards = np.full((100,), 0.0)
        box_count = np.full((100,), 0)

        state_action_values = np.full((num_states, num_actions), 0.0)
        counts = np.full((num_states, num_actions), 0.0)

        # Reset environment
        state = env.reset()
        done = False

        print(f'{datetime.now().strftime("%l:%M %p")}: Run starting')

        for i in tqdm.tqdm(range(steps)):
            # Either explore or find out which action is best to take from this state
            i_should_explore = np.random.binomial(1,self._get_epsilon(progress = i/steps))
            if i_should_explore:
                action = env.action_space.sample()
            else:
                num_possible_actions = int(env.action_space.n)
                valid_state_action_values = state_action_values[:,:num_possible_actions]
                action = self.random_argmax(valid_state_action_values[state, :])

            new_state, reward, done, info = env.step(action)
            # Count number of actions taken at each state
            counts[state,action] = counts[state,action] + 1

            # Find alpha
            alpha = 1/counts[state,action]

            # Take out previous Q(S,A)
            prevQ = state_action_values[state,action]

            # Find Q value if we take the best next action
            optimal_followup = np.max(state_action_values[new_state, :])

            # Find new Q(S,A)
            newQ = prevQ + alpha * (reward + self.discount * optimal_followup - prevQ)

            # Place new Q(S,A) in array
            state_action_values[state,action] = newQ

            # Count up the total number of rewards, binning by each avg_num_steps
            which_box = int(np.floor(i/steps_per_bin))
            total_rewards[which_box] = total_rewards[which_box] + reward
            box_count[which_box] = box_count[which_box] + 1

            # reset if done, otherwise set S to next in sequence
            if done:
                state = env.reset()
            else:
                state = new_state

        rewards = total_rewards/box_count

        self.state_action_values = state_action_values
        return state_action_values, rewards, counts

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the QLearning algorithm and the state action values. Predictions are
        run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        states = []
        actions = []
        rewards = []

        # Reset environment
        observation = env.reset()

        done = False
        while not done:
            # decide which action to take
            action = self.random_argmax(state_action_values[observation, :])

            # take action
            observation, reward, done, info = env.step(action)
            states.append(observation)

            # update lists
            actions.append(action)
            rewards.append(reward)

        states, actions, rewards = np.array(states), np.array(actions), np.array(rewards)

        return states, actions, rewards

    def _get_epsilon(self, progress):
        """
        Retrieves the current value of epsilon. Should be called by the fit
        function during each step.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        """
        An adaptive policy for epsilon-greedy reinforcement learning. Returns
        the current epsilon value given the learner's progress. This allows for
        the amount of exploratory vs exploitatory behavior to change over time.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return (1 - progress) * self.epsilon

    def get_policy(self):
        return np.argmax(self.state_action_values, axis = 1)

    def plot_policy(self):
        N = len(self.get_policy())
        plt.plot(np.arange(N), self.get_policy())
        plt.title(f"tt/ts: {self.env.Tt}/{self.env.Ts}, N: {N - 1}, Stop Error: {1}")
        plt.xlabel("Size of Hypothesis Space")
        plt.xlim([0, N-1])
        plt.ylabel("Movement into Hypothesis Space")
        plt.ylim([0, N-1])
        plt.show()

