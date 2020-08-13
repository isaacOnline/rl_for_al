from abc import ABC, abstractmethod
import gym
import numpy as np

class MctsAgent(ABC):
    """
    Just a base class to be built out as I build agents for other environments
    """
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.N = int(round(1/self.params['delta']))
        try:
            self.epsilon = params['epsilon']
        except KeyError:
            self.epsilon = params['delta']

    @abstractmethod
    def run(self):
        pass


class NodeNotVisitedError(Exception):
    pass


class ActionNode(object):
    def __init__(self, action, parent):
        self.state = parent.state
        self.delta = parent.delta
        self.action = action
        self.total_reward = 0
        self.n = 0
        self.children = [StateNode(action, self),
                         StateNode(self.state - action, self)]
        self.parent = parent
        self.index = np.Inf

    def ev(self, parent_n):
        """
        Calculate the state action value, using the basic UCT formulation from Kocsis and Szepesvari.

        :param parent_n: The number of times the parent has been visited, used in calculating the exploration term
        :return:
        """
        if self.n == 0:
            raise NodeNotVisitedError
        c = np.sqrt(2) # Todo: Choose the actual value here
        # The value I've chosen is the recommended value from page 100 of RL & OC - Bertsekas ASSUMING THAT THE REWARD
        # HAS BEEN NORMALIZED. So either the reward needs to be normalized, or c needs to change.
        return self.total_reward / self.n + c * np.sqrt(np.log(parent_n)/ self.n)

    def get_state(self, cp):
        if self.action > cp:
            return self.children[0]
        else:
            return self.children[1]

    def update(self, value, movement_cost, sample_cost):
        value -= sample_cost + self.action * movement_cost
        self.total_reward += value
        self.n += 1
        return value



class StateNode(object):
    def __init__(self, state, parent = None, delta = None):
        self.state = state
        if parent:
            self.delta = parent.delta
        elif delta:
            self.delta = delta
        else:
            raise ValueError("If a parent node is not specified then a delta value must be given")
        self.total_reward = 0
        self.n = 0
        self.index = np.Inf
        self.children = []
        self.parent = parent

        # Randomize order so that we can just take off the first one when expanding
        N = int(round(state/self.delta))
        self.unexpanded_children = np.random.choice(np.linspace(self.delta, state - self.delta, N - 1), N - 1, replace= False)

    def get_action(self):
        best_index = -np.Inf
        best_action_node = None
        for action_node in self.children:
            ev = action_node.ev(self.n)
            if ev > best_index:
                best_index = ev
                best_action_node = action_node
        return best_action_node

    def expand(self):
        assert len(self.unexpanded_children) > 0, "Cannot expand full tree"
        new_action = self.unexpanded_children[0]
        self.unexpanded_children = self.unexpanded_children[1:]
        new_child = ActionNode(new_action, self)
        self.children += [new_child]
        return new_child

    def is_fully_expanded(self):
        num_possible_actions = self.state/self.delta - 1
        return np.isclose(num_possible_actions, len(self.children))

    def is_terminal(self, epsilon):
        return np.isclose(self.state, epsilon)

    def update(self, value, movement_cost, sample_cost):
        self.total_reward += value
        self.n += 1
        return value

    def terminal_cost(self, dist_to_cp, epsilon, movement_cost, sample_cost):
        # This should be the expected cost to finish from this state using the base policy
        # TODO: Make sure the base policy is what it should be. Right now just performs bisection.

        assert 0 <= dist_to_cp and dist_to_cp <= self.state, "Given change point not accessible"
        h_space_len = self.state
        cost = 0
        while h_space_len > epsilon and not np.isclose(h_space_len, epsilon):
            action = round((h_space_len / 2) / self.delta) * self.delta
            cost -= action * movement_cost + sample_cost
            if dist_to_cp < action:
                h_space_len = action
                dist_to_cp = action - dist_to_cp
            else:
                h_space_len = h_space_len - action
                dist_to_cp = dist_to_cp - action

        return cost


class UniformMctsAgent(MctsAgent):
    # TODO: Switch this to being more universal/less environment specific
    def run(self):
        # TODO: Switch to where it can be run starting at any state, so that it can be used on-line
        self.root = StateNode(1, delta = self.params['delta'])
        for i in range(10000):
            dist_to_cp = self.env.dist.rvs(1)[0]
            tree, dist_to_cp = self._select_node(self.root, dist_to_cp)
            if tree.is_terminal(self.epsilon):
                value = 0
            else:
                new_action = tree.expand()
                tree = new_action.get_state(dist_to_cp)
                dist_to_cp = abs(new_action.action - dist_to_cp)
                value = tree.terminal_cost(dist_to_cp,
                                                 self.epsilon,
                                                 self.params['movement_cost'],
                                                 self.params['sample_cost'])
            self._backpropogate(tree, value)

    def _select_node(self, tree: StateNode, cp):
        while tree.is_fully_expanded() and not tree.is_terminal(epsilon=self.epsilon):
            action = tree.get_action()
            tree = action.get_state(cp)
            cp = abs(action.action - cp)
        return tree, cp

    def _backpropogate(self, tree, value):
        while tree is not None:
            value = tree.update(value,
                                self.params['movement_cost'],
                                self.params['sample_cost'])
            tree = tree.parent


if __name__ == "__main__":
    # TODO: Build a way for this agent to run on the uniform gym environment
    # TODO: Make sure that the tree is saved as you progress through the environment when doing ^
    parameters = {
        "movement_cost": 1,
        "sample_cost": 1,
        "delta": 0.1
    }
    env = gym.make("change_point:uniform-v0",**parameters)
    agent = UniformMctsAgent(env, parameters)
    agent.run()