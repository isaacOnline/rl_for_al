import numpy as np
import matplotlib.pyplot as plt
from src import ChangePoint


class OptimalAgent(object):
    def __init__(self, Ts, Tt, N=1e4, stopErr=1e-3,length=1):
        self.Ts = Ts
        self.Tt = Tt
        self.N = N
        self.stopErr = stopErr
        self.calc_policy(Ts, Tt, length, N, stopErr)
        self.env = ChangePoint(self.Ts, self.Tt, self.N, self.stopErr)

    def calc_policy(self, Ts, Tt, length=1, N=1e4, stopErr=1e-3):
        N = int(N)
        Delta = length/N
        # initialize value, number of samples, and distance for each state
        # states are possible lengths of hypothesis space, so there are N total

        S = np.linspace(0,length,N+1)
        f = [0]*(N+1)      # policy stored as array of where to go from each state
        val = [0]*(N+1)    # value of state - total cost to termination
        ns = [0]*(N+1)    # number of samples to termination from this state
        dist = [0]*(N+1)   # distance to termination from this state

        # terminal states are those smaller than stopErr
        T = np.where(S < stopErr)[0]

        # loop over non-terminal states
        for ss in range(T[-1]+1,N+1):
            minVal = np.inf
            # loop over all states we can go to from state ss
            for tt in range(1,ss):
                Pst = tt/ss          # probability of transition to state tt
                Pstc = (ss-tt)/ss    # probability of transition to state ss - tt
                tempVal = Ts*(1 + Pst*ns[tt] + Pstc*ns[ss-tt]) + Tt*(tt*Delta + Pst*dist[tt] + Pstc*dist[ss-tt])
                if tempVal < minVal:
                    minVal = tempVal
                    f[ss] = tt

            # update value, ns, and dist for this state
            Psf = f[ss]/ss
            Psfc = (ss - f[ss])/ss
            ns[ss] = 1 + Psf*ns[f[ss]] + Psfc*ns[ss - f[ss]]
            dist[ss] = f[ss]*Delta + Psf*dist[f[ss]] + Psfc*dist[ss - f[ss]]
            val[ss] = Ts*ns[ss] + Tt*dist[ss]

        fout = [fs/N for fs in f]
        #policy = fout / S
        #return fout, val, dist, ns, policy
        self.S = S
        self.fout = fout
        self.val = val
        self.dist = dist
        self.ns = ns

    def plot_policy(self):
        plt.plot(self.S,self.fout)
        plt.title(f"ts/tt: {self.Ts}/{self.Tt}, N: {self.N}, Stop Error: {self.stopErr}")
        plt.xlabel("Size of Hypothesis Space")
        plt.xlim([0,1])
        plt.ylabel("Movement into Hypothesis Space")
        plt.ylim([0,1])
        plt.show()
        plt.clf()

    def fit(self):
        observation, reward, done, _ = self.env.reset()
        self.total_dist = 0
        self.num_samples= 0
        self.location = 0
        self.location_hist = []
        self.min_loc = 0
        self.max_loc = 1
        self.direction = 1
        self.length_of_hypothesis_space = 1
        while not done:
           action = self.fout[int(self.length_of_hypothesis_space*self.N)]
           observation, reward, done, _ = self.env.step(int(action*self.N)) # Temporarily rescale action to get around gym's Discrete, which seems to only allow Discrete input for natural numbers
           self.location_hist.append(self.location)
           self.location += action * self.direction
           self.total_dist += action
           self.num_samples += 1
           if observation == 1:
               self.direction = 1
               self.min_loc = self.location
           else:
               self.direction = -1
               self.max_loc = self.location
           self.length_of_hypothesis_space = self.max_loc - self.min_loc
           self.length_of_hypothesis_space = np.round(self.length_of_hypothesis_space,
                                                      int(np.log10(self.N)))  # ignore rounding errors


if __name__ == "__main__":
    OptimalAgent()
