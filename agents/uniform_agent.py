import numpy as np
import matplotlib.pyplot as plt
import gym
from other.Scorer import scorer


class UniformAgent(object):
    def __init__(self, Ts, Tt, N=1000, stop_error=1,prize=10000):
        self.Ts = Ts
        self.Tt = Tt
        self.N = N
        self.stop_error = stop_error
        self.calc_policy(Ts, Tt, N, stop_error)
        kwargs = {
            'Ts':self.Ts,
            'Tt':self.Tt,
            'N':self.N,
            'stop_error':self.stop_error,
            'prize':prize
        }
        self.env = gym.make("change_point:uniform-v0", **kwargs)

    def calc_policy(self, Ts, Tt, N=1000, stop_error=1):
        stop_error = stop_error / N
        N = int(N)
        Delta = 1/N

        # initialize value, number of samples, and distance for each state
        # states are possible lengths of hypothesis space, so there are N total

        S = np.linspace(0,1,N+1)
        f = [0]*(N+1)      # policy stored as array of where to go from each state
        val = [0]*(N+1)    # value of state - total cost to termination
        ns = [0]*(N+1)    # number of samples to termination from this state
        dist = [0]*(N+1)   # distance to termination from this state

        # terminal states are those smaller than stopErr
        T = np.where(S < stop_error)[0]

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

        #policy = fout / S
        #return fout, val, dist, ns, policy
        self.S = S
        self.fout = f
        self.val = val
        self.ns = ns

    def plot_policy(self):
        plt.clf()
        plt.plot(self.S * self.N, self.fout)
        plt.title(f"tt/ts: {self.Tt}/{self.Ts}, N: {self.N}, Stop Error: {self.stop_error}")
        plt.xlabel("Size of Hypothesis Space")
        plt.xlim([0,self.N])
        plt.ylabel("Movement into Hypothesis Space")
        plt.ylim([0,self.N])
        plt.savefig("visualizations/ideal_policy.png")



if __name__ == "__main__":
    Ts = 1
    Tt = 10
    N = 1000
    kwargs = {
        'Ts': Ts,
        'Tt': Tt,
        'N': N
    }
    agnt = UniformAgent(Ts=Ts, Tt=Tt, N=N)
    scorer().score(agnt.fout, gym.make("change_point:uniform-v0", **kwargs))
    agnt.plot_policy()
