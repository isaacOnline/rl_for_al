import numpy as np
import matplotlib.pyplot as plt
import gym
from other.Scorer import scorer


class NonUniformAgent(object):
    def __init__(self, Ts, Tt, N=1000, stop_error=1,prize=10000):
        self.Ts = Ts
        self.Tt = Tt
        self.N = N
        self.stop_error = stop_error
        self.calc_policy(Ts, Tt, np.repeat(1/(N), N), N, stop_error)
        kwargs = {
            'Ts':self.Ts,
            'Tt':self.Tt,
            'N':self.N,
            'stop_error':self.stop_error,
            'prize':prize
        }
        self.env = gym.make("change_point:normal-v0", **kwargs)

    def calc_policy(self, Ts, Tt, p, N=10, stop_error=1):
        N = int(N)
        stop_error = stop_error / N
        Delta = 1 / N
        # states are tuples (x,xh) where x is the current location and xh is
        # the opposite end of the hypothesis space
        f = np.zeros((N + 1, N + 1))
        val = np.zeros((N + 1, N + 1))

        # loop over possible lengths of hypothesis space
        # dd = |x - xh|
        for dd in range(N + 1):
            print(dd / N)
            # loop over possible current locations
            for xx in range(N):
                # calculate possible values of xh
                xhList = []
                if xx - dd >= 0:
                    xhList.append(xx - dd)
                if xx + dd <= N:
                    xhList.append(xx + dd)

                # now compute value of each state
                if dd / N <= stop_error:
                    for xh in xhList:
                        val[xx, xh] = 0
                else:
                    for xh in xhList:
                        # look at all possible actions from this state
                        # actions are distances to travel
                        minVal = np.inf
                        bestAction = 0
                        for aa in range(1, dd):
                            # Pst is the probability theta lies between xx and xx +/- aa
                            if xx <= xh:
                                # move forward
                                direction = 1
                                Pst = np.sum(p[xx:xx + aa]) / np.sum(p[xx:xh])
                            else:
                                # move backward
                                direction = -1
                                Pst = np.sum(p[xx - aa:xx]) / np.sum(p[xh:xx])
                                # Pstc is the complement (theta between xx +/- aa and xh)
                            Pstc = 1 - Pst
                            tempVal = Ts + Tt * aa / N + Pst * val[xx + direction * aa, xx] + Pstc * val[
                                xx + direction * aa, xh]
                            if tempVal < minVal:
                                bestAction = (xx + direction * aa) / N
                                minVal = tempVal

                            f[xx, xh] = bestAction
                            val[xx, xh] = minVal

        self.fout = [fs / N for fs in f]

        self.f = f
        self.val = val

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
    stop_error = 1
    Ts = 1
    Tt = 10
    N = 10
    reward = 0
    kwargs = {
        'Ts': Ts,
        'Tt': Tt,
        'N': N,
        'stop_error': stop_error,
        'prize': reward
    }
    agnt = NonUniformAgent(Ts=Ts, Tt=Tt, N=N, stop_error=stop_error,prize=reward)
    scorer().score(agnt.fout, gym.make("change_point:uniform-v0", **kwargs))
    agnt.plot_policy()
