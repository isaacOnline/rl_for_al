import numpy as np


def calcPolicy(Ts, Tt, length=1, N=1e4, stopErr=1e-3):
    N = int(N)
    Delta = length / N
    # initialize value, number of samples, and distance for each state
    # states are possible lengths of hypothesis space, so there are N total
    # Off by one here?
    S = np.linspace(0, length, N + 1)
    f = [0] * (N + 1)  # policy stored as array of where to go from each state
    val = [0] * (N + 1)  # value of state - total cost to termination
    ns = [0] * (N + 1)  # number of samples to termination from this state
    dist = [0] * (N + 1)  # distance to termination from this state

    # terminal states are those smaller than stopErr
    T = np.where(S < stopErr)[0]

    # loop over non-terminal states
    for ss in range(T[-1] + 1, N + 1):
        minVal = np.inf
        # loop over all states we can go to from state ss
        for tt in range(1, ss):
            Pst = tt / ss  # probability of transition to state tt
            Pstc = (ss - tt) / ss  # probability of transition to state ss - tt
            tempVal = Ts * (1 + Pst * ns[tt] + Pstc * ns[ss - tt]) + Tt * (
                        tt * Delta + Pst * dist[tt] + Pstc * dist[ss - tt])
            if tempVal < minVal:
                minVal = tempVal
                f[ss] = tt

        # update value, ns, and dist for this state
        Psf = f[ss] / ss
        Psfc = (ss - f[ss]) / ss
        ns[ss] = 1 + Psf * ns[f[ss]] + Psfc * ns[ss - f[ss]]
        dist[ss] = f[ss] * Delta + Psf * dist[f[ss]] + Psfc * dist[ss - f[ss]]
        val[ss] = Ts * ns[ss] + Tt * dist[ss]

    fout = [fs / N for fs in f]
    # policy = fout / S
    # return fout, val, dist, ns, policy
    return fout, val, dist, ns


