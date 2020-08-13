import numpy as np
from scipy.stats import truncnorm, uniform


def get_truncnorm():
    """
    Return a truncated normal distribution as specified in section 4 of the sps paper
    :return:
    """
    min = 0
    max = 1
    mean = 0.5
    sd = np.sqrt(0.1)
    a = (min - mean) / sd
    b = (max - mean) / sd
    dist = truncnorm(a, b, loc=mean, scale=sd)
    return dist


def get_unif():
    return uniform(0,1)