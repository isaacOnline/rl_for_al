from agents.isaac_vi_agent import UniformIterator


def test_probability_calculation():
    agnt = UniformIterator(1000)
    assert agnt._calculate_prob(0.5, 1, 0.5) == 0.5
    assert agnt._calculate_prob(0.4, 1, 0.6) == 0.4
    assert agnt._calculate_prob(0.25, 0.5, 0.5) == 0.5
    assert agnt._calculate_prob(0.4, 0.37, 0.2) == 0
    assert agnt._calculate_prob(0, 1, 0) == 0
    assert agnt._calculate_prob(1, 1, 1) == 1


