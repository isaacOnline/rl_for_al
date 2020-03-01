from gym.envs.registration import register

register(
    id='uniform-v0',
    entry_point='change_point.envs:UniformPrior',
)

register(
    id='non-uniform-v0',
    entry_point='change_point.envs:NonUniformPrior',
)