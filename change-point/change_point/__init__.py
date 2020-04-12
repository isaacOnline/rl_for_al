from gym.envs.registration import register

register(
    id='uniform-v0',
    entry_point='change_point.envs:UniformCP',
)

register(
    id='normal-v0',
    entry_point='change_point.envs:TruncNormCP',
)