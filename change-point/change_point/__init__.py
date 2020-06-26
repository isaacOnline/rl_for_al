from gym.envs.registration import register

register(
    id='uniform-v0',
    entry_point='change_point.envs:UniformCP',
)

register(
    id='non_uniform-v0',
    entry_point='change_point.envs:NonUniformCP',
)

register(
    id='recharging-v0',
    entry_point='change_point.envs:RechargingCP',
)