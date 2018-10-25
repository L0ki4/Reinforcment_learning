from gym.envs.registration import register

register(
    id='gpn-v0',
    entry_point='gym_gpn.envs:GpnEnv',
)