from gym.envs.registration import register

register(
    id='shooter-v0',
    entry_point='gym_shooter.envs:ShooterEnv',
)