import gym
from gym_catcher.envs.catcher_env import CatcherEnv

env = CatcherEnv()

for i_episode in range(1):
    observation = env.reset()
    total_reward = 0
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        print(observation)
    print("Episode finished with total reward {}".format(total_reward))

env.close()