import sys
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random

# Initialize
env = gym.make('CartPole-v1', render_mode='human')
print(env.action_space)
print(env.observation_space)
print(env.action_space.sample())

# env.reset()

# for i in range(100):
#     env.render()
#     env.step(env.action_space.sample())
# env.close()

# env.reset()

# done = False
# while not done:
#     env.render()
#     obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
#     done = terminated or truncated
#     print(f"{obs} -> {rew}")
# env.close()

# get min and max value of those numbers
print(env.observation_space.low)
print(env.observation_space.high)

# function that will take the observation from our model and produce a tuple of 4 integer values
def discretize(x):
    return tuple((x / np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))

def create_bins(i, num):
    return np.arange(num + 1) * (i[1] - i[0]) / num + i[0]

print("Sample bins for interval (-5, 5) with 10 bins\n", create_bins((-5, 5), 10))

ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
nbins = [20,20,10,10] # number of bins for each parameter
bins = [create_bins(ints[i],nbins[i]) for i in range(4)]

def discretize_bins(x):
    return tuple(np.digitize(x[i], bins[i]) for i in range(4))


env.reset()

done = False
while not done:
    #env.render()
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    #print(discretize_bins(obs))
    print(discretize(obs))
env.close()