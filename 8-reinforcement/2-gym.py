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
    return tuple((x / np.array([0.25, 0.25, 0.01, 0.1])).astype(int))

def create_bins(i, num):
    return np.arange(num + 1) * (i[1] - i[0]) / num + i[0]

print("Sample bins for interval (-5, 5) with 10 bins\n", create_bins((-5, 5), 10))

ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
nbins = [20,20,10,10] # number of bins for each parameter
bins = [create_bins(ints[i],nbins[i]) for i in range(4)]

def discretize_bins(x):
    return tuple(np.digitize(x[i], bins[i]) for i in range(4))


obs, info = env.reset()

done = False
while not done:
    #env.render()
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    #print(discretize_bins(obs))
    print(discretize_bins(obs))
env.close()

env = gym.make("CartPole-v1", render_mode="human")

Q = {}
actions = (0, 1)

def qvalues(state):
    return [Q.get((state, a), 0) for a in actions]

print("--------> Before setting some hyperparameters")

# Set some hyperparameters
alpha = 0.3
gamma = 0.9
epsilon = 0.90

# Collect all cumulative rewards at each simulation at `rewards` vector for futher plotting
def probs(v, eps=1e-4):
    v = v-v.min() + eps
    v = v/v.sum()
    return v

Qmax = 0
cum_rewards = []
rewards = []
for epoch in range(100000):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs = reset_out[0]
    else:
        obs = reset_out

    done = False
    cum_reward=0
    # == do the simulation ==
    while not done:
        s = discretize_bins(obs)
        if random.random() < epsilon:
            # exploitation - chose the action according to Q-Table probabilities
            v = probs(np.array(qvalues(s)))
            a = random.choices(actions,weights=v)[0]
        else:
            # exploration - randomly chose the action
            a = np.random.randint(env.action_space.n)

        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated

        cum_reward+=rew
        ns = discretize_bins(obs)

        Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
    cum_rewards.append(cum_reward)
    rewards.append(cum_reward)
    # == Periodically print results and calculate average reward ==
    if epoch%5000==0:
        print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
        if np.average(cum_rewards) > Qmax:
            Qmax = np.average(cum_rewards)
            Qbest = Q
        cum_rewards=[]