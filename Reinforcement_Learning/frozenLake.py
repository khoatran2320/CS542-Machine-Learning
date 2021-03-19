import gym
import numpy as np
# env = gym.make('FrozenLake-v0')
env = gym.make('FrozenLake-v0').unwrapped
env.reset()
# for _ in range(10):
#     env.render()
#     observation, reward, done, info  = env.step(env.action_space.sample()) # take a random action
#     print(reward, done)
# env.close()

def q_learning(env, alpha=0.5,gamma=.95, epsilon=0.1,num_episodes=500, init_method = "Z", test = False):
    np.random.seed(42)
    if init_method == "Z":
        Q = np.zeros((env.nS, env.nA))
    elif init_method == "R":
        Q = np.random.rand(env.nS, env.nA)

    r = 0
    for i in range(num_episodes):
        done = False
        currState = env.reset()

        #decrease exploration percentage
        ep = epsilon
        if ep < .01:
            ep = .01
        else:
            ep = ep - .0001
        while not done:
            if not test:
                #epsilon greedy
                exploreProb = np.random.uniform(0,1)
                if exploreProb < ep:
                    currAction = env.action_space.sample()
                else:
                    currAction = np.argmax(Q[currState])
            else:
                #argmax
                currAction = np.argmax(Q[currState])

            #take step    
            newState, reward, done, info= env.step(currAction)
            max_Q_NS_NA = np.max(Q[newState])

            newQ = (Q[currState][currAction] * (1 - alpha)) + (alpha * ((reward + (gamma * max_Q_NS_NA))))
            Q[currState][currAction] = newQ
            currState = newState

            r = reward + r
            # print(currState, currAction, Q[currState][currAction], newQ)
        if i == 5000 or i == 10000 or i == 15000 or i == 20000 or i == 25000 or i == 30000 or i == 35000 or i == 40000 or i == 45000 or i == 49999: 
            print(r/5000)
            r = 0
    return Q

print(q_learning(env, .5, .95, .1, 50000, "Z", False))
env.close()