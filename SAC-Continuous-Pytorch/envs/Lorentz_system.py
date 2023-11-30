import gym.spaces as spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class env:
    def __init__(self):
        self.a1 = -1.
        self.a2 = 1.
        self.a3 = 2.
        self.step_num = 0
        self.action_space = spaces.Box(low=np.array(
            [-1]), high=np.array([1]), dtype=np.float32)

    def reset(self):
        self.a1 = -1.
        self.a2 = 1.
        self.a3 = 2.
        self.step_num = 0

        return [self.a1, self.a2,self.a3]

    def step(self, action):

        h = 0.001
        sigma_ = 10
        b_ = 8/3
        r_ = 28
        self.a1 +=  h * ( sigma_ * (self.a2 - self.a1) )
        self.a2 += h * ( self.a1*(r_ - self.a3) - self.a2 + action[0] )
        self.a3 += h * (self.a1 * self.a2 - b_ * self.a3)

        s_ = [self.a1, self.a2,self.a3]

        reward = 1/((self.a1 * self.a1 + self.a2 * self.a2 + self.a3 * self.a3 + 0.1 * action[0] * action[0] )+0.0001)

        done = False

        if (self.a1 * self.a1 + self.a2 * self.a2 + self.a3 * self.a3 ) < 0.00000001:
            done = True

        self.step_num += 1

        return s_, reward, done, []
    
def plot_state(states,cfg,tag = 'test'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} for {cfg['env_name']}")
    plt.xlabel('epsiodes')
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    states = np.array(states)
    colors = np.array(range(len(states[:,0])))
    plt.scatter(states[:,0],states[:,1], label='states',c= colors,s = 2 ,cmap= 'viridis')
    plt.legend()
    plt.colorbar()
    plt.show()
    
def main():
    state = env.reset()
    states = []
    for i in range(1000):
        state,_,done,_ = env.step([0])
        states.append(state)
        if done:
            break
    plot_state(state)

if __name__ == '__main__':
    main()
