import gym.spaces as spaces
import numpy as np

class env:
    def __init__(self):
        self.a1 = 1.
        self.a2 = 0.
        self.step_num = 0
        self.action_space = spaces.Box(low=np.array(
            [-1]), high=np.array([1]), dtype=np.float32)

    def reset(self):
        self.a1 = 1.
        self.a2 = 0.
        self.step_num = 0

        return [self.a1, self.a2]

    def step(self, action):

        h = 0.1
        sigma = 0.1 - self.a1**2 - self.a2**2

        self.a1 +=  h * ( sigma * self.a1 - self.a2 )
        self.a2 += h * ( sigma * self.a2 + self.a1 + action[0] )

        # if self.a1 < -bound:
        #     self.a1 = -bound
        # if self.a1 > bound:
        #     self.a1 = bound
        # if self.a2 < -bound:
        #     self.a2 = -bound
        # if self.a2 > bound:
        #     self.a2 = bound
        # print('action',action)
        # print('state',self.a1,self.a2)



        s_ = [self.a1, self.a2]

        #reward = - (self.a1 * self.a1 + self.a2 * self.a2 + (sigma - 0.1 )**2 )
        reward = -100 *(self.a1 * self.a1 + self.a2 * self.a2 + 0.1 * action[0] **2 )
        done = False

        if (self.a1 * self.a1 + self.a2 * self.a2) < 0.0000001:
            done = True

        self.step_num += 1

        return s_, reward, done, []