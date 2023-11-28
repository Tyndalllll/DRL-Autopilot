 #coding:UTF-8
import logging
from scipy.spatial.transform import Rotation
from scipy import io as sio
import numpy as np
import time
import gym.spaces as spaces
import gym
from geometry_msgs.msg import Pose, Twist, TwistStamped
from gazebo_msgs.msg import ModelStates
from mavros_msgs.msg import ActuatorOutputsDRL
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import CommandBool, SetMode, SetModeRequest
from gazebo_msgs.srv import SetModelState  # 设置模型状态、得到模型状态
from std_msgs.msg import Bool
import rospy
import os
import sys
import threading
import random
import random
sys.path.append(os.getcwd())



log = logging.getLogger(__name__)

class env():
    def __init__(self):
        self.a1= 3.
        self.a2 =-2.
        self.a3 =0.5
        self.pos_x_low = 3
        self.pos_x_high = 7
        self.pos_y_low = -2
        self.pos_y_high = 2
        self.pos_z_low = 0.5
        self.pos_z_high = 2.5
        self.action_space = spaces.Box(low=np.array(
            [-1,-1,-1]), high=np.array([1,1,1]), dtype=np.float32)
        self.state_space = spaces.Box(low=np.array(
            [self.pos_x_low,self.pos_y_low,self.pos_z_low,-2,-2,-1,-1,-1,-1]), 
                                      high=np.array([self.pos_x_high,self.pos_y_high,self.pos_z_high,2,2,1,1,1,1]), dtype=np.float32)

        self.target_pos = np.array([5,0,1.5,0,0,0])
        self.velocity = np.array([0,0,0])
        self.new_velocity = np.array([0,0,0])
        self.pos = np.array([self.a1,self.a2,self.a3])
        self.last_pos = np.array([0,0,0])
        self.dis = self.pos - self.target_pos[0:3]
        self.success_count=0
        self.success_flag = False
        self.region_out =False

    def reset(self):
        self.a1 = np.random.uniform(self.pos_x_low+0.5,self.pos_x_high-0.5)
        self.a2 = np.random.uniform(self.pos_y_low+0.5,self.pos_y_high-0.5)
        self.a3 = np.random.uniform(self.pos_z_low+0.2,self.pos_z_high-0.2)
        self.velocity = np.array([0,0,0])
        self.new_velocity = np.array([0,0,0])

        self.pos = np.array([self.a1,self.a2,self.a3])
        self.dis = self.pos - self.target_pos[0:3]
        state = np.concatenate([self.pos,self.dis,self.velocity])
        norm_state = (state-self.state_space.low)/(self.state_space.high-self.state_space.low)
        print("reset pos :",self.pos)
        return norm_state
    
    def random_target(self):
        x = np.random.uniform(self.pos_x_low+0.5,self.pos_x_high-0.5)
        y = np.random.uniform(self.pos_y_low+0.5,self.pos_y_high-0.5)
        z = np.random.uniform(self.pos_z_low+0.2,self.pos_z_high-0.2)
        self.target_pos =np.array([x,y,z,0,0,0])
        
    def rewardCacul(self, t, action):
        self.dis = self.pos - self.target_pos[0:3]
        dist = np.linalg.norm(self.dis)
        delt_dist = np.linalg.norm(self.dis)- np.linalg.norm((np.array(self.target_pos[0:3]) - np.array(self.last_pos)))

        if t == 0:
            rew = 0
        else:
            # reward_ang = -0.5 * (self.state_ang[0] - self.target_ang[0])**2 - 0.5*(self.state_ang[1]- self.target_ang[1])**2
            # reward_ang_po = -(self.state_ang[0]-self.last_state_ang[0])**2 - (self.state_ang[1]-self.last_state_ang[1])**2 - (self.state_ang[2]-self.last_state_ang[2])**2
            # reward_pos = -0.5 * (self.state_pos[0] - self.target_pos[0])**2 - 0.5*(self.state_pos[1]- self.target_pos[1])**2 - 0.5*(self.state_pos[2]- self.target_pos[2])**2
            # reward_pos_po = -(self.state_pos[0]-self.last_state_pos[0])**2 - (self.state_pos[1]-self.last_state_pos[1])**2 - (self.state_pos[2]-self.last_state_pos[2])**2
            # rew =  reward_pos + reward_pos_po
            # rew = - 0.1*delt_dist - 0.1 * velocity_stable - 100 * accelerate_stable
            rew = - 1*dist - 3*delt_dist
        if dist<0.1:
            rew =rew+6
            self.success_count+=1
            if self.success_count>=10:
                self.success_flag = True
                rew +=300
        if self.pos[0]<=self.pos_x_low or self.pos[0]>=self.pos_x_high or self.pos[1]<=self.pos_y_low or self.pos[1]>=self.pos_y_high or self.pos[2]<=self.pos_z_low or self.pos[2]>=self.pos_z_high:
            print(f"When out, pos is {self.pos[0:3]}")
            rew = rew - 300
            self.region_out = True
        # print(rew)
        rew = rew/6
        if rew <-50:
            rew = -1
            #self.out_flag = True
        if rew > 50:rew = 1

        return rew
    
    #TODO:save the velocity and get the position
    def PID(self,vel,tar_vel):
        f = tar_vel-vel
        return 
    
    def step(self, action, t):
        self.new_velocity = action*0.1 +self.new_velocity
        real_vel = self.new_velocity
        real_vel = real_vel.clip(-1,1)
        self.last_pos = self.pos
        self.velocity = real_vel*(1+0.1*(2*random.random()-1))
        self.pos = self.pos + self.velocity*0.1     
        self.dis = self.pos - self.target_pos[0:3]

        #todo: calculate next pos by velocity
        #self.pos = self.PID(self.velocity,real_vel)
        state = np.concatenate([self.pos,self.dis,self.velocity])
        #print("state:",state)
        norm_state = (state-self.state_space.low)/(self.state_space.high-self.state_space.low)
        reward = self.rewardCacul(t,action)
        return norm_state,reward,self.success_flag
        
def main():
    env = env()
    print(env.reset())

if __name__ == '__main__':
    main()

