import numpy as np
from scipy.spatial.transform import Rotation

theta_raw = np.array([-0.3,-0.3,-0.3],dtype=float)
omega_b = np.array([-0.3,-0.3,-0.3],dtype=float)
rot = Rotation.from_euler('xyz',theta_raw).as_matrix()
state_ang = np.concatenate((rot[0], rot[1], rot[2], omega_b), axis=0) # type: ignore

print(rot[0],rot[1],rot[2],omega_b)
print(state_ang.shape)

""" for x in abs(angle):
    if x>0.3:
        print("big")
        break """


""" import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Stuart_Landau_oscillator
def plot_state(states):
    fig = plt.figure()  # 创建一个图形实例，方便同时多画几个图
    states = np.array(states)
    colors = np.array(range(len(states[:,0])))
    ax = plt.axes(projection='3d')
    ax.scatter(states[:,0],states[:,1], states[:,2],label='states',c = colors , s =1 ,cmap =  'viridis' )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    ax.set_zlim(-10,50)
    ax.plot3D(states[:,0],states[:,1], states[:,2],'gray')    #绘制空间曲线

    plt.show()

def plot_state_2(states):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    states = np.array(states)
    colors = np.array(range(len(states[:,0])))
    plt.scatter(states[:,0],states[:,1], label='states',c= colors,s = 2 ,cmap= 'viridis')
    plt.legend()
    plt.colorbar()
    plt.show()
    
def plot_states(states):
    fig = plt.figure()  # 创建一个图形实例，方便同时多画几个图
    states = np.array(states)
    colors = np.array(range(len(states[:,0])))
    ax = plt.axes(projection='3d')
    z = np.linspace(0,13,1000)
    x = 5*np.sin(z)
    y = 5*np.cos(z)
    zd = 13*np.random.random(100)
    xd = 5*np.sin(zd)
    yd = 5*np.cos(zd)
    ax.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
    ax.plot3D(x,y,z,'gray')    #绘制空间曲线

    plt.show()
    
def main():
    env=Stuart_Landau_oscillator.env()
    state = env.reset()
    states = []
    for i in range(5000):
        state,reward,done,_ = env.step([0])
        print(state,reward,done)
        states.append(state)

    plot_state_2(states)

if __name__ == '__main__':
    main() """
""" import numpy as np

import rospy
from env import Gazebo_env

def main():
    env = Gazebo_env(None,None)
    while not rospy.is_shutdown():
        env.ramp_up()
        for epi in range(300):
            print("==== Starting episode No.", epi+1, "test====", "\n")
            env.reset_safe()
            for _  in range(500):
                state = env.velocity_test()
                print("state:",state)  
        
if __name__ == '__main__':
    main() """