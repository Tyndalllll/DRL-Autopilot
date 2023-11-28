""" with open("./episode_reward.txt", "r") as f:
    data = f.readlines()
    print(data) """
import os
import matplotlib.pyplot as plt

file_path = "SAC-Continuous-Pytorch/episode_reward.txt"

if os.path.exists(file_path):
    f = open(file_path)
    lines = f.readlines()
    rewards=[]
    for line in lines:
        line =  float(line)
        rewards.append(line)
    f.close()
else:
    print("wrong")

print(len(lines))

def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards, tag='train'):
    ''' 画图
    '''
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()
    
plot_rewards(rewards)