 #coding:UTF-8
import gym
from gym.spaces import Box
import numpy as np

import rospy

from SAC import SAC_Agent
from ReplayBuffer import RandomBuffer, device

from env import Gazebo_env
import argparse
import os
# import tensorflow.compat.v1 as tf
import time
import scipy.io as sio
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import matplotlib.pyplot as plt



def norm(env, a):
    a_norm = []
    for i in range(env.action_space.shape[0]):
        a_norm.append(-1. + (2 / (env.action_space.high[i] - env.action_space.low[i])) * (
                a[i] - env.action_space.low[i]))
    return a_norm


def s_norm(env, obs):
    obs_norm = []
    for i in range(env.observation_space.shape[0]):
        # print(obs[i])
        obs_norm.append(0. + (1) / (env.observation_space.high[i] - env.observation_space.low[i]) * (
                obs[i] - env.observation_space.low[i]))
    return obs_norm


def decouple_norm(env, a_norm):
    a = []
    for i in range(env.action_space.shape[0]):
        a.append((a_norm[i] - (-1)) * ((env.action_space.high[i] - env.action_space.low[i]) / (1 - (-1))) +
                 env.action_space.low[i])
    return a

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--episodes', default=5000, type=int)
    parser.add_argument("--max_steps", type=int, default=400, help="maximum max_steps length")  # 每个episode的步数为400步
    parser.add_argument('--saveData_dir', default="./save/UAV_target",
                        help="directory to store all experiment data_BCQ")
    parser.add_argument('--saveModel_dir', default='./save/UAV_target',
                        help="where to store/load network weights")
    parser.add_argument('--load_dir', default='./save/UAV_randomtarget_math',
                        help="where to load network weights")
    parser.add_argument('--load_Q_dir', default='./save/UAV_TD3_pretrain',
                        help="where to load network weights")
    parser.add_argument('--load_littleQ_dir', default='./save/UAV_TD3_littleQ',
                        help="where to load network weights")
    parser.add_argument('--getdata_frequency', default=50, type=int,
                        help="how often to getdata")
    parser.add_argument('--getcritic_frequency', default=50, type=int,
                        help="how often to getdata")
    parser.add_argument('--checkpoint_frequency', default=1, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument("--restore", default=False, action="store_true")
    parser.add_argument('--memory_size', default=100000, type=int)
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    # BC 数据个数
    parser.add_argument('--demo_memory_size', default=100000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--ebatch_size', default=100, type=int)
    parser.add_argument('--batch_norm', default=True, action="store_true")
    parser.add_argument('--ou_mu', default=0, type=float,
                        help="OrnsteinUhlenbeckActionNoise mus for each action for each agent")
    parser.add_argument('--ou_theta', default=0.15, type=float,
                        help="OrnsteinUhlenbeckActionNoise theta for each agent")
    parser.add_argument('--ou_sigma', default=0.3, type=float,
                        help="OrnsteinUhlenbeckActionNoise sigma for each agent")
    parser.add_argument('--getPID', default=False, action="store_true",
                        help="reduces exploration substantially")
                        
    args = parser.parse_args()
    print(args)
    
    for train_count in range(1):
        #print("start")
        env = Gazebo_env(None,None)
        max_steps = args.max_steps  # max_steps per episode
        assert isinstance(env.observation_space, Box), "observation space must be continuous"
        assert isinstance(env.action_space, Box), "action space must be continuous"

        # Randomly initialize critic,actor,target critic, target actor network  and replay buffer
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]
        a_bound = env.action_space.high
        HIDDEN_DIM = 64
        POLICY_TARGET_UPDATE_INTERVAL = 3  # delayed steps for updating the policy network and target networks
        replay_buffer = RandomBuffer(s_dim,a_dim,Env_with_dead=True)# 第三个参数有关人为终止与机器人摔倒的区别，以后再分析
        agent = SAC_Agent(s_dim, a_dim,alpha = 0.12)
        reward_per_episode = 0
        total_reward = 0
        print("Number of States:", s_dim)
        print("Number of Actions:", a_dim)
        print("Number of Steps per episode:", max_steps)
        # saving reward:
        reward_st = np.array([0])
        start_time = time.time()
        # 存储数据列表
        reward_each_episode = []
        action_each_episode = []
        state_each_episode = []
        time_each_episode = []
        obs_each_episode = []
        done_each_episode = []
        d_state_each_episode = []
        VAR = 3
        frame_idx = 0
        ######################Hyperameters#########################
        EXPLORE_STEPS = 1000
        EXPLORE_NOISE_SCALE = 0.5  # range of action noise for exploration
        UPDATE_ITR = 3  # repeated updates for single step
        EVAL_NOISE_SCALE = 0.5  # range of action noise for evaluation of action value
        REWARD_SCALE = 1.  # value range of reward
        #################################################################
        t_state = time.time()
        #if args.restore:
        #   agent.load()

        if not args.testing:
            
            # agent_offline.imitationUpdate(args.batch_size, args.saveModel_dir)
            # agent.criticUpdate(args.batch_size, args.saveModel_dir)

            # agent_offline.criticUpdate(args.load_littleQ_dir)
            # agent.loadQ(args.load_Q_dir)
            while not rospy.is_shutdown():
                env.ramp_up()
                for episode in range(args.episodes):

                
                    # print('a step time__', time.time() - start

                    print("==== Starting episode No.", episode+1, "test====", "\n")
                    observation = env.reset_safe()#初始化，返回状态归一化的值 
                    # observation = env.reset()
                    
                    # env.ramp_up()
                    # print(1)
                    # observation = env.new_state
                    # observation = s_norm(env,observation)
                    # observation = np.array(observation)
                    reward_per_episode = 0  # 用来存储一个episode的总和
                    # 存储每个episode的数据
                    state_steps = []
                    reward_steps = []
                    action_steps = []
                    obs_steps = []
                    done_steps = []
                    d_state_steps = []
                    
                    steps = 0
                    #### one long episode training
                    # if episode == 0:
                    #     max_steps = 5000
                    # else:
                    #     max_steps = args.max_steps
                    start_time_episode = time.time()
                    
                    for t in range(max_steps):
                        # rendering environmet (optional)
                        start = time.time()
                        # env.render()  # test

                        state = env.norm_now_state()
                        # if len(agent.memory.storage) <= args.batch_size:
                        #     action = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
                        # else:
                        #     action = agent.select_action(state)
                        # print(state)
                        if replay_buffer.size < 512:  # Take the random actions in the beginning for the better exploration
                            action = env.action_space.sample()
                        else:
                            action = agent.select_action(state,deterministic=False, with_logprob=False)
                        # action = agent.select_action(state)
                        # action = action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])
                        # action = action.clip(env.action_space.low, env.action_space.high)
                        # print("action:",action)
                        # action = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
                        # action = [0.5,0.5,0.5]
                        # print(action)

                        # print(type(action))
                        # action = decouple_norm(env,action)
                        # print("Action at step", t, " :", action, "\n")
                        # start = time.time()
                        #print("step time:",time.time()-start_time_episode)
                        print("input action:",action)
                        observation, reward, done = env.step(action, t)#state要限制住
                        #start_time_episode = time.time()
                        print("pos:",env.state_pos[0:3],"vel:",env.state_pos[3:6],"angle:",env.now_angle,"angular",env.state_ang[9:12],"reward:",reward)
                        #print("now twist is",env.state_ang)
                        # observation = s_norm(env, observation)

                        replay_buffer.add(state, action, reward, observation, done)
                        # if t+1 % 10 == 0:
                        #     print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                        if replay_buffer.size>= args.batch_size:
                            print("start train!")
                            #start_train= time.time() 
                            agent.train(replay_buffer)#训练时间太长　约为发送动作指令时间的１／７，改善方法一：增长动作指令发送时长 改善方法二：将目前时刻的状态输入模型
                            #print("trian time:",time.time()-start_train)

                        d_state = observation - state

                        # print('observation', observation)
                        # 每个episode、每一步的数据
                        d_state_steps.append(d_state)
                        reward_steps.append(reward)
                        state_steps.append(state)
                        # reward += env.rewardshaping(state, observation, target, df=0.8)

                        action_steps.append(action)
                        obs_steps.append(observation)
                        done_steps.append(done)
                        # print(reward)
                        # add s_t,s_t+1,action,reward to experience memory
                        reward_per_episode += reward
                        frame_idx += 1
                        # check if episode ends:
                        # print(t)
                        # print(done)

                        # print(time.time() - start)

                        if (done or (t == max_steps - 1)or env.region_out):
                            # if (done or (t == max_steps - 1) ):
                            # if ((t == max_steps - 1)):
                            # Testbc
                            # dist_c = agent.disCriticCacul('./save/xplane_TD3_SUCCESS')
                            # dist_p = agent.disPolicyCacul('./save/xplane_TD3_SUCCESS')
                            # critic_dist = np.append(critic_dist, dist_c)
                            # policy_dist = np.append(policy_dist, dist_p)
                            # print('EPISODE:  ', episode, ' Steps: ', t, ' Total Reward: ', reward_per_episode)
                            # print("Printing reward to file")
                            # exploration_noise.reset()  # reinitializing random noise for action exploration
                            reward_st = np.append(reward_st, reward_per_episode)
                            state_each_episode.append(state_steps)
                            reward_each_episode.append(reward_steps)
                            action_each_episode.append(action_steps)
                            obs_each_episode.append(obs_steps)
                            done_each_episode.append(done_steps)
                            d_state_each_episode.append(d_state_steps)
                            time_each_episode.append(time.time() - start_time_episode)
                            #np.savetxt('episode_reward.txt', reward_st, newline="\n")
                            # np.savetxt('episode_critic_dist.txt', critic_dist, newline="\n")
                            # np.savetxt('episode_policy_dist.txt', policy_dist, newline="\n")
                            print('\n\n')
                            print("done: {}, t: {}, env.region_out: {}".format(done,t,env.region_out))
                            env.region_out = False
                            env.success_count = 0
                            env.success_flag = False
                            break

                        # print('a step time__', time.time() - start)

                    if (episode + 1) % args.checkpoint_frequency == 0:
                        if not os.path.exists('model'): os.mkdir('model')
                        agent.save(dict_a="./model/sac_actor_test.pth",dict_c="./model/sac_q_critic_test.pth")

                    print("saving model to 'model'")
                    print("steps: {}, episodes: {}, episode reward: {}, time: {} \n".format(
                        t, episode, reward_per_episode, round(time.time() - start_time, 3)))
                    """ if not os.path.exists('data'):
                        os.makedirs('data')
                    sio.savemat('./data/data' + str(train_count) + '.mat',
                                {'episode_reward': reward_each_episode,
                                'episode_action': action_each_episode,
                                'episode_state': state_each_episode,
                                'episode_obs': obs_each_episode,
                                'episode_d_state': d_state_each_episode,
                                'episode_time': time_each_episode
                                }) """
                # 保存数据
                plot_rewards(reward_st)
                break
                

        else:

            agent.load(dict_a="./model/sac_actor1.pth",dict_c="./model/sac_q_critic1.pth")

            print('finish load,begin to test!')
            env.ramp_up()
            # state = np.array(state)
            # state = state.astype(np.float32)
            # agent.policy_net([state])
            
            # agent.imitationUpdate(args.ebatch_size)
            
            # episode_times = []
            for episode in range(10):
                # observation = env.reset()
                observation = env.reset_safe()
                # observation = np.array(observation)
                # observation = observation.astype(np.float32)
                reward_per_episode = 0
                start = time.time()
                state_steps = []
                reward_steps = []
                action_steps = []
                obs_steps = []
                done_steps = []
                d_state_steps = []
                pos_steps=[]
                # if episode == 0:
                #     max_steps = 2500
                # else:
                #     max_steps = args.max_steps
                max_steps = 500
                for step in range(max_steps):
                    state = observation

                    action = agent.select_action(state,deterministic=True, with_logprob=False)
                    # action = action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])
                    # action = action.clip(env.action_space.low, env.action_space.high)
                    start = time.time()
                    # print("Action at step", step, " :", action, "\n")
                    observation, reward, done = env.step(action, step)
                    # observation = np.array(observation)
                    # observation = observation.astype(np.float32)
                    pos = env.next_state[0:3]
                    # state = np.array(state)
                    # state = state.astype(np.float32)

                    # d_state = observation - state
                    # d_state = np.array(d_state)
                    # d_state = d_state.astype(np.float32)

                    # d_state_steps.append(d_state)
                    reward_steps.append(reward)
                    # reward += env.rewardshaping(state, observation, target, df=0.8)

                    reward_per_episode += reward
                    action_steps.append(action)
                    obs_steps.append(observation)
                    state_steps.append(state)
                    pos_steps.append(pos)
                    print("state:",env.next_state,"action:",action,"reward:",reward)
                    print("now twist is",env.state_ang)
                    # if (time.time() - start) < 0.1:
                    #     time.sleep(0.1 - time.time() + start)
                    # if (done or (step == max_steps - 1)):
                    if ((step == max_steps - 1) or done or env.region_out):
                        # if (done or (t == max_steps-1) ):
                        print('EPISODE:  ', episode, ' Steps: ', step, ' Total Reward: ', reward_per_episode)

                        print("Printing reward to file")
                        # exploration_noise.reset()  # reinitializing random noise for action exploration
                        reward_st = np.append(reward_st, reward_per_episode)
                        reward_each_episode.append(reward_steps)
                        action_each_episode.append(action_steps)
                        state_each_episode.append(state_steps)
                        d_state_each_episode.append(d_state_steps)
                        done_each_episode.append(done_steps)
                        obs_each_episode.append(obs_steps)
                        time_each_episode.append(time.time() - start)
                        # env.crash_flag = False
                        print(time.time() - start)
                        print('\n\n')
                        print("done: {}, t: {}, env.region_out: {}".format(done,step,env.region_out))
                        env.region_out = False
                        env.success_count = 0
                        env.success_flag = False
                        break

                print(
                    'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        episode + 1, 10, reward_per_episode,
                        time.time() - t_state
                    )
                )
                plot_state(pos_steps)
            plot_rewards(reward_st)
            """ if not os.path.exists('data'):
                    os.makedirs('data')
                sio.savemat('./data' + '/data_test_best.mat', {'episode_reward': reward_each_episode,
                                                                        'episode_action': action_each_episode,
                                                                        'episode_state': state_each_episode,
                                                                        'episode_d_state': d_state_each_episode,
                                                                        'episode_obs': obs_each_episode,
                                                                        'episode_time': time_each_episode
                                                                        }) """

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
    
def plot_state(states):
    fig = plt.figure()  # 创建一个图形实例，方便同时多画几个图
    states = np.array(states)
    colors = np.array(range(len(states[:,0])))
    ax = plt.axes(projection='3d')
    ax.scatter(states[:,0],states[:,1], states[:,2],label='states',c = colors , s =1 ,cmap =  'viridis' )
    ax.scatter(5,0,1.5,marker='^')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z") 
    ax.set_xlim(3,7)
    ax.set_ylim(-2,2)
    ax.set_zlim(0.5,2.5)
    ax.plot3D(states[:,0],states[:,1], states[:,2],'gray')    #绘制空间曲线
    
if __name__ == '__main__':
    main()
