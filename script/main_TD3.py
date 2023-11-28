import gym
from gym.spaces import Box
import numpy as np

import rospy

# from TD3_offline import TD3_offline
# from TD3_offline import Replay_buffer_offline as ReplayBuffer_offline
from TD3 import TD3
from TD3 import Replay_buffer as ReplayBuffer

from RSMPC_env import Gazebo_env
import argparse
import os
# import tensorflow.compat.v1 as tf
import time
import scipy.io as sio
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path


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
    parser.add_argument("--max_steps", type=int, default=200, help="maximum max_steps length")  # 每个episode的步数为400步
    parser.add_argument('--saveData_dir', default="./save/UAV_offlinespread_1",
                        help="directory to store all experiment data_BCQ")
    parser.add_argument('--saveModel_dir', default='./save/UAV_offlinespread_1',
                        help="where to store/load network weights")
    parser.add_argument('--load_dir', default='./save/UAV_offlinespread_1',
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
    parser.add_argument('--batch_size', default=128, type=int)
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
    
    for train_count in range(1):
        # session = tf.Session()
        env = Gazebo_env(None,None)
        # env = gym.make('Pendulum-v0')
        max_steps = args.max_steps  # max_steps per episode
        assert isinstance(env.observation_space, Box), "observation space must be continuous"
        assert isinstance(env.action_space, Box), "action space must be continuous"

        # Randomly initialize critic,actor,target critic, target actor network  and replay buffer
        # session.run(tf.initialize_all_variables())
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]
        # a_dim = 4
        a_bound = env.action_space.high
        HIDDEN_DIM = 64
        POLICY_TARGET_UPDATE_INTERVAL = 3  # delayed steps for updating the policy network and target networks
        replay_buffer = ReplayBuffer(args.memory_size)

        # replay_buffer_offline = ReplayBuffer_offline(args.demo_memory_size)

        agent = TD3(s_dim, a_dim, a_bound, replay_buffer)
        # agent_offline = TD3_offline(s_dim, a_dim, a_bound, replay_buffer_offline)

        # 初始化demo

        # agent.initDemoBuffer('./save/xplane_TD3_pretrain/data_expert.mat', args.getdata_frequency)
        # agent_offline.initDemoBuffer('./save/xplane_TD3_dim5/data_expert_real_expert_pitch_roll_4action.mat', args.getdata_frequency)
        # agent.initCriticBuffer('./save/xplane_TD3_dim5/data_expert_Q.mat', args.getcritic_frequency)
        # agent.initCriticBuffer('./save/xplane_TD3_QtoPID/data_noise.mat', args.getcritic_frequency)

        # agent_offline.initBuffer('./save/xplane_TD3_traindata/data0.mat', 200)
        # exploration_noise = OUNoise(a_dim, args.ou_mu, args.ou_theta, args.ou_sigma)

        reward_per_episode = 0
        total_reward = 0
        print("Number of States:", s_dim)
        print("Number of Actions:", a_dim)
        print("Number of Steps per episode:", max_steps)
        # saving reward:
        reward_st = np.array([0])
        # critic_dist = np.array([0])
        # policy_dist = np.array([0])
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
        # 目标值
        # target_PID = [0, 0, 0, 0, 4000, 50]  # heading_rate roll pitch heading altitude KIAS
        # target = [0, 0, 0, 0, 0, 0, 4000, 50]
        # Xplane_pid = xplanePID(target_PID)  # heading_rate roll pitch heading altitude KIAS
        # state = env.reset()
        # state = np.array(state)
        # state = state.astype(np.float32)
        # agent.actor(state)
        # agent.actor_target(state)
        # agent_offline.actor(state)
        # agent_offline.actor_target(state)
        frame_idx = 0
        ######################Hyperameters#########################
        EXPLORE_STEPS = 1000
        EXPLORE_NOISE_SCALE = 0.5  # range of action noise for exploration
        UPDATE_ITR = 3  # repeated updates for single step
        EVAL_NOISE_SCALE = 0.5  # range of action noise for evaluation of action value
        REWARD_SCALE = 1.  # value range of reward
        #################################################################
        t_state = time.time()
        if args.restore:
            agent.load(args.load_dir)

        if not args.testing:
            
            # agent_offline.imitationUpdate(args.batch_size, args.saveModel_dir)
            # agent.criticUpdate(args.batch_size, args.saveModel_dir)

            # agent_offline.criticUpdate(args.load_littleQ_dir)
            # agent.loadQ(args.load_Q_dir)
            # env.reset()
            env.ramp_up()
            
            
            for episode in range(args.episodes):

            
                # print('a step time__', time.time() - start

                print("==== Starting episode no:", episode, "test====", "\n")
                env.reset()
                
                
                # print(env.region_out)
                
                # env.ramp_up()
                # print(1)
                observation = env.new_state
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

                    state = observation
                    
                    # if len(agent.memory.storage) <= args.batch_size:
                    #     action = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
                    # else:
                    #     action = agent.select_action(state)
                    # print(state)
                    action = agent.select_action(state)
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
                    observation, reward, done = env.step(action, t)
                    print("state:",state[0:3],"action:",action,"reward:",reward)
                    # observation = s_norm(env, observation)

                    agent.memory.push((state, observation, action, reward, done))
                    # if t+1 % 10 == 0:
                    #     print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                    if len(agent.memory.storage) >= args.batch_size:
                        agent.update(1)

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

                        print("done:",done,"t:", t,"out:",env.region_out)
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
                        np.savetxt('episode_reward.txt', reward_st, newline="\n")
                        # np.savetxt('episode_critic_dist.txt', critic_dist, newline="\n")
                        # np.savetxt('episode_policy_dist.txt', policy_dist, newline="\n")
                        print('\n\n')
                        steps = t
                        
                        break

                    # print('a step time__', time.time() - start)

                if (episode + 1) % args.checkpoint_frequency == 0:
                    if not os.path.exists(args.saveModel_dir):
                        os.makedirs(args.saveModel_dir)
                    agent.save(args.saveModel_dir)

                print("saving model to {}".format(args.saveModel_dir))
                print("steps: {}, episodes: {}, episode reward: {}, time: {} \n".format(
                    t, episode, reward_per_episode, round(time.time() - start_time, 3)))
                if not os.path.exists(args.saveData_dir):
                    os.makedirs(args.saveData_dir)
                sio.savemat(args.saveData_dir + '/data' + str(train_count) + '.mat',
                            {'episode_reward': reward_each_episode,
                            'episode_action': action_each_episode,
                            'episode_state': state_each_episode,
                            'episode_obs': obs_each_episode,
                            'episode_d_state': d_state_each_episode,
                            'episode_time': time_each_episode
                            })


        else:

            agent.load(args.load_dir)

            env.reset()
            
            # state = np.array(state)
            # state = state.astype(np.float32)
            # agent.policy_net([state])
            observation = env.new_state
            # agent.imitationUpdate(args.ebatch_size)

            # episode_times = []
            for episode in range(args.episodes):
                env.reset()
                env.ramp_up()
                # observation = np.array(observation)
                # observation = observation.astype(np.float32)
                episode_reward = 0
                start = time.time()
                state_steps = []
                reward_steps = []
                action_steps = []
                obs_steps = []
                done_steps = []
                d_state_steps = []

                # if episode == 0:
                #     max_steps = 2500
                # else:
                #     max_steps = args.max_steps
                max_steps = 500
                for step in range(max_steps):
                    state = observation

                    action = agent.select_action(state)
                    # action = action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])
                    action = action.clip(env.action_space.low, env.action_space.high)
                    start = time.time()
                    # print("Action at step", step, " :", action, "\n")
                    observation, reward, done = env.step(action, step)
                    # observation = np.array(observation)
                    # observation = observation.astype(np.float32)

                    # state = np.array(state)
                    # state = state.astype(np.float32)

                    # d_state = observation - state
                    # d_state = np.array(d_state)
                    # d_state = d_state.astype(np.float32)

                    # d_state_steps.append(d_state)
                    reward_steps.append(reward)
                    # reward += env.rewardshaping(state, observation, target, df=0.8)

                    episode_reward += reward
                    action_steps.append(action)
                    obs_steps.append(observation)
                    state_steps.append(state)
                    # if (time.time() - start) < 0.1:
                    #     time.sleep(0.1 - time.time() + start)
                    # if (done or (step == max_steps - 1)):
                    if (done or (step == max_steps - 1) or env.region_out):
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

                        break

                print(
                    'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        episode + 1, args.episodes, episode_reward,
                        time.time() - t_state
                    )
                )
                if not os.path.exists(args.saveData_dir):
                    os.makedirs(args.saveData_dir)
                sio.savemat(args.saveData_dir + '/data_test_best.mat', {'episode_reward': reward_each_episode,
                                                                        'episode_action': action_each_episode,
                                                                        'episode_state': state_each_episode,
                                                                        'episode_d_state': d_state_each_episode,
                                                                        'episode_obs': obs_each_episode,
                                                                        'episode_time': time_each_episode
                                                                        })


if __name__ == '__main__':
    main()