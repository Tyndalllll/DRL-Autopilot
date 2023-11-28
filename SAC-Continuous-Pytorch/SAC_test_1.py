 #coding:UTF-8
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from gym.spaces import Box
import numpy as np
import rospy
from ReplayBuffer import RandomBuffer, device
import Stuart_Landau_oscillator
import Lorentz_system
import argparse
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def build_net(layer_shape, activation, output_activation):
	'''Build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
		super(Actor, self).__init__()

		layers = [state_dim] + list(hid_shape)
		self.a_net = build_net(layers, h_acti, o_acti)
		self.mu_layer = nn.Linear(layers[-1], action_dim)
		self.log_std_layer = nn.Linear(layers[-1], action_dim)

		self.LOG_STD_MAX = 2
		self.LOG_STD_MIN = -20


	def forward(self, state, deterministic=False, with_logprob=True):
		'''Network with Enforcing Action Bounds'''
		net_out = self.a_net(state)
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习
		std = torch.exp(log_std)
		dist = Normal(mu, std)

		if deterministic: u = mu
		else: u = dist.rsample() #'''  trick of Gaussian'''#
		a = torch.tanh(u)

		if with_logprob:
			# get probability density of logp_pi_a from probability density of u, which is given by the original paper.
			# logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

			# Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
			logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
		else:
			logp_pi_a = None

		return a, logp_pi_a



class Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Q_Critic, self).__init__()
		layers = [state_dim + action_dim] + list(hid_shape) + [1]

		self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = self.Q_1(sa)
		q2 = self.Q_2(sa)
		return q1, q2



class SAC_Agent(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		gamma=0.99,
		hid_shape=(256,256),
		a_lr=3e-4,
		c_lr=3e-4,
		batch_size = 256,
		alpha = 0.2,
		adaptive_alpha = True
	):

		self.actor = Actor(state_dim, action_dim, hid_shape).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

		self.q_critic = Q_Critic(state_dim, action_dim, hid_shape).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_critic_target.parameters():
			p.requires_grad = False

		self.action_dim = action_dim
		self.gamma = gamma
		self.tau = 0.005
		self.batch_size = batch_size

		self.alpha = alpha
		self.adaptive_alpha = adaptive_alpha
		if adaptive_alpha:
			# Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
			self.target_entropy = torch.tensor(-action_dim, dtype=float, requires_grad=True, device=device)
			# We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
			self.log_alpha = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=c_lr)



	def select_action(self, state, deterministic, with_logprob=False):
		# only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)
			a, _ = self.actor(state, deterministic, with_logprob)
		return a.cpu().numpy().flatten()



	def train(self,replay_buffer):
		s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)

		#----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
		with torch.no_grad():
			a_prime, log_pi_a_prime = self.actor(s_prime)
			target_Q1, target_Q2 = self.q_critic_target(s_prime, a_prime)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = r + (1 - dead_mask) * self.gamma * (target_Q - self.alpha * log_pi_a_prime) #Dead or Done is tackled by Randombuffer

		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		#----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
		# Freeze Q-networks so you don't waste computational effort
		# computing gradients for them during the policy learning step.
		for params in self.q_critic.parameters():
			params.requires_grad = 	False

		a, log_pi_a = self.actor(s)
		current_Q1, current_Q2 = self.q_critic(s, a)
		Q = torch.min(current_Q1, current_Q2)

		a_loss = (self.alpha * log_pi_a - Q).mean()
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		for params in self.q_critic.parameters():
			params.requires_grad = 	True
		#----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
		if self.adaptive_alpha:
			# we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
			# if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
			alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()
			self.alpha = self.log_alpha.exp()

		#----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



	def save(self):
		torch.save(self.actor.state_dict(), "./model/sac_actor_test1.pth")
		torch.save(self.q_critic.state_dict(), "./model/sac_q_critic_test1.pth")


	def load(self):
		self.actor.load_state_dict(torch.load("./model/sac_actor_test1.pth"),False)
		self.q_critic.load_state_dict(torch.load("./model/sac_actor_test1.pth"),False)

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
    parser.add_argument("--max_steps", type=int, default=300, help="maximum max_steps length")  # 每个episode的步数为400步
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
    parser.add_argument('--testing', default=True, action="store_true",
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
        env = Lorentz_system.env()
        max_steps = args.max_steps  # max_steps per episode


        # Randomly initialize critic,actor,target critic, target actor network  and replay buffer
        s_dim = 3
        a_dim = 1
        replay_buffer = RandomBuffer(s_dim,a_dim,Env_with_dead=True)# 第三个参数有关人为终止与机器人摔倒的区别，以后再分析
        agent = SAC_Agent(s_dim, a_dim,alpha = 0.12)
        reward_per_episode = 0
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
        frame_idx = 0
        t_state = time.time()

        if not args.testing:
            
            # agent_offline.imitationUpdate(args.batch_size, args.saveModel_dir)
            # agent.criticUpdate(args.batch_size, args.saveModel_dir)

            # agent_offline.criticUpdate(args.load_littleQ_dir)
            # agent.loadQ(args.load_Q_dir)
                for episode in range(500):

                
                    # print('a step time__', time.time() - start

                    print("==== Starting episode No.", episode+1, "test====", "\n")
                    observation = env.reset()#初始化，返回状态归一化的值 状态是当前位置 和 位置与目标点的三个距离 
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
                    reward = 0
                    for t in range(5000):
                        # rendering environmet (optional)
                        start = time.time()
                        # env.render()  # test

                        state = observation
                        # if len(agent.memory.storage) <= args.batch_size:
                        #     action = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
                        # else:
                        #     action = agent.select_action(state)
                        # print(state)
                        if replay_buffer.ptr < 512:  # Take the random actions in the beginning for the better exploration
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
                        observation, reward, done, _ = env.step(action)
                        
                        print("state:",state,"action:",action,"reward:",reward)
                        # observation = s_norm(env, observation)
                        print("obs:",observation)
                        replay_buffer.add(state, action, reward, observation, done)
                        print(f"now store num :{replay_buffer.ptr}")
                        # if t+1 % 10 == 0:
                        #     print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                        if replay_buffer.ptr>= args.batch_size:
                            print("start train!")
                            agent.train(replay_buffer)


                        # print('observation', observation)
                        # 每个episode、每一步的数据
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

                        if (done or (t == max_steps - 1)):
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
                            # np.savetxt('episode_critic_dist.txt', critic_dist, newline="\n")
                            # np.savetxt('episode_policy_dist.txt', policy_dist, newline="\n")
                            print('\n\n')
                            print("done: {}, t: {}".format(done,t))
                            break

                        # print('a step time__', time.time() - start)

                    if (episode + 1) % args.checkpoint_frequency == 0:
                        if not os.path.exists('model'): os.mkdir('model')
                        agent.save()

                    print("saving model to 'model'")
                    print("steps: {}, episodes: {}, episode reward: {}, time: {} \n".format(
                        t, episode, reward_per_episode, round(time.time() - start_time, 3)))
                plot_rewards(reward_st)

                

        else:

            agent.load()

            
            # state = np.array(state)
            # state = state.astype(np.float32)
            # agent.policy_net([state])
            
            # agent.imitationUpdate(args.ebatch_size)
            
            # episode_times = []
            for episode in range(10):
                # observation = env.reset()
                observation = env.reset()
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
                max_steps = 5000
                for step in range(max_steps):
                    state = observation

                    action = agent.select_action(state,deterministic=True, with_logprob=False)
                    # action = action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])
                    # action = action.clip(env.action_space.low, env.action_space.high)
                    start = time.time()
                    # print("Action at step", step, " :", action, "\n")
                    observation, reward, done,_ = env.step(action)
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
                    print(state)
                    if ((step == max_steps - 1) or done):
                        # if (done or (t == max_steps-1) ):
                        print('EPISODE:  ', episode, ' Steps: ', step, ' Total Reward: ', episode_reward)

                        print("Printing reward to file")
                        # exploration_noise.reset()  # reinitializing random noise for action exploration
                        reward_st = np.append(reward_st, episode_reward)
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
                plot_state(state_steps)
                print(
                    'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        episode + 1, args.episodes, episode_reward,
                        time.time() - t_state))
            plot_rewards(reward_st)



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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    ax.plot3D(states[:,0],states[:,1], states[:,2],'gray')    #绘制空间曲线
if __name__ == '__main__':
    main()







