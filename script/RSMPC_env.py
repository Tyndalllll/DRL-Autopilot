from px4_controller import controller
from filter import Filter
from time_marker import TimeMarker
from PD_controller import PD
from utils.sim import *
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from scipy.spatial.transform import Rotation
from scipy import io as sio
import numpy as np
import time
import gym.spaces as spaces
import gym
from geometry_msgs.msg import Pose, Twist, TwistStamped,Accel
from gazebo_msgs.msg import ModelStates
from mavros_msgs.msg import ActuatorOutputsDRL
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from gazebo_msgs.srv import SetModelState, GetModelState  # 设置模型状态、得到模型状态
from std_msgs.msg import Bool
from matplotlib.pyplot import flag
import rospy
import os
import sys
import pickle

sys.path.append(os.getcwd())


log = logging.getLogger(__name__)


def save(fname, obj, type='data'):
    if type == 'data':
        sio.savemat(fname, obj)
    elif type == 'model':
        NotImplemented
    else:
        raise NameError('Improper type passed')


def counter_cosine_similarity(c1, c2):
    # cos xiangsidu
    from collections import Counter
    c1 = Counter(c1)
    c2 = Counter(c2)
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))
    return dotprod / (magA * magB)


class Gazebo_env():
    def __init__(self, cfg, metric) -> None:
        rospy.init_node('RSMPC', anonymous=True)

        # data
        self.state = State()
        
        self.last_state = None
        self.arming_req = CommandBool()._request_class()  # CHANGE
        self.set_mode_req = SetMode()._request_class()  # CHANGE
        self.cmd_velocity = TwistStamped()
        self.cmd_pwm = ActuatorOutputsDRL()
        self.local_gazebo = ModelStates()
        self.local_vrpn = PoseStamped()
        self.sleep_rampup = 0.25  # 250ms
        # TODO: diff setting for real exp and sim
        self.rampup_rounds = 20  # n * 1/50Hz = 20n ms
        self.hz = 10
        self.mocap_freq = 250
        self.set_model_state_req = SetModelState._request_class()
        self.original_last_state = None
        self.act_dim = 3
        self.state_dim = 6
        self.obs_space = []
        self.act_space = []
        self.pos_x_low = 3
        self.pos_x_high = 7
        self.pos_y_low = -2
        self.pos_y_high = 2
        self.pos_z_low = 0.5
        self.pos_z_high = 2.5
        self.action_space = spaces.Box(low=np.array(
            [-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        # observations are [x, y, z,vx,vy,vz, dist2targ_x, dist2targ_y, dist2targ_z]
        self.observation_space = spaces.Box(low=np.array(
            [self.pos_x_low, self.pos_y_low,self.pos_z_low, -10, -10, -10]), high=np.array([self.pos_x_high, self.pos_y_high, self.pos_z_high,10, 10, 10]), dtype=np.float32)
        #创建连续空间
        
        self.act_space.append(self.action_space)
        self.obs_space.append(self.observation_space)
        # self.actions = [0 for _ in range(self.act_dim)]
        self.current_velocity = [0 for _ in range(self.act_dim)]
        self.success_flag = False
        self.success_count = 0
        self.region_out = False

        # self.history_len = cfg.models.model.params.history
        # self.set_model_state_res = SetModelState._response_class()
        self.new_state = np.zeros((self.state_dim,), dtype=float)
        self.next_state = np.zeros((self.state_dim,), dtype=float)
        self.last_state_ang = np.zeros((6,), dtype=float)
        self.last_state_pos = np.zeros((6,), dtype=float)
        self.last_action = np.zeros((self.act_dim,), dtype=float)
        self.state_ang = np.zeros((6,), dtype=float)
        self.state_pos = np.zeros((6,), dtype=float)
        self.target_ang = np.zeros((6,), dtype=float)
        self.target_pos = np.zeros((6,), dtype=float)
        self.dist2target_pos = np.zeros((6,), dtype=float)
        self.dist2target_ang = np.zeros((6,), dtype=float)
        self.next_dist2target_pos = np.zeros((6,), dtype=float)
        self.next_dist2target_pos = np.zeros((6,), dtype=float)
        self.last_state_timestamp = 0.  # for init checking
        self.state_timestamp = 0.  # if lost connect to PC or state, stop rollout
        self.continue_timestamp = 0.
        self.connection_duration = 0.2
        self.quat = np.array([0, 0, 0, 1])
        self.pos = np.array([0, 0, 0])
        self.last_quat = np.array([0, 0, 0, 1])
        self.last_pos = np.array([0, 0, 0])
        self.hover_pwm = 1570
        self.rampup_pwm_add = 100
        self.pos_offset = [0, 0, 0]
        self.is_sim = True
        self.filter_state = Filter(dimension=6, window=3)#创建滤波器

        print('Controller Frequency: {} Hz'.format(self.hz))
        # self.cfg = cfg
        # define metric
        self.metric = metric

        # Subcriber
        rospy.Subscriber('mavros/state', State, self.sub_state)#获取目前mavros得到的各种无人机状态
        rospy.Subscriber('exp_continue', Bool, self.callback_exp_continue)#获取当前时间戳
        if True:
            rospy.Subscriber('gazebo/model_states',
                             ModelStates, self.sub_gazebo)
            # rospy.Subscriber('msg_delay', ModelStates, self.sub_gazebo)
        else:
            # sub vrpn / other data sources
            rospy.Subscriber("/vrpn_client_node/pinn_uav0623/pose",
                             PoseStamped, self.sub_vrpn, queue_size=1)

        # Publisher
        # self.local_pos_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.pwm_pub = rospy.Publisher(
            'mavros/actuator_outputs_drl/actuator_sub', ActuatorOutputsDRL, queue_size=1)
        self.vel_pub = rospy.Publisher(
            'mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
        self.pos_pub = rospy.Publisher(
            'mavros/setpoint_position/local', PoseStamped, queue_size=1)
        # ServiceClient
        rospy.loginfo("waiting for ROS services")
        rospy.wait_for_service('mavros/cmd/arming')
        rospy.wait_for_service('mavros/set_mode')
        self.arming_client = rospy.ServiceProxy(
            'mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)
        self.set_model_proxy = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        rospy.loginfo("connection success")

        self.rate = rospy.Rate(self.hz)  # TODO
        self.max_pwm = 1950
        self.min_pwm = 1075

        # Note: ref z coordinate is wrong, but works
        # maybe because it causes a smaller desire lean angle
        self.pos_c = np.array([0., 0., 1.5])  # 参考状态
        self.dpos_c = np.array([0., 0., 0.])  # 参考状态
        self.ddpos_c = np.array([0., 0., 0.])  # 参考状态
        self.psi_c = 0.  # 参考状态
        # self.m = 1.5 # 无人机质量
        # self.pos_pd = PD(self.m)  # 位置环
        self.pos_pd = PD()
        self.first_time = True

    def pwm_normalize(self, pwm):
        pwm_norm = (pwm - self.min_pwm) / (self.max_pwm - self.min_pwm) * 2 - 1
        # for i in range(pwm.shape[-1]):
        #     if pwm[i] < -self.min_pwm or pwm[i] > self.max_pwm:
        #         print('pwm_error:', pwm)
        # pwm_norm = -1print("x:{0},y:{1},z:{2}",self.cmd_velocity.twist.linear.x,self.cmd_velocity.twist.linear.y, self.cmd_velocity.twist.linear.z)

        # elif pwm[i] > 1:
        #     print('pwm_norm_error(> 1):',pwm)
        #     # pwm_norm = 1
        return pwm_norm

    def ramp_up(self):
        '''
        TODO： Do we need to record data when doing ramp up
        '''
        # wait for FCU connection
        # print('FCU connection', self.state.connected)
        # print('mode: ', self.state.mode)
        # print('armed: ', self.state.armed)
        while (not rospy.is_shutdown()) and (not self.state.connected):
            print('FCU connection', self.state.connected)
            self.rate.sleep()

        time.sleep(0.5)
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        rate_5 = rospy.Rate(20)
        for i in range(10):   
            if(rospy.is_shutdown()):
                break
            self.cmd_velocity.twist.linear.x = 0.
            self.cmd_velocity.twist.linear.y = 0.
            self.cmd_velocity.twist.linear.z = 0.
            self.vel_pub.publish(self.cmd_velocity)
            rate_5.sleep()

        while not rospy.is_shutdown() \
                        and (self.state.armed != True or self.state.mode != "OFFBOARD"):
                # and not self.state.armed:
            # print(self.state.armed,self.state.mode)
            if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")
            if self.state.mode == "OFFBOARD" and self.state.armed != True:
                response = self.arming_client(True)
                print(f"arm :{response.success}")
            # else:
            #     print('please arming by Radio Control')
            #     num = eval(input("input anything to continue"))

            # self.set_mode_client(0, "STABILIZED")
            # print(1)
            rate_5.sleep()
        rospy.loginfo("Arming success,mode:{}".format(self.state.mode))
        self.cmd_pwm.usedrl = False
        # time.sleep(0.5)
        # print("x:{0},y:{1},z:{2}",self.cmd_velocity.twist.linear.x,self.cmd_velocity.twist.linear.y, self.cmd_velocity.twist.linear.z)
        # in python type(ActuatorOutput.output) is 'list'
        # rampup_pwm = self.hover_pwm + self.rampup_pwm_add
        # rampup_pwm_norm = (rampup_pwm - self.min_pwm) / (self.max_pwm - self.min_pwm) * 2 - 1
        # self.cmd_pwm.output = [rampup_pwm_norm] * 4
        # self.cmd_pwm.usedrl = True
        # rampup_time_marker = TimeMarker()
        # rampup_time_marker.mark('start')
        # for iter_rampup in range(self.rampup_rounds):
        #     # mark start time and sleep
        #     self.rate.sleep()

        #     # if rospy.is_shutdown() or not self.is_connected():
        #     #     print(f"is connected: {self.is_connected()}")
        #     #     break
        #     # else:
        #     print(f"ramp up pwm: {self.cmd_pwm.output}")
        #     self.pwm_pub.publish(self.cmd_pwm)
        #     rampup_time_marker.mark('pub_{}'.format(iter_rampup))
        # rampup_time_marker.print()
        # print('end ramp up')
        # print("position: ", self.state_pos[0:3])
        # print("vel: ", self.state_pos[3:6])
        # print('orientation:', self.state_ang[0:3])

    def euler_number(self, last_state, state, mag, bound, bound_yaw, output=False):
        '''
        NOTE: detect large euler angle step and large euler angle
        '''
        flag = False

        if abs(state[3]) > np.deg2rad(mag):
            if output:
                print('detect large euler angle step roll:{}'.format(
                    np.rad2deg(state[3])))
            flag = True
        if abs(state[4]) > np.deg2rad(mag):
            if output:
                print('detect large euler angle step pitch:{}'.format(
                    np.rad2deg(state[4])))
            flag = True
        if abs(state[5]) > np.deg2rad(mag):
            if output:
                print('detect large euler angle step yaw:{}'.format(
                    np.rad2deg(state[5])))
            flag = True

        if abs(state[0]) > np.deg2rad(bound):
            if output:
                print('detect large euler angle roll : {}'.format(
                    np.rad2deg(state[0])))
            flag = True
        if abs(state[1]) > np.deg2rad(bound):
            if output:
                print('detect large euler angle pitch : {}'.format(
                    np.rad2deg(state[1])))
            flag = True
        if abs(state[2]) > np.deg2rad(bound_yaw):
            if output:
                print('detect large euler angle yaw : {}'.format(
                    np.rad2deg(state[2])))
            flag = True
        return flag

    # check whether Nano is connected to PC and MoCap
    def is_connected(self) -> bool:

        time_now = time.time()
        if time_now - self.state_timestamp < self.connection_duration \
                and time_now - self.continue_timestamp < self.connection_duration:
            connected_flag = True
        else:
            connected_flag = False
            rospy.logerr('Lost connection!')

        return connected_flag

    # check if this state is safe,based on euler_number()
    # mag: bound of the change of states
    # mag_pos = max_velocity(m/s)
    # bound_pos = min and max in x,y and z axis
    def is_safe(self, last_state_pos, state_pos, last_state, state, bound_pos, mag_pos=3, mag_pos_z=3, mag_ang=1000,
                bound_ang=45, bound_yaw=361, output=False):

        # connection
        flag_connected = self.is_connected()

        # angular
        flag_ang = self.euler_number(
            last_state, state, mag_ang, bound_ang, bound_yaw, output=output)

        # linear
        flag_linear = False

        if abs(state_pos[3]) > mag_pos:
            if output:
                print('detect large pos step in x axis{}'.format(state_pos[3]))
            flag_linear = True
        if abs(state_pos[4]) > mag_pos:
            if output:
                print('detect large pos step in y axis{}'.format(state_pos[4]))
            flag_linear = True
        if abs(state_pos[5]) > mag_pos_z:
            if output:
                print('detect large pos step in z axis{}'.format(state_pos[5]))
            flag_linear = True

        if state_pos[0] < bound_pos[0] or state_pos[0] > bound_pos[1]:
            if output:
                print('detect large position_x : {}'.format(state_pos[0]))
            flag_linear = True
        if state_pos[1] < bound_pos[2] or state_pos[1] > bound_pos[3]:
            if output:
                print('detect large position_y : {}'.format(state_pos[1]))
            flag_linear = True
        if state_pos[2] < bound_pos[4] or state_pos[2] > bound_pos[5]:
            if output:
                print('detect large position_z : {}'.format(state_pos[2]))
            flag_linear = True

        # unsafe flags
        unsafe_flag = flag_ang or flag_linear

        # safe flag
        flag = flag_connected and not unsafe_flag

        # if safe retuen true
        return flag

    def start_ramp_up(self):
        # ramp up
        print("ramp up start")
        while not rospy.is_shutdown():
            print(" not connected! ")  # wait for stable connection
            time.sleep(0.2)  # check in

        if True:
            self.ramp_up()
            print("ramp up end")

    def rewardCacul(self, t, action):

        dist = np.linalg.norm(np.array(self.target_pos[0:3]) - np.array(self.new_state[0:3]))
        delt_dist = np.linalg.norm((np.array(self.target_pos[0:3]) - np.array(self.new_state[0:3])))- np.linalg.norm((np.array(self.target_pos[0:3]) - np.array(self.last_state_pos[0:3])))
        # cos_similarity = np.dot(self.dist2target_pos[0:3] / np.linalg.norm(self.dist2target_pos[0:3]), self.new_state[3:6]/ np.linalg.norm(self.new_state[3:6]))
        cos_similarity = np.dot(self.dist2target_pos[0:3] / np.linalg.norm(self.dist2target_pos[0:3]), action/ np.linalg.norm(action))
        # velocity_stable = np.linalg.norm(np.array(self.new_state[3:6]))
        velocity_stable = np.linalg.norm(action)
        accelerate_stable = np.linalg.norm(np.array(self.last_action) - np.array(action))

        # print(dist, cos_similarity)
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
        

        # print(f"dist:{3*dist}, cos_similarity: {0.8*(cos_similarity+1)}, accelerate_stable:{2 * accelerate_stable}, velocity_stable:{1 *velocity_stable}")
        if rew>5:
            rew = 5
        if rew< -5:
            rew = -5

        
        

            # print()

        # if dist < 0.8:
        #     rew -= 0.1 * velocity_stable - 0.1*accelerate_stable


        # if dist < 0.5:
        #     self.success_count += 1
        #     rew = rew + 1000

        #     if self.success_count > 10:
        #         rew = rew + 1500
        #         self.success_flag = True
        # if self.state_pos[0]<=self.pos_x_low-1 or self.state_pos[0]>=self.pos_x_high+1 or self.state_pos[1]<=self.pos_y_low-1 or self.state_pos[1]>=self.pos_y_high+1 or self.state_pos[2]<=self.pos_z_low-0.4 or self.state_pos[2]>=self.pos_z_high+1:
        #     rew = rew - 500
        #     self.region_out = True




        if dist < 0.3:
            rew = rew + 5
            self.success_count+=1
            if self.success_count>=10:
                self.success_flag = True
                rew +=300

        if self.new_state[0]<=self.pos_x_low or self.new_state[0]>=self.pos_x_high or self.new_state[1]<=self.pos_y_low or self.new_state[1]>=self.pos_y_high or self.new_state[2]<=self.pos_z_low or self.new_state[2]>=self.pos_z_high:
            print(f"When out: {self.new_state[0:3]}")
            rew = rew - 200
            self.region_out = True
        # print(rew)
        rew = rew/5
        return rew
    
#step有一点小问题，但不影响，之后用的时候再改 
    def step(self, action, t):
        # print(action)
        # self.new_state = np.concatenate([self.state_ang,self.state_pos,self.dist2target_pos[0:3]])
        pos = self.state_pos
        ang = self.state_ang
        self.dist2target_pos = pos[0:6] - self.target_pos
        self.new_state = np.concatenate(
            [pos[0:3], self.dist2target_pos[0:3]])

        # print(f"dist2target_pos: {self.dist2target_pos}, target_pos:{self.target_pos}, pos:{pos}")
        # self.norm_new_state = (self.new_state - self.observation_space.low)/(self.observation_space.high-self.observation_space.low)
        # print(self.norm_new_state)
        # real_action = action*500 + np.array([1500,1500,1500,1500])

        # velocity vectort in range(max_steps)
        self.current_velocity += action * 0.1
        real_velocity = self.current_velocity
        real_velocity = real_velocity.clip(-1, 1)
        # print("action:",real_velocity)

        # add velocity control by lilong
        self.cmd_velocity.twist.linear.x = real_velocity[0]
        self.cmd_velocity.twist.linear.y = real_velocity[1]
        self.cmd_velocity.twist.linear.z = real_velocity[2]
        # self.cmd_velocity.twist.linear.x = 0.
        # self.cmd_velocity.twist.linear.y = 0.
        # self.cmd_velocity.twist.linear.z = 0.2
        # self.cmd_velocity.header.stamp = rospy.time()
        # print("x:{0},y:{1},z:{2}".format(self.cmd_velocity.twist.linear.x,
        #       self.cmd_velocity.twist.linear.y, self.cmd_velocity.twist.linear.z))
        self.vel_pub.publish(self.cmd_velocity)

        # real_action = np.array([1800,1800,1800,1800])
        # self.cmd_pwm.output = list(self.pwm_normalize(real_action))
        # print(self.cmd_pwm.output)
        # self.pwm_pub.publish(self.cmd_pwm)
        self.rate.sleep()#等待一段时间，无人机进入了下一个状态
        next_pos = self.state_pos
        next_ang = self.state_ang
        self.next_dist2target_pos = next_pos[0:6] - self.target_pos
        self.next_state = np.concatenate(
            [next_pos[0:3], self.next_dist2target_pos[0:3]])
        self.norm_next_state = (self.next_state - self.observation_space.low)/(self.observation_space.high-self.observation_space.low)
        
        # print(self.state_ang)
        self.last_state_pos = self.new_state[0:6]
        self.last_state_ang = self.new_state[9:12]
        # self.dist2target_ang = self.state_ang - self.target_ang
        # print(self.dist2target_pos)
        rew = self.rewardCacul(t,action)
        # print(state)
        # print(f"time: {t}")


        self.last_action = action
        print(f"error action: {real_velocity - next_pos[0:3]}")
        # obs = np.concatenate([self.state_ang, self.state_pos, self.dist2target_pos[0:3]])
        obs = self.norm_next_state

        # while not rospy.is_shutdown():
        #
        #     # if self.state does not update, state should not update
        #     if np.linalg.norm(self.state_pos - state_pos) < 1e-6:
        #         print('state not update')
        #         self.rate.sleep()  # do we really need this? (or send actions immediately once states update)
        #         continue  # do not control or record this state?
        #     else:
        #         last_state = state
        #         last_state_pos = state_pos
        #         state = self.state_ang
        #         state_pos = self.state_pos
        #
        #
        #     # print('action:{}'.format(action))
        #     self.cmd_pwm.output = self.pwm_normalize(action)
        #     self.pwm_pub.publish(self.cmd_pwm)
        #
        #
        #     if last_euler is None:
        #         cul_reward = 0
        #         last_euler = state[:3]
        #     else:
        #         cul_reward = cul_reward -(0.5 * (state[0] - phi_c)**2 + 0.5*(state[1]- theta_c)**2 + (state[0]-last_euler[0])**2
        #         + (state[1]-last_euler[1])**2 + (state[1]-last_euler[1])**2 )
        #
        #     rews.append(cul_reward / (steps+1))
        #
        #     self.rate.sleep()
        # print('HZ: {}'.format(1/(time.time() - st)))

        return obs, rew, self.success_flag

    def land(self):
        '''
        Land the quadrotor
        '''
        NotImplemented
        # self.cmd_pwm.usedrl=False
        # time.sleep(5)
        # self.cmd_pwm.output = [2000, 2000, 2000, 2000]
        # self.cmd_pwm.usedrl=True

    #reset mode written by lilong
    def reset_safe(self, test=False):
        init_pose = PoseStamped()
        self.current_velocity = 0
        # pos and angle
        init_pose.pose.orientation.w = 1.0
        init_pose.pose.orientation.x = 0.
        init_pose.pose.orientation.y = 0.
        init_pose.pose.orientation.z = 0.

        if self.first_time == True:
            print("Start to fly!!!!!!!!!!!!!!!!11")
            init_pose.pose.position.x = 0.
            init_pose.pose.position.y = 0.
            init_pose.pose.position.z = 2.

            self.pos_pub.publish(init_pose)
            
            for i in range(100):   
                if(rospy.is_shutdown()):
                    break

                self.pos_pub.publish(init_pose)
                time.sleep(0.01)

            time.sleep(5)
            self.first_time = False
            print("end fly")
        
        init_pose.pose.position.x = np.random.uniform(self.pos_x_low+1,self.pos_x_high-1)
        init_pose.pose.position.y = np.random.uniform(self.pos_y_low+1,self.pos_y_high-1)
        init_pose.pose.position.z = np.random.uniform(self.pos_z_low+0.7,self.pos_z_high-0.8)

        # target pos
        self.target_ang = np.array([0, 0, 0, 0, 0, 0])
        self.target_pos = np.array([5, 0, 1.5, 0, 0, 0])
            
        pos_before_reset = np.array([init_pose.pose.position.x,init_pose.pose.position.y,init_pose.pose.position.z])
        sleep = rospy.Rate(50)
        while True:
            pos_after_reset = self.pos
            sleep.sleep()
            try:
                self.pos_pub.publish(init_pose)
                # print(np.linalg.norm(pos_after_reset - pos_before_reset))
            except rospy.ServiceException:
                pass
                # print(np.linalg.norm(pos_after_reset - pos_before_reset))
            if np.linalg.norm(pos_after_reset - pos_before_reset) < 1e-1:
                break

        print(f"pos_after_reset:{pos_after_reset}")
        pos = self.state_pos
        ang = self.state_ang
        self.dist2target_pos = pos[0:6] - self.target_pos
        self.new_state = np.concatenate(
            [self.state_pos[0:3], self.dist2target_pos[0:3]])
        print(f'after reset: {self.new_state}')
        norm_new_state = (self.new_state - self.observation_space.low)/(self.observation_space.high-self.observation_space.low)
        return norm_new_state


    def reset(self, test=False):
        init_pose = Pose()
        init_twist = Twist()
        self.success_count = 0
        self.success_flag = False
        self.region_out = False
        if True:

            # pos and angle
            init_pose.orientation.w = 1.0
            init_pose.orientation.x = 0.
            init_pose.orientation.y = 0.
            init_pose.orientation.z = 0.

            # init_pose.position.x = 0
            # init_pose.position.y = 0
            # init_pose.position.z = 1

            init_pose.position.x = np.random.uniform(self.pos_x_low+1,self.pos_x_high-1)
            init_pose.position.y = np.random.uniform(self.pos_y_low+1,self.pos_y_high-1)
            init_pose.position.z = np.random.uniform(self.pos_z_low+0.7,self.pos_z_high-0.8)

            # print(init_pose.position.x, init_pose.position.y, init_pose.position.z)

            # velocity
            init_twist.angular.x = 0.
            init_twist.angular.y = 0.
            init_twist.angular.z = 0.
            init_twist.linear.x = 0.
            init_twist.linear.y = 0.
            init_twist.linear.z = 0.01


            # target pos
            self.target_ang = np.array([0, 0, 0, 0, 0, 0])
            self.target_pos = np.array([5, 0, 1.5, 0, 0, 0])
            # self.target_pos = np.array([np.random.uniform(self.pos_x_low,self.pos_x_high), np.random.uniform(self.pos_y_low,self.pos_y_high), np.random.uniform(self.pos_z_low,self.pos_z_high), 0, 0, 0])

        else:
            init_pose.orientation.w = 0.992
            init_pose.orientation.x = 0.087
            init_pose.orientation.y = 0.087
            init_pose.orientation.z = -0.008
            init_pose.position.x = 0.
            init_pose.position.y = 0.
            init_pose.position.z = 0

            init_twist.angular.x = 0.
            init_twist.angular.y = 0.
            init_twist.angular.z = 0.
            init_twist.linear.x = 0.
            init_twist.linear.y = 0.
            init_twist.linear.z = 0.

        self.set_model_state_req.model_state.model_name = 'if750a'
        self.set_model_state_req.model_state.pose = init_pose
        self.set_model_state_req.model_state.twist = init_twist
        # self.cmd_pwm.output = [0.1,0.1,0.1,0.1]
        # self.cmd_pwm.usedrl = True
        # self.pwm_pub.publish(self.cmd_pwm)
        # state_before_reset = self.state_pos

        # pos_before_reset = self.pos
        # print(f"pos_before_reset:{pos_before_reset}")

        rospy.wait_for_service('/gazebo/set_model_state')
        # try:
        #     set_model_state_res = self.set_model_proxy(
        #         self.set_model_state_req)
        #     print('Reset model successfully: {} \n msg: {}'.format(set_model_state_res.success, set_model_state_res.status_message))
        # except rospy.ServiceException:
        #     print('Reset model successfully: {} \n msg: {}'.format(set_model_state_res.success,
        #                                                            set_model_state_res.status_message))
        # # self.last_state = self.get_state()
        
        # pos_after_reset = self.pos
        while True:
            # pos_before_reset = pos_after_reset
            pos_before_reset = self.pos
            try:
                set_model_state_res = self.set_model_proxy(
                    self.set_model_state_req)
                print('Reset model successfully: {} \n msg: {}'.format(set_model_state_res.success, set_model_state_res.status_message))
            except rospy.ServiceException:
                print('Reset model successfully: {} \n msg: {}'.format(set_model_state_res.success,
                                                                        set_model_state_res.status_message))
            pos_after_reset = self.pos

            if np.linalg.norm(pos_after_reset - pos_before_reset) < 1e-6:
                break

        print(f"pos_after_reset:{pos_after_reset}")
        pos = self.state_pos
        ang = self.state_ang
        self.dist2target_pos = pos[0:6] - self.target_pos
        self.new_state = np.concatenate(
            [self.state_pos[0:3], self.dist2target_pos[0:3]])
        print(f'after reset: {self.new_state}')
        norm_new_state = (self.new_state - self.observation_space.low)/(self.observation_space.high-self.observation_space.low)
        return norm_new_state

    # update continue_timestamp when a new msg form PC is received
    def callback_exp_continue(self, data) -> None:

        # we only concern whether Nano has lost the connection to PC
        # Hence, we just update the timestamp, and pay no attention to the data

        self.continue_timestamp = time.time()

    def update_state(self, data_pose):

        # update time stamp
        # TODO: should we use timestamp in the msg ?
        # TODO: seperate state timestamp and state sub connected timestamp? because they have different meaning
        # timestamp will must update, but state may be not (sub source not update or data disturbance)
        self.last_state_timestamp = self.state_timestamp
        self.state_timestamp = time.time()

        # get quat and pos from pose
        quat_raw = np.array([data_pose.orientation.x,
                             data_pose.orientation.y,
                             data_pose.orientation.z,
                             data_pose.orientation.w],
                            dtype=float)
        pos_raw = np.array([data_pose.position.x,
                            data_pose.position.y,
                            data_pose.position.z],
                           dtype=float)

        # list to ndarray?
        # quat_raw, pos_raw -> quat & pos
        # quat_raw = np.array(data_pose.orientation, dtype=float)
        # pos_raw = np.array(data_pose.position, dtype=float)
        theta_raw = Rotation.from_quat(quat_raw).as_euler('xyz')
        pos_general_raw = np.concatenate((pos_raw, theta_raw), axis=0)
        pos_general=pos_general_raw
        # self.filter_state.add(pos_general_raw)
        # pos_general = self.filter_state.get_state()
        quat = Rotation.from_euler('xyz', pos_general[3:6]).as_quat()
        pos = pos_general[0:3]

        # check data
        # if sub source does not update, system also does not update
        # if self.state_timestamp - state_timestamp < 1e-3:    # < 1ms
        # may be beacuse the gazebo will pub same msg multiple times
        # there are some msg have the same date but with different timestamp
        # if np.linalg.norm(pos - self.state_pos[0:3]) < 1e-6:
        if np.linalg.norm(pos - self.state_pos[0:3]) < 1e-6:
            # print('state not update,from %f to %f' % (self.last_state_timestamp, self.state_timestamp))
            pass  # does not update the states
        else:
            # update quat pos and last_quat last_pos
            self.last_quat = self.quat
            self.last_pos = self.pos
            self.quat = quat
            self.pos = pos

            # calculate vel,Theta
            # RPY is extrinsic XYZ euler angle, equal to instrinsic ZYX euler angle
            Theta = Rotation.from_quat(quat).as_euler('ZYX')

            vel = (self.pos - self.last_pos) * \
                self.mocap_freq  # equal to divide it with dt
            rot_last_q = Rotation.from_quat(self.last_quat)
            rot_q = Rotation.from_quat(self.quat)
            rot_delta = rot_last_q.inv() * rot_q
            omega_b = rot_delta.as_rotvec() * self.mocap_freq

            # update state
            self.last_state_ang = self.state_ang
            self.last_state_pos = self.state_pos
            # flip euler angles from zyx to xyz
            self.state_ang = np.concatenate(
                (np.flipud(Theta), omega_b), axis=0)
            self.state_pos = np.concatenate((pos, vel), axis=0)

    # input: data from vrpn
    # output: none
    # update: state, last_state (multiple variables)
    def sub_vrpn(self, data):

        # save raw data (do we need this?)
        self.local_vrpn = data

        # edit data
        motion_data = self.local_vrpn
        # from mm to m and add offset
        motion_data.pose.position.x = motion_data.pose.position.x / \
            1000 - self.pos_offset[0]
        motion_data.pose.position.y = motion_data.pose.position.y / \
            1000 - self.pos_offset[1]
        motion_data.pose.position.z = motion_data.pose.position.z / \
            1000 - self.pos_offset[2]

        # update state
        self.update_state(motion_data.pose)

        # print("position: ", self.state_pos)
        # print("orientation: ", self.state_ang)

    # input: data from gazebo
    # output: none
    # update: state, last_state (multiple variables)
    def sub_gazebo(self, data):

        # save raw data (do we need this?)
        self.local_gazebo = data
        # update state
        self.update_state(self.local_gazebo.pose[2])

    def sub_state(self, data):
        self.state = data


@hydra.main(config_path=to_absolute_path('learn/conf'), config_name='mpc.yaml')
def main_loop(cfg):
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    # print parameter
    print(OmegaConf.to_yaml(cfg))
    # print(to_absolute_path('learn/data'))

    # check params
    if cfg.experiment.quick_save_enable:
        assert cfg.experiment.seeds == 1 and cfg.experiment.repeat == 1, 'please disable quick save if seeds != 1'

    # create env
    env = Gazebo_env(cfg, None)

    # only reset in simulation
    if cfg.experiment.is_sim:
        env.reset()
    else:

        # Do nothing in code (manually reset quadrotor)
        pass

        # # for sim2real
        # print("sim2real reset")
        # env.reset()

    # wait for connection (record at least 2 states to calculate velocity)
    # init as zero
    # TODO: replace with is_connected()? (private -> public)
    time.sleep(0.1)
    while (env.last_state_timestamp < 1e-5 or env.continue_timestamp < 1e-5) \
            and not rospy.is_shutdown():
        print('waiting connection')
        env.rate.sleep()

    rospy.loginfo('connection confirmed')

    # nums of experiments (train in each episode)
    # num_exp * num_traj * rollout return obj (only exploration data)
    list_data_total = []
    for seed_num in range(cfg.experiment.seeds):
        # log.info(f"Random Seed: %d", s)   // priority level low, will not be shown on scerrn
        print(f"Random Seed: {seed_num + 1}")
        total_costs = []
        data_rand = []
        total_steps = []
        list_data_exp = []  # num_traj * rollout return obj

        X = np.empty((0, cfg.models.model.params.training.dx), dtype=float)
        dX = np.empty((0, cfg.models.model.params.training.dx), dtype=float)
        U = np.empty((0, cfg.models.model.params.training.du *
                     (cfg.models.model.params.history + 1)), dtype=float)

        if not cfg.experiment.use_previous_rand_data:
            # nums of random controller rollouts（fly trajectories) in one exp  (in real env, should always be 1?)
            for r in range(cfg.experiment.random):

                # break condition
                if rospy.is_shutdown():
                    break

                # log.info(f"random rollout: {r}")
                print('random rollout: ', r + 1)

                # use symm model or reflect data into one quadrant
                # get datas from rollouts
                data_r = env.rollout2(RandomController(
                    cfg), cfg.experiment, history=cfg.experiment.history)
                # data processing for one rollout
                # return from rollout2: states, actions, rews, flag0, LengthofTime, ...
                rews = data_r[2]
                # flag0, large lean angle/angular vel detected
                sim_error = data_r[3]
                LengthofTime = data_r[4]
                # deal with large lean angle and large angular velocity
                if sim_error:
                    # NOTE: do we really drop data if large Euler number is detected?
                    print("Repeating strange simulation")
                    # continue

                # collect data
                total_costs.append(np.sum(rews))  # for minimization
                log.info(f" - Cost {np.sum(rews) / cfg.experiment.r_len}")
                data_rand.append(data_r)
                total_steps.append(LengthofTime[1])
                list_data_exp.append(data_r)

                # quick saving
                # TODO: packaging theis into a data saving func
                # Note: expression forms of num_roll for random rollout and MPC rollout are different
                if cfg.experiment.quick_save_enable:
                    print('quick save random traj: {}'.format(r + 1))
                    list_data_total_quick = [list_data_exp]
                    trial_log_quick = dict(
                        raw_data=list_data_total_quick,
                        seed=cfg.experiment.seed,
                        seeds=cfg.experiment.seeds,  # must be 1
                        # num_roll=cfg.experiment.num_roll,
                        num_roll=r + 1,
                        # random=cfg.experiment.random,
                        random=r + 1,
                        repeat=cfg.experiment.repeat,  # must be 1
                        sym=cfg.models.model.params.sym,
                        freq=cfg.robot.load.freq,
                        use_previous_rand_data=cfg.experiment.use_previous_rand_data,
                        rand_data=cfg.experiment.rand_data_list
                    )
                    # save('data_total.mat', trial_log)
                    # with open("data_total_quick.txt", 'wb') as f:

            # data processing for random rollouts
            # # TODO: why do we need data_r to derive X,dX and U ? only data_rand is required
            # X, dX, U = to_XUdX(data_r)  # create X,dX,U ?
            # X, dX, U = combine_data(data_rand[:-1], (X, dX, U), 0)  # combind other data with the data of the last rollout

            # update X,dX,U with random data
            X, dX, U = combine_data(data_rand, (X, dX, U), 0)

            msg = "Random Rollouts completed of "
            msg += f"Mean Cumulative reward {np.mean(total_costs)}, "
            msg += f"Mean length {LengthofTime[1]}"
            print(msg)
            trial_log = dict(
                # cfg=cfg,
                # Note: only the X,dX and U of the last exp will be saved !!!
                X=X,
                dX=dX,
                U=U,
            )
            save('initial_data.mat', trial_log)
            print("save data")

        else:
            '''
            # load previous random data
            # for rand_data_pkg in cfg.experiment.rand_data_list:
            #
            #     # rand_data_list: file, exp, rand_rollout
            #     with open(rand_data_pkg[0], 'rb') as f:
            #         rand_data_dict = pickle.load(f)
            #
            #     rand_data = rand_data_dict['raw_data']
            #     rand_data_r = rand_data[rand_data_pkg[1]][rand_data_pkg[2]]
            #     print('use previous rand data:{}'.format(rand_data_pkg))
            #     print(f"Data length {rand_data_r[4][1]}")
            #
            #     data_rand.append(rand_data_r)
            #     # X, dX, U = to_XUdX(rand_data_r)  # create X,dX,U ?
            #     # # combind other data with the data of the last rollout
            #     # X, dX, U = combine_data(data_rand[:-1], (X, dX, U), 0)
            '''

            load_data = sio.loadmat(to_absolute_path(
                "learn/data/initial_data.mat"))
            X, dX, U = load_data["X"], load_data['dX'], load_data["U"]
            print(f"X:{X.shape}, dX:{dX.shape}, U:{U.shape}")

        if False:
            # train model
            model, train_log = train_model(X, U, dX, cfg.models.model)
            torch.save(model.state_dict(), 'model.pth')

            # save first model and data
            torch.save(model.state_dict(),
                       'model_first_{}.pth'.format(seed_num))
            data_first = dict(
                X=X,
                U=U,
                dX=dX)
            sio.savemat('data_first_{}.mat'.format(seed_num), data_first)

            model_cfg = cfg.models.model
            model = GeneralNN(model_cfg.params)
            model.load_state_dict(torch.load('model.pth'))
            model.half()
            model.cuda()
            model.eval()
            model.preprocess_cuda((X.squeeze(), U, dX.squeeze()))

            Xe = X
            Ue = U
            dXe = dX
            LengthofX = X.shape[0]
            LengthofXe = Xe.shape[0]

            # nums of NN-MPC controller rollouts
            for i in range(cfg.experiment.num_roll - cfg.experiment.random):
                controller = MPController(env, model, cfg)

                data_rs = []
                for r in range(cfg.experiment.repeat):
                    # log.info(f"MPC episode {i} rollout {r}")
                    print(f"MPC episode {i + 1} rollout {r + 1}")

                    data_r = env.rollout2(
                        controller, cfg.experiment, cfg.experiment.history, flag_e=True, num_roll=i)
                    rews = data_r[2]
                    # flag0, large lean angle/angular vel detected
                    sim_error = data_r[3]
                    LengthofTime = data_r[4]

                    if sim_error:
                        print("Repeating strange simulation")
                        # continue
                    # cum_costs.append(np.sum(rews) / len(rews))  # for minimization
                    total_costs.append(np.sum(rews))  # for minimization
                    log.info(f" - Cost {np.sum(rews) / cfg.experiment.r_len}")
                    # r += 1
                    data_rs.append(data_r)
                    total_steps.append(LengthofTime[1])
                    list_data_exp.append(data_r)

                    # quick saving
                    # TODO: packaging theis into a data saving func
                    # Note: expression forms of num_roll for random rollout and MPC rollout are different
                    if cfg.experiment.quick_save_enable:
                        print('quick save MPC traj: {}'.format(i + 1))
                        list_data_total_quick = [list_data_exp]
                        trial_log_quick = dict(
                            raw_data=list_data_total_quick,
                            seed=cfg.experiment.seed,
                            seeds=cfg.experiment.seeds,  # must be 1
                            num_roll=i + 1 + cfg.experiment.random,
                            # or len(previous_rand_data) ?
                            random=cfg.experiment.random,
                            repeat=cfg.experiment.repeat,  # must be 1
                            sym=cfg.models.model.params.sym,
                            freq=cfg.robot.load.freq,
                            use_previous_rand_data=cfg.experiment.use_previous_rand_data,
                            rand_data=cfg.experiment.rand_data_list
                        )
                        # save('data_total.mat', trial_log)
                        with open("data_total_quick.txt", 'wb') as f:
                            pickle.dump(trial_log_quick, f)

                # update X, dX, U with MPC rollout data
                X, dX, U = combine_data(data_rs, (X, dX, U), 0)
                ##################################################
                datae = []
                # for data in data_rs:
                #     # print('data[0]: ', data[0])
                #     print('data[-1]: ', data[-1])
                #     print(' data[-2]', data[-2])
                #     if data[-2]:
                #         datae.append([data[0][:data[-1]],data[1][:data[-1]], None, None, data[4]])
                #     else:
                #         datae.append([data[0], data[1], None, None, data[4]])
                #
                # Xe, dXe, Ue = combine_data(datae, (Xe, dXe, Ue), 0)
                ##################################################
                Xm = X[LengthofX:, :]
                Um = U[LengthofX:, :]
                dXm = dX[LengthofX:, :]
                newXe = []
                newUe = []
                newdXe = []
                X_max = np.array([0.5, 0.5, 0.5, 4, 3, 0.5])
                X_min = np.array([-0.5, -0.5, -0.5, -4, -3, -0.5])

                Xm = (Xm - X_min) / (X_max - X_min)
                Um = (Um - 1075) / (1950 - 1075)
                Xe_ = (Xe - X_min) / (X_max - X_min)
                Ue_ = (Ue - 1075) / (1950 - 1075)

                for point_a in range(Xm.shape[0]):
                    dist = 0
                    dist_min = 100
                    dist_avg = 0
                    xp = Xm[point_a, :]
                    up = Um[point_a, :]
                    for point_b in range(Xe.shape[0]):
                        temp = np.linalg.norm(np.concatenate(
                            (xp, up)) - np.concatenate((Xe_[point_b, :], Ue_[point_b, :])))
                        # temp = np.linalg.norm(np.concatenate((xp, up)) - np.concatenate((Xe[point_b,:], Ue_[point_b, :])))

                        dist_min = temp if (dist_min > temp) else dist_min

                        dist += temp

                    dist_avg = dist / (Xe.shape[0])
                    # print(f",dist: {dist_avg}")
                    # print(f"dist_min: {dist_min}")

                    # if dist_avg > 1.5:
                    #     newXe.append(X[LengthofX + point_a, :])
                    #     newUe.append(U[LengthofX + point_a, :])
                    #     newdXe.append(dX[LengthofX + point_a, :])
                    if dist_min > 0.25:
                        newXe.append(X[LengthofX + point_a, :])
                        newUe.append(U[LengthofX + point_a, :])
                        newdXe.append(dX[LengthofX + point_a, :])

                if (len(newXe) != 0):
                    Xe = np.concatenate((Xe, newXe))
                    Ue = np.concatenate((Ue, newUe))
                    dXe = np.concatenate((dXe, newdXe))

                LengthofX = X.shape[0]

                ######################################################

                # save data
                print(f"save data X: {X.shape}, U:{U.shape}")
                print(f"save data X: {Xe.shape}, U:{Ue.shape}")
                # trial_log = dict(
                #     # cfg=cfg,
                #     # Note: only the X,dX and U of the last exp will be saved !!!
                #     X = X,
                #     dX = dX,
                #     U = U,
                #     Xe = Xe,
                #     dXe = dXe,
                #     Ue = Ue,
                # )
                # save('data_total.mat', trial_log)

                #######################################################

                msg = "Rollouts completed of "
                # / cfg.experiment.r_len
                msg += f"Cumulative reward {total_costs[-1]}, "
                msg += f"length {len(data_r[0])}"
                log.info(msg)

                # model, train_log = train_model(X, U, dX, cfg.models.model)
                if (Xe.shape[0] != LengthofXe):
                    model, train_log = train_model(
                        Xe, Ue, dXe, cfg.models.model)

                    torch.save(model.state_dict(), 'model.pth')

                    model_cfg = cfg.models.model
                    model = GeneralNN(model_cfg.params)
                    model.load_state_dict(torch.load('model.pth'))
                    model.half()
                    model.cuda()
                    model.eval()
                    model.preprocess_cuda((Xe.squeeze(), Ue, dXe.squeeze()))

                # save final model and data
                if i == (cfg.experiment.num_roll - cfg.experiment.random - 1):
                    torch.save(model.state_dict(),
                               'model_final_{}.pth'.format(seed_num))
                    data_final = dict(
                        X=Xe,
                        U=Ue,
                        dX=dXe)
                    sio.savemat('data_final_{}.mat'.format(
                        seed_num), data_final)

                LengthofXe = Xe.shape[0]
            list_data_total.append(list_data_exp)

            # save data
            trial_log = dict(
                # cfg=cfg,
                # Note: only the X,dX and U of the last exp will be saved !!!
                X=X,
                dX=dX,
                U=U,
                raw_data=list_data_total,
                seed=cfg.experiment.seed,
                seeds=cfg.experiment.seeds,
                num_roll=cfg.experiment.num_roll,
                random=cfg.experiment.random,
                repeat=cfg.experiment.repeat,
                sym=cfg.models.model.params.sym,
                freq=cfg.robot.load.freq,
                use_previous_rand_data=cfg.experiment.use_previous_rand_data,
                rand_data=cfg.experiment.rand_data_list
            )
            save(f'data_total_{seed_num}.mat', trial_log)
            save(f'data_tota.mat', trial_log)
            with open("data_total.txt", 'wb') as f:
                pickle.dump(trial_log, f)

        else:
            model_cfg = cfg.models.model
            model = GeneralNN(model_cfg.params)
            if cfg.models.model.params.sym:
                print("use sym")
                data_load = sio.loadmat(to_absolute_path(
                    "learn/data/data_final_sym.mat"))
                X = data_load["X"]
                U = data_load["U"]
                dX = data_load["dX"]
                model.load_state_dict(torch.load(
                    to_absolute_path('learn/data/model_sym.pth')))
            else:
                data_load = sio.loadmat(
                    to_absolute_path("learn/data/data_final.mat"))
                X = data_load["X"]
                U = data_load["U"]
                dX = data_load["dX"]
                model.load_state_dict(torch.load(
                    to_absolute_path('learn/data/model.pth')))

            model.half()
            model.cuda()
            model.eval()
            model.preprocess_cuda((X.squeeze(), U, dX.squeeze()))

            controller = MPController(env, model, cfg)
            for nnn in range(3):
                print(
                    f"nn is {nnn}\n --------------------------------------------")
                data_rs = []
                data_r = env.rollout2(
                    controller, cfg.experiment, cfg.experiment.history, flag_e=True, num_roll=0, test=True)

                trial_log = dict(
                    # cfg=cfg,
                    # Note: only the X,dX and U of the last exp will be saved !!!
                    raw_data=data_r,
                    sym=cfg.models.model.params.sym,
                    freq=cfg.robot.load.freq,
                )
                save(f'data_total_{nnn}.mat', trial_log)
            break


def to_XUdX(data):
    states = np.stack(data[0])
    X = states[:-1, :]
    dX = states[1:, :] - states[:-1, :]
    U = np.stack(data[1])[:-1, :]
    return X, dX, U


def combine_history(X, U, history=0):
    state_len = np.shape(X)[1]
    action_len = np.shape(U)[1]
    X_t = X
    U_t = U
    for i in range(history):
        # X_t.append(np.concatenate((np.zeros(i+1,np.shape(X)[1]), X[:-1,:])))
        # U_t.append(np.concatenate((np.zeros(i+1,np.shape(U)[1]), U[:-1,:])))
        # print(np.shape(np.concatenate((np.zeros((i+1,state_len) ,dtype=float), X_t[:-1-i,:]))))
        # print(np.shape(X))
        # print(np.shape(np.concatenate((np.zeros((i+1,state_len) ,dtype=float), X_t[:-1-i,:]))))
        X = np.concatenate((X, np.concatenate(
            (np.zeros((i + 1, state_len), dtype=float), X_t[:-1 - i, :]))), axis=1)
        U = np.concatenate((U, np.concatenate(
            (1075 * np.ones((i + 1, action_len), dtype=float), U_t[:-1 - i, :]))), axis=1)

    X = X[history:, :]
    U = U[history:, :]

    return X, U


def combine_data(data_rs, full_data, history):
    X = full_data[0]
    U = full_data[2]
    dX = full_data[1]

    for data in data_rs:

        # skip traj with not enough data points (2 points to calculat dX)
        steps = data[4][1]
        if steps < 2:
            continue
        X_new, dX_new, U_new = to_XUdX(data)
        X_new, U_new = combine_history(X_new, U_new, history)
        X = np.concatenate((X, X_new), axis=0)
        U = np.concatenate((U, U_new), axis=0)
        dX = np.concatenate((dX, dX_new), axis=0)

    return X, dX, U


if __name__ == '__main__':
    main_loop()
