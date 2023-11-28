import logging
from scipy.spatial.transform import Rotation
from scipy import io as sio
import numpy as np
import time
import gym.spaces as spaces
import gym
from geometry_msgs.msg import Pose, Twist, TwistStamped
from gazebo_msgs.msg import ModelStates
from mavros_msgs.msg import ActuatorOutputsDRL,AttitudeTarget
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
sys.path.append(os.getcwd())


log = logging.getLogger(__name__)

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
        self.cmd_att = AttitudeTarget()
        self.local_gazebo = ModelStates()
        self.local_vrpn = PoseStamped()
        self.sleep_rampup = 0.25  # 250ms
        # TODO: diff setting for real exp and sim
        self.rampup_rounds = 20  # n * 1/50Hz = 20n ms
        self.hz = 10
        self.mocap_freq = 250
        self.set_model_state_req = SetModelState._request_class()
        self.original_last_state = None
        self.act_dim = 4
        self.state_dim = 18
        self.obs_space = []
        self.act_space = []
        self.pos_x_low = 3
        self.pos_x_high = 7
        self.pos_y_low = -2
        self.pos_y_high = 2
        self.pos_z_low = 0.5
        self.pos_z_high = 2.5
        self.action_space = spaces.Box(low=np.array(
            [-0.5, -0.5, -0.5,0]), high=np.array([0.5, 0.5, 0.5,1.4]), dtype=np.float32)
        # observations are [x, y, z,vx,vy,vz, dist2targ_x, dist2targ_y, dist2targ_z]
        self.observation_space = spaces.Box(low=np.array(
            [-2, -2, -1,-5,-5,-5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-0.5,-0.5,-0.5]), high=np.array([2, 2, 1,5,5,5,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5]), dtype=np.float32)
        #创建连续空间
        
        self.act_space.append(self.action_space)
        self.obs_space.append(self.observation_space)
        # self.actions = [0 for _ in range(self.act_dim)]
        self.current_velocity = [0 for _ in range(self.act_dim)]
        self.success_flag = False
        self.out_flag = False
        self.success_count = 0
        self.region_out = False

        # self.history_len = cfg.models.model.params.history
        # self.set_model_state_res = SetModelState._response_class()
        self.new_state = np.zeros((self.state_dim,), dtype=float)
        self.next_state = np.zeros((self.state_dim,), dtype=float)
        self.last_state_ang = np.zeros((12,), dtype=float)
        self.last_state_pos = np.zeros((6,), dtype=float)
        self.last_action = np.zeros((self.act_dim,), dtype=float)
        self.state_ang = np.zeros((12,), dtype=float)
        self.state_pos = np.zeros((6,), dtype=float)
        self.target_ang = np.zeros((12,), dtype=float)
        self.target_pos = np.zeros((6,), dtype=float)
        self.dist2target_pos = np.zeros((3,), dtype=float)
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
        self.att = np.zeros((self.act_dim,), dtype=float)
        self.angle = np.zeros((3,), dtype=float)
        self.now_angle = np.zeros((3,), dtype=float)

        print('Controller Frequency: {} Hz'.format(self.hz))
        # self.cfg = cfg
        # define metric
        self.metric = metric
        print("start to sub")
        """ 
        add_thread = threading.Thread(target = self.thread_job)
        add_thread.start()
        print("new threading")
         """
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
        self.att_pub = rospy.Publisher(
            'mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)
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
        TODO: Do we need to record data when doing ramp up
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
        rate_5 = rospy.Rate(10)
        for i in range(10):   
            if(rospy.is_shutdown()):
                break
            self.cmd_velocity.twist.linear.x = 0.
            self.cmd_velocity.twist.linear.y = 0.
            self.cmd_velocity.twist.linear.z = 0.
            self.vel_pub.publish(self.cmd_velocity)
            rate_5.sleep()

        while not rospy.is_shutdown() and (self.state.armed != True or self.state.mode != "OFFBOARD"):
                # and not self.state.armed:
            # print(self.state.armed,self.state.mode)
            self.set_mode_client.call(offb_set_mode)
            if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")
                print(f"now mode is {self.state.mode}")
            if (self.state.mode == "OFFBOARD" and self.state.armed != True):
                response = self.arming_client(True)
                print(f"arm :{response.success}")
            # else:
            #     print('please arming by Radio Control')
            #     num = eval(input("input anything to continue"))

            # self.set_mode_client(0, "STABILIZED")
            # print(1)
            rate_5.sleep()
        rospy.loginfo("Arming success,mode:{}".format(self.state.mode))
        #self.cmd_pwm.usedrl = False
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
#TODO:action用一下
        """ dist = np.linalg.norm(np.array(self.target_pos[0:3]) - np.array(self.new_state[0:3]))
        delt_dist = np.linalg.norm((np.array(self.target_pos[0:3]) - np.array(self.new_state[0:3])))- np.linalg.norm((np.array(self.target_pos[0:3]) - np.array(self.last_state_pos[0:3])))
        # cos_similarity = np.dot(self.dist2target_pos[0:3] / np.linalg.norm(self.dist2target_pos[0:3]), self.new_state[3:6]/ np.linalg.norm(self.new_state[3:6]))
        cos_similarity = np.dot(self.dist2target_pos[0:3] / np.linalg.norm(self.dist2target_pos[0:3]), action/ np.linalg.norm(action))
        # velocity_stable = np.linalg.norm(np.array(self.new_state[3:6]))
        velocity_stable = np.linalg.norm(action)
        accelerate_stable = np.linalg.norm(np.array(self.last_action) - np.array(action)) """
        dist = np.linalg.norm(np.array(self.next_dist2target_pos))
        delt_dist = np.linalg.norm(np.array(self.next_dist2target_pos))- np.linalg.norm(np.array(self.dist2target_pos)) # type: ignore
        rew_dist = dist + 0.3*delt_dist
        rew_step = 0.02
        rew_angle = np.linalg.norm(np.array(self.now_angle))
        #print("angle:",self.now_angle)
        # cos_similarity = np.dot(self.dist2target_pos[0:3] / np.linalg.norm(self.dist2target_pos[0:3]), self.new_state[3:6]/ np.linalg.norm(self.new_state[3:6]))
        #cos_similarity = np.dot(self.dist2target_pos[0:3] / np.linalg.norm(self.dist2target_pos[0:3]), action/ np.linalg.norm(action))
        # velocity_stable = np.linalg.norm(np.array(self.new_state[3:6]))
        #velocity_stable = np.linalg.norm(action)
        #accelerate_stable = np.linalg.norm(np.array(self.last_action) - np.array(action))
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
            #rew = - 1*dist - 3*delt_dist # type: ignore
            rew = -rew_dist - t*rew_step - rew_angle  # type: ignore
            print("rew_sorts:",rew_dist,t*rew_step,rew_angle)# type: ignore
        

        # print(f"dist:{3*dist}, cos_similarity: {0.8*(cos_similarity+1)}, accelerate_stable:{2 * accelerate_stable}, velocity_stable:{1 *velocity_stable}")
        """ if rew>5:
            rew = 5
        if rew< -5:
            rew = -5 """

        
        

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

        for i in abs(self.now_angle):
            if i > 0.4:
                print("drone will clash,attitude is:",self.now_angle)
                rew =rew - 300
                break
        if dist < 0.3: # type: ignore
            rew = rew + 6
            print("dist less 0.3")
            if dist <0.1: # type: ignore
                rew = rew + 14
                print("dist less 0.1")
                self.success_count+=1
            if self.success_count>=10:
                self.success_flag = True
                rew +=300
        """ if self.new_state[0]<=self.pos_x_low or self.new_state[0]>=self.pos_x_high or self.new_state[1]<=self.pos_y_low or self.new_state[1]>=self.pos_y_high or self.new_state[2]<=self.pos_z_low or self.new_state[2]>=self.pos_z_high:
            print(f"When out, pos is {self.new_state[0:3]}") """
        if self.state_pos[0]<=self.pos_x_low or self.state_pos[0]>=self.pos_x_high or self.state_pos[1]<=self.pos_y_low or self.state_pos[1]>=self.pos_y_high or self.state_pos[2]<=self.pos_z_low or self.state_pos[2]>=self.pos_z_high:
            print(f"When out, pos is {self.state_pos[0:3]}")
            rew = rew - 300
            self.region_out = True
        # print(rew)
        rew = rew/6
        #if rew <-50: rew = -1
            #self.out_flag = True
        #if rew > 50:rew = 1

        return rew
    
#step有一点小问题，但不影响，之后用的时候再改 
    def step(self, action, t):
        # print(action)
        # self.new_state = np.concatenate([self.state_ang,self.state_pos,self.dist2target_pos[0:3]])
        pos1 = self.state_pos
        ang1 = self.state_ang
        self.dist2target_pos = pos1[0:3] - self.target_pos[0:3]
        #self.new_state = np.concatenate([pos1[0:3], self.dist2target_pos[0:3],pos1[3:6]])
        #print("before change velocity,state:",pos1)
        # print(f"dist2target_pos: {self.dist2target_pos}, target_pos:{self.target_pos}, pos:{pos}")
        # self.norm_new_state = (self.new_state - self.observation_space.low)/(self.observation_space.high-self.observation_space.low)
        # print(self.norm_new_state)
        # real_action = action*500 + np.array([1500,1500,1500,1500])

        # velocity vectort in range(max_steps)
        #self.current_velocity += action * 0.05
        #self.current_velocity = action * 0.1
        #real_velocity = self.current_velocity
        #real_velocity = real_velocity.clip(-1, 1)
        #print("current velocity:",real_velocity)
        #print("current velocity:",self.local_gazebo.twist[2].linear)

        # add velocity control by lilong
        #self.cmd_velocity.twist.linear.x = real_velocity[0]
        #self.cmd_velocity.twist.linear.y = real_velocity[1]
        #self.cmd_velocity.twist.linear.z = real_velocity[2]
        # self.cmd_velocity.twist.linear.x = 0.
        # self.cmd_velocity.twist.linear.y = 0.
        # self.cmd_velocity.twist.linear.z = 0.2
        # self.cmd_velocity.header.stamp = rospy.time()
        # print("x:{0},y:{1},z:{2}".format(self.cmd_velocity.twist.linear.x,
        #       self.cmd_velocity.twist.linear.y, self.cmd_velocity.twist.linear.z))
        #self.vel_pub.publish(self.cmd_velocity)
        self.att = action.clip(self.action_space.low,self.action_space.high)
        body_rate = self.att[0:3]
        thrust = self.att[3]
        self.cmd_att .body_rate.x = body_rate[0]
        self.cmd_att .body_rate.y = body_rate[1]
        self.cmd_att .body_rate.z = body_rate[2]
        self.cmd_att.thrust = thrust
        self.cmd_att.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        self.att_pub.publish(self.cmd_att)
        print("pub action:",self.att)
        # real_action = np.array([1800,1800,1800,1800])
        # self.cmd_pwm.output = list(self.pwm_normalize(real_action))
        # print(self.cmd_pwm.output)
        # self.pwm_pub.publish(self.cmd_pwm)
        self.rate.sleep()#等待0.1s，无人机进入了下一个状态
        next_pos = self.state_pos
        next_ang = self.state_ang
        self.now_angle = self.angle
        self.next_dist2target_pos = next_pos[0:3] - self.target_pos[0:3]
        self.next_state = np.concatenate(
            [self.next_dist2target_pos[0:3],next_pos[3:6],next_ang])
        #todo : 归一化距离
        self.norm_next_state = (self.next_state - self.observation_space.low)/(self.observation_space.high-self.observation_space.low)
        
        # print(self.state_ang)
        self.last_state_pos = pos1
        self.last_state_ang =ang1
        #self.last_state_pos = self.new_state[0:6]
        #self.last_state_ang = self.new_state[9:12]
        # self.dist2target_ang = self.state_ang - self.target_ang
        # print(self.dist2target_pos)
        rew = self.rewardCacul(t,action)
        # print(state)
        # print(f"time: {t}")


        self.last_action = action
        #print(f"error action: {real_velocity - next_pos[0:3]}")
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
    
    def norm_now_state(self):
        next_pos = self.state_pos
        next_ang = self.state_ang
        next_dist2target_pos = next_pos[0:3] - self.target_pos[0:3]
        next_state = np.concatenate(
            [next_dist2target_pos[0:3],next_pos[3:6],next_ang])
        #todo : 归一化距离
        norm_next_state = (next_state - self.observation_space.low)/(self.observation_space.high-self.observation_space.low)
        return norm_next_state
        
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
        #self.current_velocity = 0
        # pos and angle
        init_pose.pose.orientation.w = 1.0
        init_pose.pose.orientation.x = 0.
        init_pose.pose.orientation.y = 0.
        init_pose.pose.orientation.z = 0.

        if self.first_time == True:
            print("Start to fly!!!!!!!!!!!!!!!!11")
            #print(f"now mode is {self.state.mode}")

            init_pose.pose.position.x = 0.
            init_pose.pose.position.y = 0.
            init_pose.pose.position.z = 2.

            #self.pos_pub.publish(init_pose)
            
            for i in range(500):   
                if(rospy.is_shutdown()):
                    break

                self.pos_pub.publish(init_pose)
                time.sleep(0.01)

            time.sleep(3)
            print(f"pos is {self.pos}")#测试时飞不到(0,0,2)就end fly了
            self.first_time = False
            print("end first fly")
        
        init_pose.pose.position.x = np.random.uniform(self.pos_x_low+1,self.pos_x_high-1)
        init_pose.pose.position.y = np.random.uniform(self.pos_y_low+1,self.pos_y_high-1)
        init_pose.pose.position.z = np.random.uniform(self.pos_z_low+0.7,self.pos_z_high-0.8)

        # target pos
        self.target_ang = np.array([1, 0, 0,0,1,0,0,0,1, 0, 0, 0])
        self.target_pos = np.array([5, 0, 1.5, 0, 0, 0])
            
        pos_before_reset = np.array([init_pose.pose.position.x,-init_pose.pose.position.y,init_pose.pose.position.z])
        print(f"pos_before_reset:{pos_before_reset}")
        sleep = rospy.Rate(50)
        while True:
            pos_after_reset = self.pos
            #print(pos_after_reset)
            sleep.sleep()
            try:
                self.pos_pub.publish(init_pose)
                #print(f"pos_before_reset:{pos_before_reset}")
                #print(f"now pos is {pos_after_reset}")
                #print("delta pos:",np.linalg.norm(pos_after_reset - pos_before_reset))
            except rospy.ServiceException:
                pass
            
            if np.linalg.norm(pos_after_reset - pos_before_reset) < 0.15: # type: ignore
                break
        #print(f"pos_after_reset:{pos_after_reset}")
        pos = self.state_pos
        ang = self.state_ang
        self.dist2target_pos = pos[0:3] - self.target_pos[0:3]
        self.new_state = np.concatenate([self.dist2target_pos[0:3],self.state_pos[3:6],self.state_ang])
        print(f'after reset: {self.state_pos[0:3]}')
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

    def update_state(self, data_pose,data_twist):

        # update time stamp
        # TODO: should we use timestamp in the msg ?
        # TODO: seperate state timestamp and state sub connected timestamp? because they have different meaning
        # timestamp will must update, but state may be not (sub source not update or data disturbance)
        #self.last_state_timestamp = self.state_timestamp
        #self.state_timestamp = time.time()
#TODO:速度确实得用姿态除出来
        # get quat and pos from pose
        quat_raw = np.array([data_pose.orientation.x,
                             data_pose.orientation.y,
                             data_pose.orientation.z,
                             data_pose.orientation.w],
                            dtype=float)
        pos_raw = np.array([data_pose.position.x,
                            -data_pose.position.y,
                            data_pose.position.z],
                           dtype=float)
        linear_raw = np.array([data_twist.linear.x,
                            -data_twist.linear.y,
                            data_twist.linear.z],
                           dtype=float)
        angular_raw = np.array([data_twist.angular.x,
                            data_twist.angular.y,
                            data_twist.angular.z],
                           dtype=float)
        # list to ndarray?
        # quat_raw, pos_raw -> quat & pos
        # quat_raw = np.array(data_pose.orientation, dtype=float)
        # pos_raw = np.array(data_pose.position, dtype=float)
        theta_raw = Rotation.from_quat(quat_raw).as_euler('xyz')
        pos_general_raw = np.concatenate((pos_raw, theta_raw), axis=0)
        pos_general=pos_general_raw
        self.angle = theta_raw
        # self.filter_state.add(pos_general_raw)
        # pos_general = self.filter_state.get_state()
        quat = Rotation.from_euler('xyz', pos_general[3:6]).as_quat()
        pos = pos_general[0:3]
        #print(pos,self.state_pos[0:3])
        # check data
        # if sub source does not update, system also does not update
        # if self.state_timestamp - state_timestamp < 1e-3:    # < 1ms
        # may be beacuse the gazebo will pub same msg multiple times
        # there are some msg have the same date but with different timestamp
        # if np.linalg.norm(pos - self.state_pos[0:3]) < 1e-6:
        if np.linalg.norm(pos - self.state_pos[0:3]) < 1e-6: # type: ignore
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
            #Theta = Rotation.from_quat(quat).as_euler('ZYX')
            vel = linear_raw
            """   vel = (self.pos - self.last_pos) * \
                self.mocap_freq  # equal to divide it with dt """
            #rot_last_q = Rotation.from_quat(self.last_quat)
            #rot_q = Rotation.from_quat(self.quat)
            #rot_delta = rot_last_q.inv() * rot_q
            #omega_b = rot_delta.as_rotvec() * self.mocap_freq
            rot = Rotation.from_quat(quat).as_matrix()
            omega_b = angular_raw
            # update state
            self.last_state_ang = self.state_ang
            self.last_state_pos = self.state_pos
            # flip euler angles from zyx to xyz
            self.state_ang = np.concatenate((rot[0], rot[1], rot[2], omega_b), axis=0) # type: ignore
            #print("now the twist is:",self.state_ang)
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
        self.update_state(self.local_gazebo.pose[2],self.local_gazebo.twist[2])

    def sub_state(self, data):
        self.state = data
        
    def velocity_test(self):
        action = self.action_space.sample()
        self.current_velocity += action*0.05
        real_velocity = self.current_velocity
        real_velocity = real_velocity.clip(-1, 1)
        print("current pub velocity:",real_velocity)
        print("current velocity:",self.local_gazebo.twist[2].linear)
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
        self.rate.sleep()#等待一段时间，无人机进入了下一个状态
        next_pos = self.state_pos
        return next_pos
    
    def thread_job(self):
        print("spin")
        rospy.spin() 
    
