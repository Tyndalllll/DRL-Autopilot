# 插件和环境设置

## 力插件

通过施加力到无人机中心来模仿风。

```cpp
ros::SubscribeOptions so0 = ros::SubscribeOptions::create<std_msgs::Float32MultiArray>(
                "/wind_force",
                100,
                boost::bind(&TestforcePlugin::OnRosMsg0, this, _1),
                ros::VoidPtr(), &this->rosQueue);
```

插件以100hz接受`/wind_force`topic上的消息，将需要施加的力应用在无人机上。

## Installations

1. delete all files in build folder.
2. In build folder, run `cmake ..` and `cmake --build .`.
3. open environment file, `.bashrc`, add

```
export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:~/path_to_plugins_files/plugins/build
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/path_to_plugins_files/plugins/build
```

> replace *path_to_plugins_files* with the true path.

4. In launch files `posix_sitl_customized.launch`, set **world** value as 

```xml
<arg name="world" default="/home/path_to_world_folder/3box.world"/>
```
## 流程
1.环境初始化
    无人机状态订阅（真值订阅频率250hz,姿态命令发布频率 8或9hz）
2.SAC智能体初始化
3.开始训练
    1.env.ramp_up
        检查连接
        请求起飞
    2.5000 episodes
        1.env.reset_safe
            每个episode之前的初始化，第一次飞行时，在(0 0 2)起飞，随后每次在范围里随机取一个点，返回一个初始状态
        2.开始训练　5000个step:(每个step大概0.015s)
            1.if replay_buffer.size < 512:  
                action = env.action_space.sample()
            else:
                 action = agent.select_action(state,deterministic=False, with_logprob=False)
            2.env.step
                限制action范围并发布0.1s，归一化状态并计算奖励，得到下一时刻的状态，并存进buffer
            3.利用buffer 里的数据训练(一次训练大概0.012s)
            4.if (done or (t == max_steps - 1)or env.region_out):保存数据，该episode结束
        3.保存agent模型并展示该episode花费的step与时间
    3.画出奖励曲线并结束



