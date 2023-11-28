# Install PX4

## Directly copy from source code

1. git clone整个仓库后，在ros1的工作空间中运行"catkin clean"后，在运行"catkin build".
    - missing "geographic_msgs", run `sudo apt-get install ros-noetic-geographic-msgs`.
    - missing "GeographicLib_LIBRARIES", 按照 "mavros/mavros/scripts/install_geographiclib_datasets.sh"手动安装
    - intsall `sudo apt-get install libgeographic-dev ros-your-geographic-msgs`

2. 运行“bash ./PX4-Autopilot/Tools/setup/ubuntu.sh”，根据自己的情况修改路径
3. 删除PX4中的build目录后，运行“make ...”, 可能存在swap内存不足，需要临时分配swap内存
参考："https://blog.csdn.net/qq_42585108/article/details/106780885"， 记得关闭临时内存
4. 添加路径

    ```bash
    source ~/PX4_Firmware/Tools/setup_gazebo.bash ~/PX4_Firmware/ ~/PX4_Firmware/build/px4_sitl_default
    export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4_Firmware
    export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4_Firmware/Tools/sitl_gazebo
    ```

根据自己的情况修改路径。

## Disable safe protection

- QGroundcontrol 设置，将“Failsafe mode activated” unactivate.
  - set *Object Detection/Collison prevention* disable. Set *RC Loss Failsafe Trigger/Failsafe action* to **Hold mode**. Disable rest except last two.
  - 在parameters, 找到failsafe,*COM_OBL_ACT* and *COM_OBL_RC_ACT* ,改为disable
  - 在parameters, 找到*FD_FAIL_P(/R)*,改为 *180*

## Problem may occur

- 出现 `Resource not found: mavros`，没有安装好 mavros.
