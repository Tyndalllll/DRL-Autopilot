#ifndef _TESTFORCE_PLUGIN_HH_
#define _TESTFORCE_PLUGIN_HH_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
// #include <gazebo/math/gzmath.hh>
// #include <ignition/math/Vector3.hh>
// #include <ignition/math/Quaternion.hh>
#include <ignition/math/Pose3.hh>
// #include <math.h>
#include<cmath>
#include<geometry_msgs/Pose.h>
#include <gazebo/physics/Base.hh>
#include <gazebo/physics/Link.hh>
#include <thread>
#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Float32MultiArray.h"
#include <gazebo/common/common.hh>
//#include "ros/Quaternion.h"
//#include "ros/Matrix3x3.h"
//#include "sensor_msgs/ChannelFloat32.h"

namespace gazebo
{
        /// \brief A plugin to control a Velodyne sensor.
        class TestforcePlugin : public WorldPlugin
        {
                /// \brief Constructor
                public: TestforcePlugin() {
                                printf("=============================\n");
                                printf("load force plugin success!!!!!!!!!\n");
                                printf("===========================\n");
                        }

                public: virtual void Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf)
                        {
                                this->world=_parent;
                                this->drone = this->world->ModelByName("if750a");
                                this->toprod=this->drone->GetLink("if750a::base_link");


                                // Initialize ros, if it has not already bee initialized.
                                if (!ros::isInitialized())
                                {
                                        int argc = 0;
                                        char **argv = NULL;
                                        ros::init(argc, argv, "gazebo_client",
                                                        ros::init_options::NoSigintHandler);
                                }

                                // Create our ROS node. This acts in a similar manner to
                                // the Gazebo node
                                this->rosNode.reset(new ros::NodeHandle("gazebo_client"));
                                this->prevtime=this->world->SimTime();//获取世界模拟时间，
                                                                      // Create a named topic, and subscribe to it.
                                                                      //++++++++++++++++++++++++++++++++++++++++++

                                ros::SubscribeOptions so0 =
                                        ros::SubscribeOptions::create<std_msgs::Float32MultiArray>(
                                                        "/wind_force",
                                                        100,
                                                        boost::bind(&TestforcePlugin::OnRosMsg0, this, _1),
                                                        ros::VoidPtr(), &this->rosQueue);
                                this->rosSub0 = this->rosNode->subscribe(so0);

                                // Spin up the queue helper thread.
                                //  this->rosQueueThread =
                                //    std::thread(std::bind(&TestforcePlugin::QueueThread, this));
                                //    +++++++++++++++++++++++++++++++++++++++

                                // Spin up the queue helper thread.
                                this->rosQueueThread =
                                        std::thread(std::bind(&TestforcePlugin::QueueThread, this));
                                //    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

                                std::cerr <<"testcycle\n";
                        }

                public: void ApplyForce0(const double & fx,const double & fy,const double & fz)
                        {
                                this->force = ignition::math::Vector3d(fx,fy,fz);
                                this->toprod->AddForceAtRelativePosition(this->force, this->pos);
                        }


                        /// \brief Handle an incoming message from ROS

                        // car和无人机整合
                public: void OnRosMsg0(const std_msgs::Float32MultiArrayConstPtr &_msg)
                        {

                                if (this->world->SimTime() - this->prevtime >= this->timeinterval||this->world->SimTime() - this->prevtime<0)
                                {
                                        std::cerr <<"timedifference:"<<this->world->SimTime() - this->prevtime<<'\n';
                                        this->ApplyForce0(_msg->data[0], _msg->data[1], _msg->data[2]);
                                        this->prevtime=this->world->SimTime();
                                        std::cerr <<"===========1============"<<'\n';
                                        std::cerr <<"Wind force: ["<<_msg->data[0]<<","<<_msg->data[1]<<","<<_msg->data[2]<<"] N \n"<<'\n';
                                        std::cerr <<common::Time::GetWallTime()<<'\n'<<'\n';
                                        std::cerr <<"+++++++++++++2+++++++++++++++"<<'\n';
                                }
                        }

                        /// \brief ROS helper function that processes messages
                private: void QueueThread()
                         {
                                 static const double timeout = 0.01;
                                 while (this->rosNode->ok())
                                 {
                                         this->rosQueue.callAvailable(ros::WallDuration(timeout));
                                 }
                         }
                         //set the forces and positions
                private:
                         //初始化风力
                         double fx=0, fy=0, fz=0;

                         // relative position of wind force applied to quadrotors
                         ignition::math::Vector3d const pos = ignition::math::Vector3d(0.0, 0.0, 0.0);

                         //state the class of the variable used
                private:
                         physics::ModelPtr drone;

                         physics::WorldPtr world;
                         //UAV
                         physics::LinkPtr toprod;
                         // Force
                         ignition::math::Vector3d force=ignition::math::Vector3d(fx,fy,fz);

                         //absolute position of nodes on the bottom link

                         //temporaty variable for force calculation

                         common::Time timeinterval=common::Time(0, common::Time::SecToNano(0.001));
                         common::Time prevtime;

                         /// \brief A node use for ROS transport
                private: std::unique_ptr<ros::NodeHandle> rosNode;

                         /// \brief A ROS subscriber
                private: ros::Subscriber rosSub0;

                         /// \brief A ROS callbackqueue that helps process messages
                private: ros::CallbackQueue rosQueue;

                         /// \brief A thread the keeps running the rosQueue
                private: std::thread rosQueueThread;

        };

        // Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
        GZ_REGISTER_WORLD_PLUGIN(TestforcePlugin)
}
#endif
