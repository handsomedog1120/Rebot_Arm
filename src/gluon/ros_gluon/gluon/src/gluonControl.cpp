/*
*执行器位置控制。订阅话题/joint_states，将得到的关节数据发给执行器执行。
*/

#include <iostream>
#include "actuatorcontroller.h"
#include <thread>
#include <signal.h>
#include <string.h>
#include <chrono>
#include <queue>
#include <vector>

#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>
#include <std_msgs/Int8.h>
#include <std_msgs/String.h>

#define DEG_TO_RAD(x) ((x)*M_PI / 180.0)
#define RAD_TO_DEG(x) ((x)*180.0 / M_PI)

#define STEERING_GEAR_RATIO 36

#define RAD_TO_POS(x) ((x / (2 * M_PI)) * STEERING_GEAR_RATIO)
#define POS_TO_RAD(x) ((x * (2 * M_PI)) / STEERING_GEAR_RATIO)

using namespace std;

static queue<sensor_msgs::JointState> s_jointStatesList;

static double sensor_pos[6] = {0, 0, 0, 0, 0, 0};
static sensor_msgs::JointState p_sensor_js;

void jointStatesCallback(const sensor_msgs::JointStatePtr msg)
{
  if (msg == NULL) return;
  s_jointStatesList.push(*msg);
  ROS_INFO("111");
  if (s_jointStatesList.size() >= 10) {
    s_jointStatesList.pop();
    ROS_WARN("jointStatesCallback, pop");
  }
}

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "gluonControl");
  ros::NodeHandle nh_;


  // 实例化订阅者为sub_jointstates，订阅话题/joint_states，回调函数为jointStatesCallback
  ros::Subscriber sub_jointstates = nh_.subscribe("/joint_states", 5, jointStatesCallback);

  // (jzm)实例化发布者为pub_jointstates，发布话题/sensor_jointstates，【队列中最大保存的消息数，超出时，先进的先销毁(时间早的先销毁)】
  ros::Publisher pub_jointstates = nh_.advertise<sensor_msgs::JointState>("/sensor_jointstates", 10);

    //+----------------------------------------------------------------------------------------------------------------+
    //Intialize the controller
    ActuatorController * pController = ActuatorController::initController();
    //ec Define an error type, ec==0x00 means no error, ec will be passed to pcontroller-> lookupActuators(ec) by reference,
    //when the error occurs, ec value will be modified by SDK to the corresponding error code
    Actuator::ErrorsDefine ec;
    //Find the connected actuators and return the UnifiedID of all actuators found.
    std::vector<ActuatorController::UnifiedID> uIDArray = pController->lookupActuators(ec);
    //If the size of the uIDArray is greater than zero, the connected actuators have been found
    //+----------------------------------------------------------------------------------------------------------------+

    //+----------------------------------------------------------------------------------------------------------------+
    //activate the controllers
    if (uIDArray.size() > 0)
    {
        for(int k = 0; k < uIDArray.size(); k++) {
          ActuatorController::UnifiedID actuator = uIDArray.at(k);
          //初始化执行器Enable actuator
          ROS_INFO("jzm_actuator ID %d, ipAddr %s", actuator.actuatorID,actuator.ipAddress.c_str());
          pController->enableActuator(actuator.actuatorID,actuator.ipAddress);
          //激活位置控制模式activate profile position mode
          pController->activateActuatorMode(actuator.actuatorID,Actuator::Mode_Profile_Pos);

          //设置pid参数
          pController->setPositionKp(actuator.actuatorID, 0.15);
          pController->setPositionKi(actuator.actuatorID, 0.00);
          pController->setPositionKd(actuator.actuatorID, 0.00);
          //显示pid参数
          ROS_INFO("ID = %d, Kp, Ki, Kd = %f, %f, %f", actuator.actuatorID, pController->getPositionKp(actuator.actuatorID, true),
                     pController->getPositionKi(actuator.actuatorID, true), pController->getPositionKd(actuator.actuatorID, true));

          cout << "set position to 10 revolutions " << endl;
          //回到零位置
          pController->setPosition(actuator.actuatorID,0);
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        //Disable all connected actuators
        //pController->disableAllActuators();
        //insure that all actuators have been closed
        //std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    else
    {
        cout << "Connected error code:" << hex << ec << endl;
        return -1;
    }
    //+----------------------------------------------------------------------------------------------------------------+

    ros::Rate loop_rate(50);
    while(ros::ok()) {
      ros::spinOnce();
      loop_rate.sleep();


      //+--------------------------------------------------------------------------------------------------------------+
      //(jzm)获取每个传感器数据，存在sensor_pos，写入pub_jointstates.position中，发布给话题/sensor_jointstates
      for(int k = 0; k < 6; k++) {
        ActuatorController::UnifiedID actuator = uIDArray.at(k);
        sensor_pos[k] = POS_TO_RAD(pController->getPosition(actuator.actuatorID, true));
      }

//      for (int j = 0; j < 6; j++) {
//        printf("q%d:%f|", j+1, sensor_pos[j]);
//      }
//      printf("\n");

      //update p_sensor_js
      p_sensor_js.header.stamp = ros::Time::now();
      p_sensor_js.name.resize(6);
      p_sensor_js.position.resize(6);
      p_sensor_js.name[0]="q1";
      p_sensor_js.position[0] = sensor_pos[0];
      p_sensor_js.name[1] ="q2";
      p_sensor_js.position[1] = sensor_pos[1];
      p_sensor_js.name[2] ="q3";
      p_sensor_js.position[2] = sensor_pos[2];
      p_sensor_js.name[3] ="q4";
      p_sensor_js.position[3] = sensor_pos[3];
      p_sensor_js.name[4] ="q5";
      p_sensor_js.position[4] = sensor_pos[4];
      p_sensor_js.name[5] ="q6";
      p_sensor_js.position[5] = sensor_pos[5];

      pub_jointstates.publish(p_sensor_js);
      //+--------------------------------------------------------------------------------------------------------------+


      //jointstates话题接受到消息存在这个s_jointStatesList）把关节角发送到各个执行器:
      if (!s_jointStatesList.empty()) {
        sensor_msgs::JointState js_data = s_jointStatesList.front();
        s_jointStatesList.pop();
        for(int i = 0; i < 6; i++) {
          ActuatorController::UnifiedID actuator = uIDArray.at(i);
          //ROS_INFO("jointstate_%d, %s, %f", i+1, js_data.name[i].c_str(), js_data.position[i]);
          //set Position
          pController->setPosition(actuator.actuatorID, RAD_TO_POS(js_data.position[i]));

          //(jzm)get Postion
          //ROS_INFO("%f", pController->getPosition(actuator.actuatorID, true));
        }
      }
    }

    return 0;
}
