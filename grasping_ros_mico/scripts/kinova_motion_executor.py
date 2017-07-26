#! /usr/bin/env python
"""A helper program to test cartesian goals for the JACO and MICO arms."""

#import roslib; roslib.load_manifest('kinova_demo')
import rospy

import sys
import numpy as np

import actionlib
import kinova_msgs.msg
import std_msgs.msg
import geometry_msgs.msg


import math
import argparse

#DEFAULT_FINGER_MAX_TURN = 6800
#DEFAULT_FINGER_MAX_DIST = 18.9/2/1000
#DEFAULT_HOME_POS = [0.212322831154, -0.257197618484, 0.509646713734, 1.63771402836, 1.11316478252, 0.134094119072] # default home in unit mq
# DEFAULT_ROBOT_TYPE = 'm1n6a200'
#DEFAULT_ROBOT_TYPE = 'm1n6s200'

from robot_config import *

class KinovaExecutor(object):
    def __init__(self, node=None):

        # arm settings
        self._arm_joint_number = 0
        self._finger_number = 0
        self._prefix = 'NO_ROBOT_TYPE_DEFINED_'
        self._finger_maxDist = DEFAULT_FINGER_MAX_DIST  # max distance for one finger
        self._finger_maxTurn = DEFAULT_FINGER_MAX_TURN  # max thread rotation for one finger
        self._currentCartesianCommand = DEFAULT_HOME_POS
        self._currentFingerPosition = [0.0, 0.0] # closed position
        self._currentJointCommand = [0]*7

        # specify robot configuration (default: 6DOF MICO with 2 fingers)
        self.__kinova_robotTypeParser(DEFAULT_ROBOT_TYPE)
        if node is None:
            rospy.init_node(self._prefix + 'pose_action_client')

        # latest pose and finger estimate from driver
        self.__getcurrentCartesianCommand(self._prefix)
        self.__getCurrentFingerPosition(self._prefix)
        self.__getcurrentJointCommand(self._prefix)
        
        self.location_predefined = {'home_t':HOME_POSE_T, 'home_r':HOME_POSE_R,\
                'pre_push_rot': PRE_PUSH, 'pre_grasp_rot':PRE_GRASP,\
                'pre_push_trans':PRE_PUSH_TRANS, 'pre_grasp_trans1':\
                PRE_GRASP_TRANS1, 'pre_grasp_trans': PRE_GRASP_TRANS,\
                'top_of_books': TOP_OF_BOOKS, 'tra_pose': tra_pose, 'table_pre_grasp2':
                PRE_GRASP_2, 'wallet_pre_grasp': WALLET_PRE_GRASP}
        print "Kinova Motion Controller Ready!"

    def __QuaternionNorm(self, Q_raw):
        qx_temp,qy_temp,qz_temp,qw_temp = Q_raw[0:4]
        qnorm = math.sqrt(qx_temp*qx_temp + qy_temp*qy_temp + qz_temp*qz_temp + qw_temp*qw_temp)
        qx_ = qx_temp/qnorm
        qy_ = qy_temp/qnorm
        qz_ = qz_temp/qnorm
        qw_ = qw_temp/qnorm
        Q_normed_ = [qx_, qy_, qz_, qw_]
        return Q_normed_


    def __Quaternion2EulerXYZ(self, Q_raw):
        Q_normed = self.__QuaternionNorm(Q_raw)
        qx_ = Q_normed[0]
        qy_ = Q_normed[1]
        qz_ = Q_normed[2]
        qw_ = Q_normed[3]

        tx_ = math.atan2((2 * qw_ * qx_ - 2 * qy_ * qz_), (qw_ * qw_ - qx_ * qx_ - qy_ * qy_ + qz_ * qz_))
        ty_ = math.asin(2 * qw_ * qy_ + 2 * qx_ * qz_)
        tz_ = math.atan2((2 * qw_ * qz_ - 2 * qx_ * qy_), (qw_ * qw_ + qx_ * qx_ - qy_ * qy_ - qz_ * qz_))
        EulerXYZ_ = [tx_,ty_,tz_]
        return EulerXYZ_


    def __EulerXYZ2Quaternion(self, EulerXYZ_):
        tx_, ty_, tz_ = EulerXYZ_[0:3]
        sx = math.sin(0.5 * tx_)
        cx = math.cos(0.5 * tx_)
        sy = math.sin(0.5 * ty_)
        cy = math.cos(0.5 * ty_)
        sz = math.sin(0.5 * tz_)
        cz = math.cos(0.5 * tz_)

        qx_ = sx * cy * cz + cx * sy * sz
        qy_ = -sx * cy * sz + cx * sy * cz
        qz_ = sx * sy * cz + cx * cy * sz
        qw_ = -sx * sy * sz + cx * cy * cz

        Q_ = [qx_, qy_, qz_, qw_]
        return Q_

    def __joint_angle_client(self, angle_set):
        """Send a joint angle goal to the action server.
            angle_set: list of angles in degree
        """

        action_address = '/' + self._prefix + 'driver/joints_action/joint_angles'
        client = actionlib.SimpleActionClient(action_address,
                                              kinova_msgs.msg.ArmJointAnglesAction)
        client.wait_for_server()
    
        goal = kinova_msgs.msg.ArmJointAnglesGoal()
    
        goal.angles.joint1 = angle_set[0]
        goal.angles.joint2 = angle_set[1]
        goal.angles.joint3 = angle_set[2]
        goal.angles.joint4 = angle_set[3]
        goal.angles.joint5 = angle_set[4]
        goal.angles.joint6 = angle_set[5]
        goal.angles.joint7 = angle_set[6]

        client.send_goal(goal)
        if client.wait_for_result(rospy.Duration(20.0)):
            return client.get_result()
        else:
            print('        the joint angle action timed-out')
            client.cancel_all_goals()
            return None

    def __cartesian_pose_client_send_goal(self, position, orientation):
        """Send a cartesian goal to the action server."""
        action_address = '/' + self._prefix + 'driver/pose_action/tool_pose'
        client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.ArmPoseAction)
        client.wait_for_server()

        goal = kinova_msgs.msg.ArmPoseGoal()
        goal.pose.header = std_msgs.msg.Header(frame_id=(self._prefix + 'link_base'))
        goal.pose.pose.position = geometry_msgs.msg.Point(
            x=position[0], y=position[1], z=position[2])
        goal.pose.pose.orientation = geometry_msgs.msg.Quaternion(
            x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])

        # print('goal.pose in client 1: {}'.format(goal.pose.pose)) # debug

        client.send_goal(goal)
        return client

    def __cartesian_pose_client(self, position, orientation):
        """Send a cartesian goal to the action server."""
        client = self.__cartesian_pose_client_send_goal(position, orientation)
        
        """ Get result """
        if client.wait_for_result(rospy.Duration(10.0)):
            return client.get_result()
        else:
            client.cancel_all_goals()
            print('        the cartesian action timed-out')
            return None


    def __gripper_client(self, finger_positions):
        """Send a gripper goal to the action server."""
        action_address = '/' + self._prefix + 'driver/fingers_action/finger_positions'

        client = actionlib.SimpleActionClient(action_address,
                                              kinova_msgs.msg.SetFingersPositionAction)
        client.wait_for_server()

        goal = kinova_msgs.msg.SetFingersPositionGoal()
        goal.fingers.finger1 = float(finger_positions[0])
        goal.fingers.finger2 = float(finger_positions[1])

        client.send_goal(goal)
        
        if client.wait_for_result(rospy.Duration(5.0)):
            return client.get_result()
        else:
            client.cancel_all_goals()
            rospy.WARN('        the gripper action timed-out')
            return None

    def __getcurrentJointCommand(self, prefix_):
        # wait to get current position
        topic_address = '/' + prefix_ + 'driver/out/joint_command'
        rospy.Subscriber(topic_address, kinova_msgs.msg.JointAngles, self.__setcurrentJointCommand)
        rospy.wait_for_message(topic_address, kinova_msgs.msg.JointAngles)
        print 'position listener obtained message for joint position. '

    
    def __setcurrentJointCommand(self, feedback):
    
        currentJointCommand_str_list = str(feedback).split("\n")
        for index in range(0,len(currentJointCommand_str_list)):
            temp_str=currentJointCommand_str_list[index].split(": ")
            self._currentJointCommand[index] = float(temp_str[1])
    
        # print 'currentJointCommand is: '
        # print currentJointCommand

    def __getcurrentCartesianCommand(self, prefix_):
        # wait to get current position
        topic_address = '/' + prefix_ + 'driver/out/cartesian_command'
        #print topic_address
        rospy.Subscriber(topic_address, kinova_msgs.msg.KinovaPose, self.__setcurrentCartesianCommand)
        rospy.wait_for_message(topic_address, kinova_msgs.msg.KinovaPose)
        # print 'position listener obtained message for Cartesian pose. '


    def __setcurrentCartesianCommand(self, feedback):

        currentCartesianCommand_str_list = str(feedback).split("\n")

        for index in range(0,len(currentCartesianCommand_str_list)):
            temp_str=currentCartesianCommand_str_list[index].split(": ")
            self._currentCartesianCommand[index] = float(temp_str[1])
        # the following directly reading only read once and didn't update the value.
        # self._currentCartesianCommand = [feedback.X, feedback.Y, feedback.Z, feedback.ThetaX, feedback.ThetaY, feedback.Z] 
        # print 'self._currentCartesianCommand in __setcurrentCartesianCommand is: ', self._currentCartesianCommand


    def __getCurrentFingerPosition(self, prefix_):
        # wait to get current position
        topic_address = '/' + prefix_ + 'driver/out/finger_position'
        rospy.Subscriber(topic_address, kinova_msgs.msg.FingerPosition, self.__setCurrentFingerPosition)
        rospy.wait_for_message(topic_address, kinova_msgs.msg.FingerPosition)
        # print 'obtained current finger position '


    def __setCurrentFingerPosition(self, feedback):
        self._currentFingerPosition[0] = feedback.finger1
        self._currentFingerPosition[1] = feedback.finger2


    def __kinova_robotTypeParser(self, kinova_robotType_):
        """ Argument kinova_robotType """
        self._robot_category = kinova_robotType_[0]
        self._robot_category_version = int(kinova_robotType_[1])
        self._wrist_type = kinova_robotType_[2]
        self._arm_joint_number = int(kinova_robotType_[3])
        self._robot_mode = kinova_robotType_[4]
        self._finger_number = int(kinova_robotType_[5])
        self._prefix = kinova_robotType_ + "_"
        self._finger_maxDist = 18.9/2/1000  # max distance for one finger in meter
        self._finger_maxTurn = 6800  # max thread turn for one finger

    
    def goto(self, named_config):
        """
            move robot arm to a named joint configuration (degree)
        """
        #self.goto_global_joint(self.location_predefined[named_config])
        self.goto_global_pose(self.location_predefined[named_config])

    def goto_global_joint(self, joint_degree):
        """
            set robot arm joint configuration in degree
        """
        result = None
        try:
            result = self.__joint_angle_client(joint_degree)
        except rospy.ROSInterruptException:
            print "program interrupted before completion"

        return result

    def goto_global_pose(self, pose):
        """ 
            set end-effector 6DOF pose with respect to the base_link 
            input: geometry_msgs.PoseStamped (meters + quaternion)
            output: goal result
        """

        # decompose pose into array
        position_ = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
        orientation_ = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]

        result = None

        try:
            result = self.__cartesian_pose_client(position_, orientation_)
        except rospy.ROSInterruptException:
            print "program interrupted before completion"

        return result

    
    def __get_cartesian_goal_from_relative_pose(self, dx=0, dy=0, dz=0, droll=0, dpitch=0, dyaw=0):
        """ create position and orientation from relative pose input"""
        
        self.__getcurrentCartesianCommand(self._prefix)

        # decompose pose into array
        position_ = [self._currentCartesianCommand[0]+dx, self._currentCartesianCommand[1]+dy, self._currentCartesianCommand[2]+dz] 
        orientation_ = [droll, dpitch, dyaw]

        orientation_deg_list = list(map(math.degrees, self._currentCartesianCommand[3:]))
        orientation_deg = [orientation_[i] + orientation_deg_list[i] for i in range(0,3)]
        orientation_rad = list(map(math.radians, orientation_deg))
        orientation_q = self.__EulerXYZ2Quaternion(orientation_rad)
        return(position_, orientation_q)

    def get_cartesian_goal_from_relative_pose(self, dx=0, dy=0, dz=0, droll=0, dpitch=0, dyaw=0):
        return self.__get_cartesian_goal_from_relative_pose(dx,dy,dz,droll,dpitch,dyaw)

    def cartesian_pose_client_send_goal(self, position, orientation):
        return self.__cartesian_pose_client_send_goal(position, orientation)

    def goto_relative_pose(self, dx=0, dy=0, dz=0, droll=0, dpitch=0, dyaw=0):
        """ 
            set end-effector 6DOF pose with respect to current pose
            input: geometry_msgs.PoseStamped (meters + degrees)
            output: goal result
        """
        (position_, orientation_q) = self.__get_cartesian_goal_from_relative_pose(dx,dy,dz,droll,dpitch,dyaw)
        

        try:
            result = self.__cartesian_pose_client(position_, orientation_q)
        except rospy.ROSInterruptException:
            print "program interrupted before completion"

        return result       

    def get_current_pose(self):
        """ 
            get end-effector pose with respect to base_link
            output: geometry_msgs.PoseStamped (meters + quaternion)
        """
        self.__getcurrentCartesianCommand(self._prefix)

        pose = geometry_msgs.msg.PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self._prefix + 'link_base'
        pose.pose.position.x = self._currentCartesianCommand[0]
        pose.pose.position.y = self._currentCartesianCommand[1]
        pose.pose.position.z = self._currentCartesianCommand[2]
        quat = self.__EulerXYZ2Quaternion(self._currentCartesianCommand[3:])
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        return pose


    def set_gripper_state(self, state='close'):
        """ 
            set gripper state (binary): open or close
            input: string ('close' or 'open')
            output: goal result
        """

        # self.__getCurrentFingerPosition(self._prefix)

        positions = [0.0, 0.0]
        if state == 'open':
            positions = [0.0, 0.0]
        elif state == 'close':
            positions = [self._finger_maxTurn, self._finger_maxTurn]
        else:
            rospy.ERROR('Invalid gripper state. Options: close OR open')
            return None

        return self.__gripper_client(positions)


