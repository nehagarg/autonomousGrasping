#! /usr/bin/env python
import rospy
import actionlib
from std_msgs.msg import String, Int16MultiArray, Float32, Float64
from mico_motion_planner.msg import *
import pickle
import copy
import os
import sys
import random

SERVER_NAME = 'mico_motion_planner'
#THRES_TOUCH_GRASPED = 1400
THRES_TOUCH_GRASPED = 300.0
THRES_TOUCH = 150.0

class Executor(object):
    def __init__(self):
        self.client = actionlib.SimpleActionClient(SERVER_NAME, MotionGoalAction)
        self.client.wait_for_server()
        self.pub_gripper_vel = rospy.Publisher('/gripper_state_vel', String, queue_size=10,latch=True)
        self.pub_gripper_pose = rospy.Publisher('/gripper_state_pose', Int16MultiArray, queue_size=10,latch=True)
        self.pub_max_speed = rospy.Publisher('/joint_max_speed', Float64, queue_size=1, latch=True)
        self.sub_touch_r = rospy.Subscriber('/touch_r', Float32, self.cb_touch, 1)
        self.sub_pressure_r = rospy.Subscriber('/pressure_calib_r', Float32, self.cb_pressure, 1)
        self.sub_touch_l = rospy.Subscriber('/touch_l', Float32, self.cb_touch, 0)
        self.sub_pressure_l = rospy.Subscriber('/pressure_calib_l', Float32, self.cb_pressure, 0)
        self.curr_pose = None
        self.last_touch = [0, 0]
        self.max_pressure = [0.0, 0.0]
        self.initial_pressure = [None, None]
        self.detected_pressure = [0, 0]

    def cb_touch(self, msg, finger_index):
        print finger_index
        print 'touch=', msg.data
        self.last_touch[finger_index] = msg.data

    def cb_pressure(self, msg, finger_index):
        pressure = msg.data
        if self.initial_pressure[finger_index] is None:
            self.initial_pressure[finger_index] = pressure
        a = self.initial_pressure[finger_index]
        self.max_pressure[finger_index] = max(self.max_pressure[finger_index] - a, pressure - a)
        
    

    def speed(self, speed):
        self.pub_max_speed.publish(speed)

    def goto(self, named_goal):
        goal = MotionGoalGoal(goal_pose=None, joint_target=named_goal, stored_trajectory=None, is_cartesian_path=None)
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        self.curr_pose = result.curr_eef_pose
        #print self.curr_pose
        return result.success

    def plan_move(self, dx=0, dy=0, dz=0, from_pose=None):
        from_pose = from_pose or self.curr_pose
        pose = copy.deepcopy(from_pose)
        pose.pose.position.x += dx
        pose.pose.position.y += dy
        pose.pose.position.z += dz
        goal = MotionGoalGoal(goal_pose=[self.curr_pose, pose], joint_target='', stored_trajectory=None, is_cartesian_path=True)
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        assert result.success
        return result.plans[1]

    def move(self, dx=0, dy=0, dz=0, from_pose=None):
        print 'move: dx=%.4f, dy=%.4f, dz=%.4f' % (dx, dy, dz)
        plan = self.plan_move(dx, dy, dz, from_pose)
        return self.execute_plan(plan)

    def execute_plan(self, plan, check_need_cancel=None):
        '''
        execute stored trajectory plan
        '''
        goal = MotionGoalGoal(goal_pose=None, joint_target='', stored_trajectory=None, plan=plan, is_cartesian_path=None)
        self.client.send_goal(goal)
        if not check_need_cancel:
            self.client.wait_for_result()
        else:
            while not self.client.wait_for_result(rospy.Duration(0.01)):
                if check_need_cancel():
                    rospy.loginfo('cancelling...')
                    self.client.cancel_goal()

        result = self.client.get_result()
        self.curr_pose = result.curr_eef_pose
        return result.success

    def gripper_action(self, state, wait=2):
        '''
        Two ways of gripper control:
        1. velocity control by specifying state of gripper, eg. open, close
        2. position control by specifying joint angle of gripper, eg. [0, 0], [6000, 6000]
        '''
        if isinstance(state, str):
            pub = self.pub_gripper_vel
            msg = state
        else:
            pub = self.pub_gripper_pose
            msg = Int16MultiArray()
            msg.data = state

        pub.publish(msg)
        rospy.sleep(wait)
        self.pub_gripper_vel.publish('stop')

    def close_gripper(self):
        self.last_touch = [0,0]
        self.max_pressure = [0,0]
        self.gripper_action('close')

    def open_gripper(self):
        self.gripper_action('open')

    @property
    def is_last_grasp_success(self):
        #return self.last_touch > THRES_TOUCH_GRASPED
        return ((self.max_pressure[0] > THRES_TOUCH_GRASPED) or (self.max_pressure[1] > THRES_TOUCH_GRASPED))

    @property
    def is_touched(self):
        self.detected_pressure = self.max_pressure
        return ((self.max_pressure[0] > THRES_TOUCH) or ((self.max_pressure[1] > THRES_TOUCH)))

    def move_until_touch_old(self, dx=0, dy=0, dz=0, max_count=20):
        from_pose = copy.deepcopy(self.curr_pose)
        x=y=z=0
        while max_count > 0:
            max_count -= 1
            x+=dx
            y+=dy
            z+=dz
            self.max_pressure = -1000
            self.move(x, y, z, from_pose)
            rospy.loginfo('move_until_touch max_pressure=%.4f', self.max_pressure)
            if self.max_pressure > THRES_TOUCH:
                break
        if max_count <= 0:
            rospy.logwarn('move_until_touch failed')

    def move_until_touch(self, dx=0, dy=0, dz=0):
        print 'move_until_touch: dx=%.4f, dy=%.4f, dz=%.4f' % (dx, dy, dz)
        plan = self.plan_move(dx, dy, dz)
        self.max_pressure = [-1000, -1000]
        self.execute_plan(plan, check_need_cancel=lambda: self.is_touched)


