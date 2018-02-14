#! /usr/bin/env python
"""A helper program to test cartesian goals for the JACO and MICO arms."""

import rospy
import kinova_motion_executor
from kinova_motion_executor import KinovaExecutor
from std_msgs.msg import String, Int16MultiArray, Float32, Float64

kinova_motion_executor.DEFAULT_FINGER_MAX_TURN = 6800
kinova_motion_executor.DEFAULT_FINGER_MAX_DIST = 18.9/2/1000
kinova_motion_executor.DEFAULT_HOME_POS = [0.212322831154, -0.257197618484, 0.509646713734, 1.63771402836, 1.11316478252, 0.134094119072] # default home in unit mq
# kinova_motion_executor.DEFAULT_ROBOT_TYPE = 'm1n6a200'
kinova_motion_executor.DEFAULT_ROBOT_TYPE = 'm1n6s200'
THRES_TOUCH_GRASPED = 300
THRES_TOUCH = 150

class KinovaExecutorWithTouch(KinovaExecutor):
    def __init__(self, node=None):
        
        self.sub_touch_r = rospy.Subscriber('/touch_r', Float32, self.cb_touch, 1)
        self.sub_pressure_r = rospy.Subscriber('/pressure_calib_r', Float32, self.cb_pressure, 1)
        self.sub_touch_l = rospy.Subscriber('/touch_l', Float32, self.cb_touch, 0)
        self.sub_pressure_l = rospy.Subscriber('/pressure_calib_l', Float32, self.cb_pressure, 0)
        self.sub_vision_movement = rospy.Subscriber('/object_vision_movement', Int8, self.cb_vision_movement)
        #self.curr_pose = None
        self.last_touch = [0, 0]
        self.max_pressure = [0.0, 0.0]
        self.initial_pressure = [None, None]
        self.detected_pressure = [0, 0]
        self.vision_movement = 0
        super( KinovaExecutorWithTouch, self ).__init__(node)

    def cb_vision_movement(self, msg):
        self.vision_movement = msg.data
        
    def cb_touch(self, msg, finger_index):
        print finger_index
        print 'touch=', msg.data
        print 'initial pressure =' , self.initial_pressure[finger_index]
        self.last_touch[finger_index] = msg.data
        self.max_pressure[finger_index] = msg.data

    def cb_pressure(self, msg, finger_index):
        pressure = msg.data
        #print finger_index
        #print 'pressure=', msg.data
        if self.initial_pressure[finger_index] is None:
            self.initial_pressure[finger_index] = pressure
        a = 0 #self.initial_pressure[finger_index]
        self.max_pressure[finger_index] = max(self.max_pressure[finger_index], pressure - a)

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
    
    @property
    def is_last_grasp_success(self):
        #return self.last_touch > THRES_TOUCH_GRASPED
        return ((self.max_pressure[0] > THRES_TOUCH_GRASPED) or (self.max_pressure[1] > THRES_TOUCH_GRASPED))

    @property
    def is_touched(self):
        self.detected_pressure = self.max_pressure
        return ((self.max_pressure[0] > THRES_TOUCH) or ((self.max_pressure[1] > THRES_TOUCH)))
    
    @property
    def has_moved(self):
        return self.vision_movement == 1
    @property
    def curr_pose(self):
        return self.get_current_pose()
    
    #move_until_touch in motion executor
    def goto_relative_pose_until_touch(self, dx=0, dy=0, dz=0, droll=0, dpitch=0, dyaw=0, check_touch=True, check_vision_movement = False):
        """ 
            set end-effector 6DOF pose with respect to current pose while checking for touch every 1 ms
            input: geometry_msgs.PoseStamped (meters + degrees)
            output: goal result
        """
        print 'move_until_touch: dx=%.4f, dy=%.4f, dz=%.4f' % (dx, dy, dz)
        self.max_pressure = [-1000, -1000]
        self.vision_movement = 0
        check_need_cancel = lambda: (check_touch and self.is_touched) or (check_vision_movement and self.has_moved)
        
        (position_, orientation_q) = self.get_cartesian_goal_from_relative_pose(dx,dy,dz,droll,dpitch,dyaw)
        
        try:
            client = self.cartesian_pose_client_send_goal(position_, orientation_q)
        
            while not client.wait_for_result(rospy.Duration(0.01)):
                    if check_need_cancel():
                        rospy.loginfo('cancelling...')
                        client.cancel_goal()

            result = client.get_result()
        except rospy.ROSInterruptException:
            print "program interrupted before completion"

        return result
