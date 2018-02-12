import roslib; roslib.load_manifest('grasping_ros_mico')
from grasping_ros_mico.srv import *
import rospy
#from moveit_commander import MoveGroupCommander
from sensor_msgs.msg import (
    JointState
)
#import motion_executor
#from motion_executor import Executor

import kinova_motion_executor_with_touch
from kinova_motion_executor_with_touch import KinovaExecutorWithTouch
import time 
from robot_config import *
import sys
import getopt

#motion_executor.THRES_TOUCH_GRASPED = 650.0
#motion_executor.THRES_TOUCH = 15.0

kinova_motion_executor_with_touch.THRES_TOUCH_GRASPED = 650.0
kinova_motion_executor_with_touch.THRES_TOUCH = 75.0

#from stop_when_touch import lift
#import task_planner.apc_util
#from task_planner.slip_classifier import slip_classifier

def joint_state_callback(data, ans):
    global finger
    pose=data.position
    ans=[pose[6],pose[7]]
    #print ans
    finger=ans


def handle_action_request(req):
    global finger
    global myKinovaMotionExecutor
    if req.action == req.ACTION_MOVE:
        myKinovaMotionExecutor.goto_relative_pose_until_touch(req.move_x, req.move_y, req.move_z)
    if req.action == req.ACTION_CLOSE:
        print "close action"
        myKinovaMotionExecutor.max_pressure = [-1000, -1000]
        myKinovaMotionExecutor.set_gripper_state('close')
    if req.action == req.ACTION_OPEN:
        print "open action"
        myKinovaMotionExecutor.max_pressure = [-1000, -1000]
        myKinovaMotionExecutor.set_gripper_state('open')
    if req.action == req.ACTION_PICK:
        print "pick action"
        myKinovaMotionExecutor.max_pressure = [-1000, -1000]
        #myKinovaMotionExecutor.goto('top_of_books')
        myKinovaMotionExecutor.goto('home_t')
    if req.action ==req.GET_TOUCH_THRESHOLD:
        print "Getting threshold"
        res = MicoActionFeedbackResponse()
        res.touch_sensor_reading = [kinova_motion_executor_with_touch.THRES_TOUCH,kinova_motion_executor_with_touch.THRES_TOUCH]
        #print myKinovaMotionExecutor.initial_pressure
        return res
    if req.action == req.INIT_POS:
        myKinovaMotionExecutor.goto('table_pre_grasp2')
        #myKinovaMotionExecutor.goto_relative_pose(dz=0.005)
        myKinovaMotionExecutor.goto_relative_pose(dy=-0.04)
        myKinovaMotionExecutor.goto_relative_pose(dy=-0.04)
    
    if req.action == req.MOVE_AWAY_POS:
        myKinovaMotionExecutor.goto('top_of_books')
        
    res = MicoActionFeedbackResponse()
    res.gripper_pose = myKinovaMotionExecutor.curr_pose #arm.get_current_pose()
    res.touch_sensor_reading =  myKinovaMotionExecutor.max_pressure
    #print myKinovaMotionExecutor.initial_pressure
    print res.touch_sensor_reading

    finger = None
    topic_address = '/' + myKinovaMotionExecutor._prefix + 'driver/out/joint_state'
    rospy.Subscriber(topic_address, JointState, joint_state_callback, res.finger_joint_state)
    while not finger :
        rospy.sleep(5)
        print "Waiting for joint state"
    res.finger_joint_state = finger
        
    ##tactile reading
    """
    #For old sensors
    cyc=5
    SENSORS = [1, 2, 3, 4]
    data=[]
    for i in range(cyc):
        for j in SENSORS:
            d=None
            while d==None:
                apc_util.sensor_write(j)
                d=apc_util.sensor_read()
            #print d
            if i==cyc-1:
                data=data+d
    res.touch_sensor_reading=data
    """

    ###stability #TODO : Replace this, is this even required?
    """
    data=task_planner.apc_util.image_moment(data)
    data=data+finger
    test_data=(data-m)/s
    result=c.predict(test_data)[0]
    """
    res.grasp_stability=0
    return res

def mico_action_feedback_server():
    
    rospy.Service('mico_action_feedback_server', MicoActionFeedback, handle_action_request)
    rospy.spin()
    
if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],"ht:")
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'mico_action_feedback_node.py -t <touch threshold value>'
         sys.exit()
      elif opt == '-t':
         kinova_motion_executor_with_touch.THRES_TOUCH = float(arg)
    """
    global arm
    arm = MoveGroupCommander("arm")
    global c,m,s,finger
    sc=slip_classifier()
    c,m,s=sc.train_svm_classifier
    """
    rospy.init_node('mico_action_feedback_server')
    global finger, myMotionExecutor, myKinovaMotionExecutor
    myKinovaMotionExecutor = KinovaExecutorWithTouch('mico_action_feedback_server')
    #myMotionExecutor = Executor()
    #myMotionExecutor.open_gripper()
    myKinovaMotionExecutor.set_gripper_state('open')
    myKinovaMotionExecutor.goto('home_t')
    time.sleep(5)
    #myKinovaMotionExecutor.goto('table_pre_grasp2')
    #myKinovaMotionExecutor.goto_relative_pose(dz=0.02)
    #myKinovaMotionExecutor.goto_relative_pose(dy=-0.04)
    #myKinovaMotionExecutor.goto_relative_pose(dy=-0.04)
    #myKinovaMotionExecutor.goto_relative_pose(dz=0.02)
    mico_action_feedback_server()
