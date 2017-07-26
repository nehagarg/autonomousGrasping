import roslib; roslib.load_manifest('grasping_ros_mico')
from grasping_ros_mico.srv import *
import rospy
#from moveit_commander import MoveGroupCommander
from sensor_msgs.msg import (
    JointState
)
import motion_executor
from motion_executor import Executor

import kinova_motion_executor_with_touch
from kinova_motion_executor_with_touch import KinovaExecutorWithTouch
import time 

motion_executor.THRES_TOUCH_GRASPED = 650.0
motion_executor.THRES_TOUCH = 15.0

kinova_motion_executor_with_touch.THRES_TOUCH_GRASPED = 650.0
kinova_motion_executor_with_touch.THRES_TOUCH = 15.0

#from stop_when_touch import lift
#import task_planner.apc_util
#from task_planner.slip_classifier import slip_classifier

def joint_state_callback(data, ans):
    global finger
    pose=data.position
    ans=[pose[6],pose[8]]
    #print ans
    finger=ans


def handle_action_request(req):
    global finger
    global myMotionExecutor
    global myKinovaMotionExecutor
    if req.action == req.ACTION_MOVE:
        myKinovaMotionExecutor.goto_relative_pose_until_touch(req.move_x, req.move_y, req.move_z)
    if req.action == req.ACTION_CLOSE:
        #print "here"
        myKinovaMotionExecutor.set_gripper_state('close')
    if req.action == req.ACTION_OPEN:
        myKinovaMotionExecutor.set_gripper_state('open')
    if req.action == req.ACTION_PICK:
        myMotionExecutor.goto('top_of_books')
    
    res = MicoActionFeedbackResponse()
    res.gripper_pose = myKinovaMotionExecutor.curr_pose #arm.get_current_pose()
    res.touch_sensor_reading =  myMotionExecutor.detected_pressure
    print res.touch_sensor_reading

    finger = None
    rospy.Subscriber("/joint_states", JointState, joint_state_callback, res.finger_joint_state)
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
    """
    global arm
    arm = MoveGroupCommander("arm")
    global c,m,s,finger
    sc=slip_classifier()
    c,m,s=sc.train_svm_classifier
    """
    rospy.init_node('mico_action_feedback_server')
    global finger, myMotionExecutor, myKinovaMotionExecutor
    myKinovaMotionExecutor = KinovaExecutorWithTouch()
    myMotionExecutor = Executor()
    myMotionExecutor.open_gripper()
    myMotionExecutor.goto('home')
    time.sleep(10)
    myMotionExecutor.goto('table_pre_grasp2')
    myKinovaMotionExecutor.goto_relative_pose(dy=-0.04)
    myKinovaMotionExecutor.goto_relative_pose(dy=-0.04)
    mico_action_feedback_server()