import roslib; roslib.load_manifest('grasping_ros_mico')
from grasping_ros_mico.srv import *
import rospy
#from moveit_commander import MoveGroupCommander
from sensor_msgs.msg import (
    JointState
)
import motion_executor
from motion_executor import Executor

motion_executor.THRES_TOUCH_GRASPED = 650
motion_executor.THRES_TOUCH = 15

#from stop_when_touch import lift
#import task_planner.apc_util
#from task_planner.slip_classifier import slip_classifier

def joint_state_callback(data, ans):
    global finger
    pose=data.position
    ans=[pose[6],pose[8]]
    finger=ans


def handle_action_request(req):
    global finger
    global myMotionExecutor
    if req.action == req.ACTION_MOVE:
        myMotionExecutor.move_until_touch(req.move_x, req.move_y, req.move_z)
    res = MicoActionFeedbackResponse()
    res.gripper_pose = myMotionExecutor.curr_pose #arm.get_current_pose()
    res.touch_sensor_reading =  myMotionExecutor.detected_pressure
    
    finger = None
    rospy.Subscriber("/joint_states", JointState, joint_state_callback, res.finger_joint_state)
    while not finger :
        rospy.sleep(5)
        print "Waiting for joint state"
        
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
    rospy.init_node('mico_action_feedback_server')
    rospy.Service('mico_action_feedback', MicoActionFeedback, handle_action_request)
    rospy.spin()
    
if __name__ == main():
    """
    global arm
    arm = MoveGroupCommander("arm")
    global c,m,s,finger
    sc=slip_classifier()
    c,m,s=sc.train_svm_classifier
    """
    
    global finger, myMotionExecutor
    myMotionExecutor = Executor()
    mico_action_feedback_server()