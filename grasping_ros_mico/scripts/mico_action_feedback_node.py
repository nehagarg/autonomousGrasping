import roslib; roslib.load_manifest('grasping_ros_mico')
from grasping_ros_mico.srv import *
import rospy
#from moveit_commander import MoveGroupCommander
from sensor_msgs.msg import (
    JointState
)
from std_msgs.msg import Int8
#import motion_executor
#from motion_executor import Executor

import kinova_motion_executor_with_touch
from kinova_motion_executor_with_touch import KinovaExecutorWithTouch
import time 
from robot_config import *
import sys
import getopt
sys.path.append('../../python_scripts')
import get_initial_object_belief as giob

#motion_executor.THRES_TOUCH_GRASPED = 650.0
#motion_executor.THRES_TOUCH = 15.0

kinova_motion_executor_with_touch.THRES_TOUCH_GRASPED = 650.0
kinova_motion_executor_with_touch.THRES_TOUCH = 75.0

#from stop_when_touch import lift
#import task_planner.apc_util
#from task_planner.slip_classifier import slip_classifier
class MicoActionRequestHandler():
    def __init__(self, motionExector, visionMovementDetector):
        self.myKinovaMotionExecutor = motionExector
        self.finger = None
        self.dummy = -1
        self.vision_movement_publisher = rospy.Publisher('start_movement_detection', Int8, queue_size=10)
        self.visionMovementDetector = visionMovementDetector
        
    def joint_state_callback(self,data, ans):
        #global finger
        pose=data.position
        ans=[pose[6],pose[7]]
        #print ans
        self.finger=ans

    def has_object_moved_callback(self,msg):
        #global dummy
        self.dummy = msg.data
        print "In dummy callback"
        print self.dummy

    def handle_action_request(self, req):
        #global finger
        #global myKinovaMotionExecutor
        #global vision_movement_publisher
        #global dummy

        if req.check_vision_movement:
            a_send = 1
            self.vision_movement_publisher.publish(Int8(a_send))
            #dummy = -1
            #rospy.Subscriber('object_vision_movement', Int8, has_object_moved_callback)
            while self.visionMovementDetector !=0 :
                #print dummy
                rospy.sleep(5)
                print "Waiting for point cloud update"
                vision_movement_publisher.publish(Int8(a_send))

        if req.action == req.ACTION_MOVE:
            self.myKinovaMotionExecutor.goto_relative_pose_until_touch(req.move_x, req.move_y, req.move_z, 
            check_touch=req.check_touch, check_vision_movement = req.check_vision_movement)
        if req.action == req.ACTION_CLOSE:
            print "close action"
            self.myKinovaMotionExecutor.max_pressure = [-1000, -1000]
            myKinovaMotionExecutor.set_gripper_state('close')
        if req.action == req.ACTION_OPEN:
            print "open action"
            self.myKinovaMotionExecutor.max_pressure = [-1000, -1000]
            self.myKinovaMotionExecutor.set_gripper_state('open')
        if req.action == req.ACTION_PICK:
            print "pick action"
            self.myKinovaMotionExecutor.max_pressure = [-1000, -1000]
            #myKinovaMotionExecutor.goto('top_of_books')
            self.myKinovaMotionExecutor.goto('home_t')
        if req.action ==req.GET_TOUCH_THRESHOLD:
            print "Getting threshold"
            res = MicoActionFeedbackResponse()
            res.touch_sensor_reading = [kinova_motion_executor_with_touch.THRES_TOUCH,kinova_motion_executor_with_touch.THRES_TOUCH]
            #print myKinovaMotionExecutor.initial_pressure
            return res
        if req.action == req.INIT_POS:
            self.myKinovaMotionExecutor.goto('table_pre_grasp2')
            #myKinovaMotionExecutor.goto_relative_pose(dz=0.005)
            self.myKinovaMotionExecutor.goto_relative_pose(dy=-0.04)
            self.myKinovaMotionExecutor.goto_relative_pose(dy=-0.04)

        if req.action == req.MOVE_AWAY_POS:
            self.myKinovaMotionExecutor.goto('top_of_books')


        res = MicoActionFeedbackResponse()
        res.gripper_pose = self.myKinovaMotionExecutor.curr_pose #arm.get_current_pose()
        res.touch_sensor_reading =  self.myKinovaMotionExecutor.max_pressure
        res.vision_movement = self.myKinovaMotionExecutor.vision_movement
        #print myKinovaMotionExecutor.initial_pressure
        print res.touch_sensor_reading
        print res.vision_movement

        self.finger = None
        topic_address = '/' + self.myKinovaMotionExecutor._prefix + 'driver/out/joint_state'
        rospy.Subscriber(topic_address, JointState, self.joint_state_callback, res.finger_joint_state)
        while not self.finger :
            rospy.sleep(5)
            print "Waiting for joint state"
        res.finger_joint_state = finger

        if req.check_vision_movement:
            a_send =-1
            self.vision_movement_publisher.publish(Int8(a_send))
            #dummy = -1
            #rospy.Subscriber('object_vision_movement', Int8, has_object_moved_callback)
            while self.visionMovementDetector !=a_send :
                rospy.sleep(5)
                print "Waiting for point cloud detection freeze"
                #vision_movement_publisher.publish(Int8(a_send))

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

def mico_action_feedback_server(h):
    
    rospy.Service('mico_action_feedback_server', MicoActionFeedback, h.handle_action_request)
    rospy.spin()


class VisionMovementDetector(object):
    def __init__(self, motionExector):
        self.start = -1
        self.object_moved = 3
        #self.not_processing = True
        self.pub_vision_movement = rospy.Publisher('object_vision_movement', Int8, queue_size=10)
        self.motion_executor = motionExector
        self.point_cloud_1 = self.get_current_point_cloud()
        self.pointCloudProcessor = giob.GetInitialObjectBelief(None,False,False,'real')
        self.T_cam_world = self.pointCloudProcessor.sensor.get_T_cam_world(
        self.pointCloudProcessor.CAM_FRAME, self.pointCloudProcessor.WORLD_FRAME, 
        self.pointCloudProcessor.config_path)
        self.point_cloud_topic = self.pointCloudProcessor.config['kinect_sensor_cfg']['cam_point_cloud']
        rospy.Subscriber(self.point_cloud_topic, PointCloud2, d.p_callback)
        
    def get_current_point_cloud(self,point_cloud_cam = None):
        if point_cloud_cam is None:
            point_cloud_cam = rospy.wait_for_message(self.point_cloud_topic, PointCloud2)
        point_cloud_world = T_camera_world * point_cloud_cam
        min_x = self.motion_executor.curr_pose.pose.position.x
        min_z = 0.40 #self.motion_executor.curr_pose.pose.position.z
        
        cfg = copy.deepcopy(self.pointCloudProcessor.detector_cfg )
        cfg['min_pt'][2] = cfg['min_z_for_movement']
        if min_z is not None:
            cfg['min_pt'][2] = min_z
        if min_x > cfg['min_pt'][0]:
            cfg['min_pt'][0] = min_x
        seg_point_cloud_world = self.pointCloudProcessor.get_segmented_point_cloud_world(cfg, point_cloud_world )
        
        #point_cloud_1 = get_current_point_cloud_for_movement(min_x + 0.04,False, False, 'real', min_z)
        return seg_point_cloud_world
    
    def check_vision_movement(self,point_cloud_cam):
        if self.start >= 0:
            self.object_moved = 3
            #get point cloud 2
            point_cloud_2 = self.get_current_point_cloud(point_cloud_cam)
            #check has object moved
            self.object_moved = giob.has_object_moved(self.point_cloud_1, point_cloud_2)
            print self.object_moved
            self.pub_vision_movement.publish(self.object_moved)
        

    def s_callback(self, msg):
        self.start = msg.data
        if self.start == 1:
            self.point_cloud_1 = self.get_current_point_cloud()
            self.start = 0
            
        
    def p_callback(self, msg):
        self.check_vision_movement(msg)
        
    
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
    #global finger, myMotionExecutor, myKinovaMotionExecutor, vision_movement_publisher
    
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
    
    d = VisionMovementDetector(myKinovaMotionExecutor)
    rospy.Subscriber('/start_movement_detection', Int8, d.s_callback)
    
    h = MicoActionRequestHandler(myKinovaMotionExecutor, d)
    
    mico_action_feedback_server()
