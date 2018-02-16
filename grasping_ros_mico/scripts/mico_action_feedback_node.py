import roslib; roslib.load_manifest('grasping_ros_mico')
from grasping_ros_mico.srv import *
import rospy
#from moveit_commander import MoveGroupCommander
from sensor_msgs.msg import JointState, Image, PointCloud2
from autolab_core import PointCloud
import numpy as np
import sensor_msgs.point_cloud2 as pcl2
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
import copy
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
        #self.vision_movement_publisher = rospy.Publisher('start_movement_detection', Int8, queue_size=10)
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
        print "Recieved request " + repr(req.action)
        if req.check_vision_movement:
            self.visionMovementDetector.update_initial_point_cloud()
            self.visionMovementDetector.start = 0
            
            #a_send = 1
            #self.vision_movement_publisher.publish(Int8(a_send))
            
            #while self.visionMovementDetector.start !=0 :
                #print dummy
            #    rospy.sleep(5)
            #    print "Waiting for point cloud update"
            #    self.vision_movement_publisher.publish(Int8(a_send))

        if req.action == req.ACTION_MOVE:
            start_time = time.time()
            p_o_array = []
            (position_,orientation_) = self.myKinovaMotionExecutor.get_cartesian_goal_from_relative_pose(
            req.move_x, req.move_y, req.move_z)
            if(abs(req.move_x)>0.01 or abs(req.move_y)>0.01):
                pos_index = 0
                move_value = req.move_x
                if(abs(req.move_y)>0.01):
                        pos_index = 1
                        move_value = req.move_y
                for i in range(1,8):                 
                    pos_interim = position_[:]
                    pos_interim[pos_index] = pos_interim[pos_index] - move_value + (i*0.01*abs(move_value)/move_value)
                    p_o_array.append((pos_interim,orientation_))
            
            p_o_array.append((position_,orientation_))
            for (pos,ori) in p_o_array:
                self.myKinovaMotionExecutor.goto_absolute_pose_until_touch(pos,ori,req.check_touch,req.check_vision_movement)      

                if self.myKinovaMotionExecutor.cancelled_execution == 1:
                    break
                self.visionMovementDetector.checked_movement = 0
                while self.visionMovementDetector.checked_movement != 1:
                    rospy.sleep(0.5)    
                rospy.sleep(1)
                if self.visionMovementDetector.object_moved_final == 1:
                    break
            
            #self.myKinovaMotionExecutor.goto_relative_pose_until_touch(req.move_x, req.move_y, req.move_z, 
            #check_touch=req.check_touch, check_vision_movement = req.check_vision_movement)
            end_time = time.time()
            print "Kinova motion time: {:.5f}".format(end_time - start_time)
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
            myKinovaMotionExecutor.goto_relative_pose(dz=0.01)
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
        res.finger_joint_state = self.finger

        if req.check_vision_movement:
            self.visionMovementDetector.checked_movement = 0
            while self.visionMovementDetector.checked_movement != 1:
                rospy.sleep(0.5)
            rospy.sleep(1)
            self.visionMovementDetector.start = -1
            """
            a_send =-1
            self.vision_movement_publisher.publish(Int8(a_send))
            #dummy = -1
            #rospy.Subscriber('object_vision_movement', Int8, has_object_moved_callback)
            while self.visionMovementDetector.start !=a_send :
                rospy.sleep(5)
                print "Waiting for point cloud detection freeze"
                self.vision_movement_publisher.publish(Int8(a_send))
                #vision_movement_publisher.publish(Int8(a_send))
            """

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
        print "Request handled"
        return res

def mico_action_feedback_server(h):
    
    rospy.Service('mico_action_feedback_server', MicoActionFeedback, h.handle_action_request)
    print "Mico Action Feedback Server Ready"
    rospy.spin()


class VisionMovementDetector(object):
    def __init__(self, motionExector):
        self.start = -1
        self.object_moved = 3
        self.checked_movement = 0
        self.object_moved_final = -1;
        #self.not_processing = True
        self.pub_vision_movement = rospy.Publisher('object_vision_movement', Int8, queue_size=10)
        self.pub_point_cloud1 = rospy.Publisher('movement_detector/initial_point_cloud', PointCloud2, queue_size=10)
        self.pub_point_cloud2 = rospy.Publisher('movement_detector/current_point_cloud', PointCloud2, queue_size=10)
        
        self.motion_executor = motionExector
        
        self.pointCloudProcessor = giob.GetInitialObjectBelief(None,False,False,'real')
        self.T_cam_world = self.pointCloudProcessor.sensor.get_T_cam_world(
        self.pointCloudProcessor.CAM_FRAME, self.pointCloudProcessor.WORLD_FRAME, 
        self.pointCloudProcessor.config_path)
        self.cam_intrinsic = self.pointCloudProcessor.sensor.get_cam_intrinsic()
        self.depth_image_topic = self.pointCloudProcessor.config['kinect_sensor_cfg']['depth_topic']
        self.point_cloud_topic = self.pointCloudProcessor.config['kinect_sensor_cfg']['cam_point_cloud']
        self.point_cloud_1 = self.update_initial_point_cloud()
        #rospy.Subscriber(self.depth_image_topic, Image, self.p_callback)
        rospy.Subscriber(self.point_cloud_topic, PointCloud2, self.p_callback)
        
    def get_current_point_cloud_from_image(self,depth_im = None):
        (self.cam_intrinsic,point_cloud_world,self.T_cam_world) = self.pointCloudProcessor.get_world_point_cloud(depth_im,self.cam_intrinsic,self.T_cam_world)
        return self.get_current_point_cloud(point_cloud_world)
    
    def get_raw_point_cloud(self, point_cloud_world):
        pc2 = PointCloud2()
        pc2.header.frame_id = self.pointCloudProcessor.WORLD_FRAME
        segmented_pc = pcl2.create_cloud_xyz32(pc2.header, np.transpose(point_cloud_world.data))
        return segmented_pc
    
    def process_raw_point_cloud(self, point_cloud_cam_raw):
        points = pcl2.read_points(point_cloud_cam_raw, field_names=('x','y','z'), skip_nans=True)
        point_cloud_raw_points = [p for p in points]
        point_cloud_cam = PointCloud(np.transpose(np.array(point_cloud_raw_points)),
        self.pointCloudProcessor.CAM_FRAME)
        
        point_cloud_world = self.T_cam_world * point_cloud_cam
        return point_cloud_world
    def get_current_point_cloud(self, point_cloud_world = None):
        if point_cloud_world is None:
            point_cloud_cam_raw = rospy.wait_for_message(self.point_cloud_topic, PointCloud2)
            point_cloud_world = self.process_raw_point_cloud(point_cloud_cam_raw)
        
        min_x = self.motion_executor.curr_pose.pose.position.x
        min_z = self.motion_executor.curr_pose.pose.position.z
        
        cfg = copy.deepcopy(self.pointCloudProcessor.detector_cfg )
        cfg['min_pt'][2] = cfg['min_z_for_movement']
        if min_z is not None:
            cfg['min_pt'][2] = min_z + 0.01
        if min_x > cfg['min_pt'][0]:
            cfg['min_pt'][0] = min_x -0.05
        cfg['max_pt'][2] = cfg['min_pt'][2] + 0.1
        seg_point_cloud_world = self.pointCloudProcessor.get_segmented_point_cloud_world(cfg, point_cloud_world )
        
        #point_cloud_1 = get_current_point_cloud_for_movement(min_x + 0.04,False, False, 'real', min_z)
        return seg_point_cloud_world
    
    def check_vision_movement(self,depth_im):
        if self.start == 0:
            self.object_moved = 3
            #get point cloud 2
            if type(depth_im) is Image:
                point_cloud_2 = self.get_current_point_cloud_from_image(
                self.pointCloudProcessor.sensor.process_raw_depth_image(depth_im))
            else: #It is point cloud 2
                point_cloud_2 = self.get_current_point_cloud(
                self.process_raw_point_cloud(depth_im))
            if point_cloud_2.num_points > 1:
                self.pub_point_cloud2.publish(self.get_raw_point_cloud(point_cloud_2))
            else:
                print "Not publishing current point cloud with "+ repr(point_cloud_2.num_points) + " points"
            #check has object moved
            self.object_moved = giob.has_object_moved(self.point_cloud_1, point_cloud_2)
            self.object_moved_final = self.object_moved
            print self.object_moved
            self.pub_vision_movement.publish(self.object_moved)
    
    
    def update_initial_point_cloud(self):
        start_time = time.time()
        self.object_moved_final = -1;
        #self.point_cloud_1 = self.get_current_point_cloud_from_image()
        self.point_cloud_1 = self.get_current_point_cloud()
        end_time = time.time()
        if self.point_cloud_1.num_points > 1:
            self.pub_point_cloud1.publish(self.get_raw_point_cloud(self.point_cloud_1))
        else:
            print "Not publishing initial point cloud with "+ repr(self.point_cloud_1.num_points) + " points"
        print "point cloud fetch time: {:.5f}".format(end_time - start_time)
        
    def s_callback(self, msg):
        #self.start = msg.data
        if msg.data == 1 and self.start < 0:
            self.start = 1
            self.update_initial_point_cloud()
            
            self.start = 0
        if msg.data < 0:
            self.start = msg.data
            
        
    def p_callback(self, msg):
        self.checked_movement = 1
        start_time = time.time()
        self.check_vision_movement(msg)
        end_time = time.time()
        if self.start == 0:
            
            print "Movement checking time: {:.5f}".format(end_time - start_time)
        
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
    #rospy.Subscriber('/start_movement_detection', Int8, d.s_callback)
    
    h = MicoActionRequestHandler(myKinovaMotionExecutor, d)
    
    mico_action_feedback_server(h)
