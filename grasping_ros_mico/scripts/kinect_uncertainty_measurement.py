#!/usr/bin/env python

import rospy
from roslib import message
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped

import cv2
from cv_bridge import CvBridge, CvBridgeError

import actionlib
import action_controller.msg
import copy

from mico_motion_planner.srv import *
from mico_motion_planner.msg import *
# import motion_executor
import kinova_motion_executor

import numpy as np
import tf
import math

# TODO: move these settings to a YAML File

# image processing
IMAGE_SCALE_WIDTH = 1280
IMAGE_CROP_X = int(0.3*1280)
# IMAGE_CROP_Y = int(1.8/5.6 * 720)
# IMAGE_CROP_Y = int(5.5/14.5 * 720)
# IMAGE_CROP_Y = int(4.5/13.0 * 720)
IMAGE_CROP_Y = int(2.15/6.2 * 720)


# grasping settings (manual fouud using joystick)
BACK_HEIGHT_SEPERATION_DISTANCE = 0.2 #0.25
TOP_HEIGHT_SEPERATION_DISTANCE = 0.2 #0.25
BACK_PULL_BACK_SEPERATION_DISTANCE = 0.115 #0.105
MOVE_PULL_BACK_SEPERATION_DISTANCE = 0.0
BACK_LOWEST_HEIGHT = 0.055
BACK_MOVE_FORWARD_DIST_FACTOR = 1.15
DOWN_MOVE_FORWARD_DIST_FACTOR = 0.9
BACK_LIFT_HEIGHT = 0.32

DOWN_LOWEST_HEIGHT = 0.0612045116723 # 0.0585791772604 
TOO_WIDE_WIDTH = 0.36
TOO_SHORT_HEIGHT = 0.03

# MICO home pose
HOME_POSE_MICO_POS_X =  0.209936052561
HOME_POSE_MICO_POS_Y = -0.262059390545
HOME_POSE_MICO_POS_Z =  0.478999018669
HOME_POSE_MICO_QUAT_X =  0.582023030814
HOME_POSE_MICO_QUAT_Y =  0.394831487846
HOME_POSE_MICO_QUAT_Z =  0.371571068021
HOME_POSE_MICO_QUAT_W =  0.606046391968

# home pose left
HOME_POSE_LEFT_POS_X =  0.268806250345
HOME_POSE_LEFT_POS_Y = -0.00273905972806
HOME_POSE_LEFT_POS_Z =  0.575176112928
HOME_POSE_LEFT_QUAT_X =  0.724436107675
HOME_POSE_LEFT_QUAT_Y =  0.145608512853
HOME_POSE_LEFT_QUAT_Z = -0.657688133129
HOME_POSE_LEFT_QUAT_W =  0.1464131361

# home pose back
HOME_POSE_BACK_POS_X = 0.0927762687206
HOME_POSE_BACK_POS_Y = -0.296744674444
HOME_POSE_BACK_POS_Z =  0.758062064648
HOME_POSE_BACK_QUAT_X =  0.670519902524
HOME_POSE_BACK_QUAT_Y =  0.0235298384606
HOME_POSE_BACK_QUAT_Z =  0.025911820423
HOME_POSE_BACK_QUAT_W =  0.7410654388

# top down gripper orientation
LEFT_TOP_DOWN_QUAT_X = -0.0242462968472
LEFT_TOP_DOWN_QUAT_Y = -0.0322296567231
LEFT_TOP_DOWN_QUAT_Z = -0.71769778686
LEFT_TOP_DOWN_QUAT_W = 0.695185768736

# back facing forward orientation
# BACK_FORWARD_QUAT_X =  0.70737982898
# BACK_FORWARD_QUAT_Y =  0.0371869976821
# BACK_FORWARD_QUAT_Z =  0.0410465017534
# BACK_FORWARD_QUAT_W =  0.704660265269

BACK_FORWARD_QUAT_X =  0.695757364137
BACK_FORWARD_QUAT_Y =  0.0584710976353
BACK_FORWARD_QUAT_Z =  0.00839321429482
BACK_FORWARD_QUAT_W =  0.715843820218

# back facing down orientation
# BACK_DOWN_QUAT_X = 0.999104566208
# BACK_DOWN_QUAT_Y = 0.0206581492597
# BACK_DOWN_QUAT_Z = -0.0303991586712
# BACK_DOWN_QUAT_W =  0.0209570466208

BACK_DOWN_QUAT_X = 0.999411067904
BACK_DOWN_QUAT_Y = 0.027348974755
BACK_DOWN_QUAT_Z = -0.0152068862875
BACK_DOWN_QUAT_W =  0.0140819580015

# demo oddities
PASS_POSE_POS_X =  -0.56769067049
PASS_POSE_POS_Y = -0.123595871031
PASS_POSE_POS_Z =  0.436970114708
PASS_POSE_QUAT_X =  0.574892834969
PASS_POSE_QUAT_Y = -0.349952696665
PASS_POSE_QUAT_Z =  -0.414082800807
PASS_POSE_QUAT_W =  0.612835028758

# wave pre position
WAVE_POSE_POS_X =  -0.123481795192
WAVE_POSE_POS_Y = -0.282588601112
WAVE_POSE_POS_Z =  0.478843092918
WAVE_POSE_QUAT_X =  0.702484313616
WAVE_POSE_QUAT_Y =  0.0364675949698
WAVE_POSE_QUAT_Z =  0.00429078336456
WAVE_POSE_QUAT_W =  0.710751357944

# wave rot left
WAVE_ROT_LEFT_QUAT_X = 0.679432435016
WAVE_ROT_LEFT_QUAT_Y = -0.182172056723
WAVE_ROT_LEFT_QUAT_Z = 0.223589198931
WAVE_ROT_LEFT_QUAT_W = 0.674679759677

# wave rot right
WAVE_ROT_RIGHT_QUAT_X = 0.673535122315
WAVE_ROT_RIGHT_QUAT_Y = 0.202896045628
WAVE_ROT_RIGHT_QUAT_Z = -0.165262595501
WAVE_ROT_RIGHT_QUAT_W = 0.691282799009

# DESCRIBE_NUM_OBJECTS = 4
DESCRIBE_CONFIDENCE_THRESHOLD = 2.0
RESULT_CONFIDENCE_THRESHOLD = 0.3

# roslaunch rosbridge_server rosbridge_websocket.launch port:=8443 address:=bigbird.d1.comp.nus.edu.sg


WORLD_FRAME = 'table_frame' #'odom_combined'

class ManipulatorState(object):
    
    idle  = 0
    pre_grasp = 1
    exe_grasp = 2
    searching = 3
    post_grasp = 4

class GraspType(object):
    
    back_forward = 0
    top_down = 1


class Action(object):
    
    def __init__(self, name, key_synonyms, check_type='max_syn_len'):
        self.name = name
        self.key_synonyms = key_synonyms
        self.check_type = check_type
        self.query = ''

        if len(key_synonyms) > 0:
            self._max_syn_len = max([len(s.split()) for s in self.key_synonyms])
        else:
            self._max_syn_len = 0

    def check(self, input_str):

        if len(self.key_synonyms) == 0:
            return False

        if self.check_type == 'max_syn_len':
            words = ' '.join(input_str.split()[:self._max_syn_len])
        else:
            words = input_str

        for syn in self.key_synonyms:
            if syn in words:
                return True
        return False

    def remove_key_str(self, input_str):

        if len(self.key_synonyms) == 0:
            return input_str

        words = ' '.join(input_str.split()[:self._max_syn_len])
        for syn in self.key_synonyms:
            if syn in words:
                return input_str.replace(syn, '').strip()
        return input_str

    def execute(self):
        """
        to be implemented by user
        """
        raise Exception('Action execution not implemented')
        return False


# all actions defined here
# NOTE: all synonyms assumed to be unique (i.e. no collisions with other action synonyms)
class ActionLibrary(object):

    actions = []

    actions.append(Action('wave', ['say hi', 'wave']))
    actions.append(Action('pick_up', ['pick up']))
    actions.append(Action('locate_object', ['locate']))
    actions.append(Action('describe', ['describe']))
    actions.append(Action('move_to', ['move to', 'put it', 'move them', 'put them', 'move it']))
    actions.append(Action('yes', ['yes', 'yep', 'yeah', 'e. s.', 'i s.', 'go ahead'], check_type='full'))
    actions.append(Action('no', ['no', 'nope', 'wrong'], check_type='full'))
    actions.append(Action('reset', ['abort', 'go home', 'reset']))
    actions.append(Action('open', ['open', 'let go', 'let it go', 'release']))
    actions.append(Action('close', ['close', 'grab', 'hold']))
    actions.append(Action('nothing', ['idle', 'you are done', 'that is it', 'done', 'thats it']))
    actions.append(Action('pass', ['pass', 'it to me']))

    # demo oddities
    actions.append(Action('profanities', ["you suck", "you're an idiot"]))
    actions.append(Action('master', ["who is your master", "who created you", "who is your god"]))


class ActionInterface():

    def parse(self, input_str):
        input_str = self.remove_to(input_str)

        actions = []
        action_strs = input_str.split('and')

        # parse with delimeter 'and'
        for action_str in action_strs:
            for action in ActionLibrary.actions:
                if action.check(action_str):
                    action.query = action_str
                    actions.append(action)

        # couldn't understand any commands
        if len(actions) == 0:
            print "!!! Action Parsing failed !!!"
            actions.append(None)

        return actions

    def retrieve_action(self, name):
        for action in ActionLibrary.actions:
            if action.name == name:
                return action
        print "!!! Invalid Action name !!!"
        return None

    def get_all_actions(self):
        return ActionLibrary.actions

    def remove_to(self, input_str):
        first_word = input_str.split()[0]
        if first_word == 'to':
            return input_str.replace('to', '', 1)

        return input_str


class RefexpManager:

    def __init__(self, image_topic, query_topic, seg_srv):   

        self.initialize_action_interface()

        # setup kinect image and query subscriber 
        self._img_msg = None
        self._img_header = None
        self._rgb_sub = rospy.Subscriber(image_topic, Image, self.image_cb)
        if query_topic is not None:
            self._query_sub = rospy.Subscriber(query_topic, String, self.query_cb)  
            self._abort_sub = rospy.Subscriber('/search_abort', String, self.abort_cb)

        rospy.wait_for_service(seg_srv)
        print "Segmentor Ready!"
        self._segmentor_srv = rospy.ServiceProxy(seg_srv, BBoxSegmentation)

        self._top_sub = rospy.Publisher('refexp_result_top_cluster', PointCloud2, queue_size=1)
        self._context_sub = rospy.Publisher('refexp_result_context_clusters', PointCloud2, queue_size=1)
        self._top_centroid_pub = rospy.Publisher('refexp_result_top_centroid', PointStamped, queue_size=1)
        self._manipulator_target_pub = rospy.Publisher('refexp_manipulator_target', PoseStamped, queue_size=1)
        self._search_result_pub = rospy.Publisher('dense_refexp_result', Image, queue_size=1, latch=True)

        self._kinova_motion_controller = None #kinova_motion_executor.KinovaExecutor()
        if 'recognition' in query_topic:
            import polly
            self._text2speech = polly.Polly()
        else:
            self._text2speech = None
            
        self._tl = tf.TransformListener()

        self._manipulator_state = ManipulatorState.idle
        self._query_str = ''
        self._abort_search = False
        self._move_to_state = False
        self._grasp_type = GraspType.back_forward

        self._all_centroids_point_stamped = []
        self._all_obj_widths_heights = []
        print "Init done"
        # debugging
        # print "Kinova Current Pose:"
        # print self.get_curr_pose()

    def initialize_action_interface(self):

        self._action_interface = ActionInterface()
        self._action_interface.retrieve_action('locate_object').execute = self.exe_locate_object
        self._action_interface.retrieve_action('pick_up').execute = self.exe_pick_up
        self._action_interface.retrieve_action('open').execute = self.exe_open
        self._action_interface.retrieve_action('close').execute = self.exe_close
        self._action_interface.retrieve_action('reset').execute = self.exe_reset
        self._action_interface.retrieve_action('yes').execute = self.exe_yes
        self._action_interface.retrieve_action('no').execute = self.exe_no
        self._action_interface.retrieve_action('move_to').execute = self.exe_move_to
        self._action_interface.retrieve_action('nothing').execute = self.exe_nothing
        self._action_interface.retrieve_action('wave').execute = self.exe_wave
        self._action_interface.retrieve_action('pass').execute = self.exe_pass
        self._action_interface.retrieve_action('describe').execute = self.exe_describe

        # demo oddities
        self._action_interface.retrieve_action('profanities').execute = self.exe_profanities
        self._action_interface.retrieve_action('master').execute = self.exe_master


    def image_cb(self, msg):
        #print "imagecb start"
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            self._img_header = msg.header

            self._img_ratio = (1.*cv2_img.shape[0]) / cv2_img.shape[1]
            img_height = int(self._img_ratio*IMAGE_SCALE_WIDTH)
            self._cv2_img = cv2.resize(cv2_img, (IMAGE_SCALE_WIDTH, img_height))
            self._x_new2orig_ratio = (1.*cv2_img.shape[0]) / self._cv2_img.shape[0]
            self._y_new2orig_ratio = (1.*cv2_img.shape[1]) / self._cv2_img.shape[1]
            #print "imagecb mid"
            self._cv2_img = self._cv2_img[IMAGE_CROP_Y:img_height-1, IMAGE_CROP_X:IMAGE_SCALE_WIDTH-1]
            #print "imagecb mid1"
            self._img_msg = CvBridge().cv2_to_imgmsg(self._cv2_img)
            #print "imagecb mid2"
            #print "image cb"
        except CvBridgeError, e:
            print(e)


    def tell_user(self, text):
        if self._text2speech is not None:
            self._text2speech.speak(text)
        else:
            print text

    def get_curr_pose(self):

        # curr_pose = self._motion_controller.get_curr_pose()
        curr_pose = self._kinova_motion_controller.get_current_pose()
        while not rospy.is_shutdown():
            try:
                time = self._tl.getLatestCommonTime(WORLD_FRAME, curr_pose.header.frame_id)
                curr_pose.header.stamp = time
                curr_pose = self._tl.transformPose(WORLD_FRAME, curr_pose)
                # print "Current Pose available"
                break
            except:
                print "Waiting for tf between %s and odom_combined ..."%(curr_pose.header.frame_id)
                continue

        return curr_pose


    def goto_home_pose(self):

        home_pose = PoseStamped()
        home_pose.header.stamp = rospy.Time.now()
        home_pose.header.frame_id = WORLD_FRAME
        home_pose.pose.position.x = HOME_POSE_MICO_POS_X
        home_pose.pose.position.y = HOME_POSE_MICO_POS_Y
        home_pose.pose.position.z = HOME_POSE_MICO_POS_Z
        home_pose.pose.orientation.x = HOME_POSE_MICO_QUAT_X
        home_pose.pose.orientation.y = HOME_POSE_MICO_QUAT_Y
        home_pose.pose.orientation.z = HOME_POSE_MICO_QUAT_Z
        home_pose.pose.orientation.w = HOME_POSE_MICO_QUAT_W        

        self._kinova_motion_controller.goto_global_pose(home_pose)
        self._kinova_motion_controller.set_gripper_state('open')

    def exe_wave(self):

        if self._manipulator_state == ManipulatorState.idle:

            wave_pose = PoseStamped()
            wave_pose.header.stamp = rospy.Time.now()
            wave_pose.header.frame_id = WORLD_FRAME
            wave_pose.pose.position.x = WAVE_POSE_POS_X
            wave_pose.pose.position.y = WAVE_POSE_POS_Y
            wave_pose.pose.position.z = WAVE_POSE_POS_Z
            wave_pose.pose.orientation.x = WAVE_POSE_QUAT_X
            wave_pose.pose.orientation.y = WAVE_POSE_QUAT_Y
            wave_pose.pose.orientation.z = WAVE_POSE_QUAT_Z
            wave_pose.pose.orientation.w = WAVE_POSE_QUAT_W        

            self._kinova_motion_controller.goto_global_pose(wave_pose)
            self._kinova_motion_controller.set_gripper_state('open')

            # self.goto_home_pose()            
            self.tell_user("Hi there! My name is Mico")

            wave_count = 2
            for c in range(wave_count):

                # rotate left
                wave_pose.header.stamp = rospy.Time.now()
                wave_pose.pose.orientation.x = WAVE_ROT_LEFT_QUAT_X
                wave_pose.pose.orientation.y = WAVE_ROT_LEFT_QUAT_Y
                wave_pose.pose.orientation.z = WAVE_ROT_LEFT_QUAT_Z
                wave_pose.pose.orientation.w = WAVE_ROT_LEFT_QUAT_W    
                self._kinova_motion_controller.goto_global_pose(wave_pose)

                wave_pose.header.stamp = rospy.Time.now()
                wave_pose.pose.orientation.x = WAVE_ROT_RIGHT_QUAT_X
                wave_pose.pose.orientation.y = WAVE_ROT_RIGHT_QUAT_Y
                wave_pose.pose.orientation.z = WAVE_ROT_RIGHT_QUAT_Z
                wave_pose.pose.orientation.w = WAVE_ROT_RIGHT_QUAT_W    
                self._kinova_motion_controller.goto_global_pose(wave_pose)


            self.goto_home_pose()

    def exe_pass(self):

        restore_pose = self._kinova_motion_controller.get_current_pose()

        if self._manipulator_state == ManipulatorState.post_grasp:

            pass_pose = PoseStamped()
            pass_pose.header.stamp = rospy.Time.now()
            pass_pose.header.frame_id = WORLD_FRAME
            pass_pose.pose.position.x = PASS_POSE_POS_X
            pass_pose.pose.position.y = PASS_POSE_POS_Y
            pass_pose.pose.position.z = PASS_POSE_POS_Z
            pass_pose.pose.orientation.x = PASS_POSE_QUAT_X
            pass_pose.pose.orientation.y = PASS_POSE_QUAT_Y
            pass_pose.pose.orientation.z = PASS_POSE_QUAT_Z
            pass_pose.pose.orientation.w = PASS_POSE_QUAT_W        

            self._kinova_motion_controller.goto_global_pose(pass_pose)
            self.tell_user("Ok here you go")
            self._kinova_motion_controller.set_gripper_state('open')
            self._manipulator_state = ManipulatorState.idle

            # go back to restore pose
            restore_pose.header.stamp = rospy.Time.now()
            restore_pose.header.frame_id = WORLD_FRAME
            self._kinova_motion_controller.goto_global_pose(restore_pose)

        else:
            print 'Cannot pass when not in pre-grasp state'
            return False

        return True

    def get_centroid_pose(self, centroid):
         # curr_pose = self._motion_controller.get_curr_pose()
        centroid_pose = PoseStamped()
        centroid_pose.header = self._seg_header
        centroid_pose.pose.position.x = centroid.point.x
        centroid_pose.pose.position.y = centroid.point.y
        centroid_pose.pose.position.z = centroid.point.z
        centroid_pose.pose.orientation.x = 0.0
        centroid_pose.pose.orientation.y = 0.0
        centroid_pose.pose.orientation.z = 0.0
        centroid_pose.pose.orientation.w = 1.0

        while not rospy.is_shutdown():
            try:
                time = self._tl.getLatestCommonTime(self._seg_header.frame_id, WORLD_FRAME)
                centroid_pose.header.stamp = time
                centroid_pose = self._tl.transformPose(WORLD_FRAME, centroid_pose)
                # print "Ready to compute pre-grasp pose"
                break
            except:
                print "Waiting for tf between %s and odom_combined ..."%(self._seg_header.frame_id)
                continue
        return centroid_pose
    
    def compute_pre_grasp_pose(self, centroid, grasp_type=GraspType.back_forward):

        centroid_pose = self.get_centroid_pose(centroid)

        # apply offset
        pose = PoseStamped()
        pose.header.stamp = time
        pose.header.frame_id = WORLD_FRAME
        pose.pose.position.x = centroid_pose.pose.position.x

        if grasp_type == GraspType.back_forward:
            pose.pose.position.y = centroid_pose.pose.position.y + (BACK_PULL_BACK_SEPERATION_DISTANCE if not self._move_to_state else MOVE_PULL_BACK_SEPERATION_DISTANCE) 
            pose.pose.position.z = centroid_pose.pose.position.z + BACK_HEIGHT_SEPERATION_DISTANCE
            pose.pose.orientation.x = BACK_FORWARD_QUAT_X
            pose.pose.orientation.y = BACK_FORWARD_QUAT_Y
            pose.pose.orientation.z = BACK_FORWARD_QUAT_Z
            pose.pose.orientation.w = BACK_FORWARD_QUAT_W
        else:
            pose.pose.position.y = centroid_pose.pose.position.y
            pose.pose.position.z = centroid_pose.pose.position.z + TOP_HEIGHT_SEPERATION_DISTANCE
            pose.pose.orientation.x = BACK_DOWN_QUAT_X
            pose.pose.orientation.y = BACK_DOWN_QUAT_Y
            pose.pose.orientation.z = BACK_DOWN_QUAT_Z
            pose.pose.orientation.w = BACK_DOWN_QUAT_W

        self._manipulator_target_pub.publish(pose)

        return pose

    def transform_point(self, point):

        while not rospy.is_shutdown():
            try:
                time = self._tl.getLatestCommonTime(point.header.frame_id, WORLD_FRAME)
                point.header.stamp = time
                point = self._tl.transformPoint(WORLD_FRAME, point)
                break
            except:
                print "Waiting for odom_combined to point transforation"
                continue

        return point


    def densecap_load(self):

        if self._abort_search:
            return [-1], [-1], [-1] 

        # load latest image
        client = actionlib.SimpleActionClient('dense_refexp_load', action_controller.msg.DenseRefexpLoadAction)
        client.wait_for_server()
        goal = action_controller.msg.DenseRefexpLoadGoal(self._img_msg)
        client.send_goal(goal)
        client.wait_for_result()  
        load_result = client.get_result()

        boxes = np.reshape(load_result.boxes, (-1, 4))   
        captions = np.array(load_result.captions)
        losses = np.array(load_result.scores)

        return boxes, captions, losses


    def refexp_query(self, boxes, losses):
        """
        NOTE: should be called only after densecap_load
        """
        if self._abort_search:
            return [-1]

        incorrect_idxs = []
        client = actionlib.SimpleActionClient('dense_refexp_query', action_controller.msg.DenseRefexpQueryAction)
        client.wait_for_server()  
        goal = action_controller.msg.DenseRefexpQueryGoal(self._query_str, incorrect_idxs)
        client.send_goal(goal)
        client.wait_for_result()
        query_result = client.get_result()

        top_idx = query_result.top_box_idx
        context_boxes_idxs = list(query_result.context_boxes_idxs)

        # prune by confidence
        pruned_idxs = [i for i, orig_idx in enumerate(context_boxes_idxs) if losses[orig_idx] > RESULT_CONFIDENCE_THRESHOLD]
        context_boxes_idxs = list(np.take(context_boxes_idxs, pruned_idxs))

        # add top box at the last
        context_boxes_idxs.append(top_idx)

        # debug publish results as image
        draw_img = self._cv2_img.copy()
        for (count, idx) in enumerate(context_boxes_idxs):

            x1 = int(boxes[idx][0])
            y1 = int(boxes[idx][1])
            x2 = int(boxes[idx][0] + boxes[idx][2])
            y2 = int(boxes[idx][1] + boxes[idx][3])

            if count == len(context_boxes_idxs)-1:
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0,255,0), 15)
            else:
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0,0,255), 11)

        result_img = CvBridge().cv2_to_imgmsg(draw_img)
        self._search_result_pub.publish(result_img)

        return context_boxes_idxs


    def segment_clusters(self, context_boxes_idxs, boxes):

        if self._abort_search:
            return

        top_cluster = PointCloud2()
        top_centroid = Point()
        self._context_centroids = []
        self._top_centroid = [0.,0.,0.]
        
        self._seg_header = copy.copy(self._img_header)
        self._all_centroids_point_stamped = []
        self._all_obj_widths_heights = []

        for (count, idx) in enumerate(context_boxes_idxs):

            # segment-out cluster
            try:
                req = BBoxSegmentationRequest()

                req.x = (boxes[idx][0] + IMAGE_CROP_X) * self._x_new2orig_ratio
                req.y = (boxes[idx][1] + IMAGE_CROP_Y) * self._y_new2orig_ratio
                req.width = boxes[idx][2] * self._x_new2orig_ratio
                req.height = boxes[idx][3] * self._y_new2orig_ratio

                resp = self._segmentor_srv(req)
            except rospy.ServiceException, e:
                print "Segmentation Service Failed: %s"%e
                return

            x1 = int(boxes[idx][0])
            y1 = int(boxes[idx][1])
            x2 = int(boxes[idx][0]+boxes[idx][2])
            y2 = int(boxes[idx][1]+boxes[idx][3])

            if count == len(context_boxes_idxs)-1: # top box
                top_cluster = resp.cluster
                top_centroid = resp.centroid
                self._top_centroid = [[resp.centroid.x, resp.centroid.y, resp.centroid.z]]
                self._seg_header.frame_id = top_cluster.header.frame_id
            else: # context box
                self._context_centroids.append([resp.centroid.x, resp.centroid.y, resp.centroid.z])

            self._all_obj_widths_heights.append([resp.object_width, resp.object_height])

        # consistent ordering for height_width
        self._all_obj_widths_heights.insert(0, self._all_obj_widths_heights.pop())

        # publish results
        centroid_stamped = PointStamped()
        centroid_stamped.header = self._seg_header
        centroid_stamped.point = top_centroid
        self._context_centroids_pc = pc2.create_cloud_xyz32(self._seg_header, self._context_centroids)

        # store centroids for future queries in prob order
        self._all_centroids_point_stamped.append(centroid_stamped)
        for context_centroid in self._context_centroids:
            context_point_stamped = PointStamped()
            context_point_stamped.header = self._seg_header
            context_point_stamped.point.x = context_centroid[0]
            context_point_stamped.point.y = context_centroid[1]
            context_point_stamped.point.z = context_centroid[2]
            self._all_centroids_point_stamped.append(context_point_stamped)

        self._top_sub.publish(top_cluster)
        self._context_sub.publish(self._context_centroids_pc)
        self._top_centroid_pub.publish(centroid_stamped)


    def is_data_ready(self):

        if self._img_msg == None:
            print "No RGB Image from Kinect, yet"
            return False
        elif not self._query_str:
            print "No query received, yet"
            return False

        return True

    def find_obj(self, home_first=True, compute_grasp_type=True, pose_only=False):

        if not self.is_data_ready():
            return

        self._manipulator_state = ManipulatorState.searching

        # move the arm away for vision
        if home_first:
            self.goto_home_pose()

        # run full pipeline (semantic densecap + spatial refexp + segmentor)
        boxes, captions, losses = self.densecap_load()
        context_boxes_idxs = self.refexp_query(boxes, losses)
        self.segment_clusters(context_boxes_idxs, boxes)

        if not self._abort_search:
            self._context_idx = 0 # top refexp cluster
            if not pose_only:
                self.pre_grasp(self._context_idx, compute_grasp_type=compute_grasp_type)
                self.tell_user("do you mean this one?")
            else:
                centroid_pose = self.get_centroid_pose(self._all_centroids_point_stamped[self._context_idx])
                print centroid_pose
                self._manipulator_target_pub.publish(centroid_pose)
                with open('temp_centroid_file', 'w') as f:
                    f.write(repr(centroid_pose.pose.position.x) + " " + repr(centroid_pose.pose.position.y))
        else:
            print "Search Aborted!"
            self._abort_search = False
            self._manipulator_state = ManipulatorState.idle

    def exe_describe(self):

        if self._manipulator_state == ManipulatorState.idle:

            if not self.is_data_ready():
                return False

            self.tell_user("Alright, let me a have look")

            self.goto_home_pose()
            self._kinova_motion_controller.set_gripper_state('close')

            boxes, captions, losses = self.densecap_load()
            top_obj_idxs = [n for n,v in enumerate(losses) if v > DESCRIBE_CONFIDENCE_THRESHOLD] # assuming boxes are already sorted by score
            top_obj_idxs.append(top_obj_idxs.pop(0)) # formatting consistensy (argh!) first object is always last 

            top_obj_captions = captions[:len(top_obj_idxs)]
            self.segment_clusters(top_obj_idxs, boxes)

            for i in range(len(top_obj_idxs)):
                self.pre_grasp(i)
                print "\tCaption: %s" % top_obj_captions[i]
                self.tell_user(top_obj_captions[i])

            self.goto_home_pose()

            self._manipulator_state = ManipulatorState.idle

        else:
            print "Describe can be only used in idle state"
            return False

        return True


    def back_forward_grasp(self):

        # compute target z position
        curr_pose = self.get_curr_pose()
        target_point = self.transform_point(self._all_centroids_point_stamped[0]) 
        delta_z = target_point.point.z - curr_pose.pose.position.z 
        delta_y = target_point.point.y - curr_pose.pose.position.y

        # vertical table collision check:
        if target_point.point.z < BACK_LOWEST_HEIGHT:
            delta_z = BACK_LOWEST_HEIGHT - curr_pose.pose.position.z

        self._kinova_motion_controller.goto_relative_pose(dz=delta_z)
        self._kinova_motion_controller.goto_relative_pose(dy=delta_y*BACK_MOVE_FORWARD_DIST_FACTOR)
        self._kinova_motion_controller.set_gripper_state('close')
        self._kinova_motion_controller.goto_relative_pose(dz=BACK_LIFT_HEIGHT)


    def top_down_grasp(self):

        # compute target z position
        curr_pose = self.get_curr_pose()
        target_point = self.transform_point(self._all_centroids_point_stamped[0]) 
        delta_z = (target_point.point.z - curr_pose.pose.position.z) * DOWN_MOVE_FORWARD_DIST_FACTOR 

        # vertical table collision check:
        if target_point.point.z < DOWN_LOWEST_HEIGHT:
            delta_z = DOWN_LOWEST_HEIGHT - curr_pose.pose.position.z

        self._kinova_motion_controller.goto_relative_pose(dz=delta_z)
        self._kinova_motion_controller.set_gripper_state('close')
        self._kinova_motion_controller.goto_relative_pose(dz=BACK_LIFT_HEIGHT)


    def pre_grasp(self, index, compute_grasp_type=True):
        
        # choose back grasp or top grasp based on obj height and width
        if self._manipulator_state != ManipulatorState.post_grasp and compute_grasp_type:
            width, height = self._all_obj_widths_heights[index]
            # print "Height: %f, Width: %f" % (height, width)
            if width > TOO_WIDE_WIDTH or height < TOO_SHORT_HEIGHT:
                self._grasp_type = GraspType.top_down
            else:
                self._grasp_type = GraspType.back_forward

        self._pre_grasp_pose = self.compute_pre_grasp_pose(self._all_centroids_point_stamped[index], grasp_type=self._grasp_type)
        self._kinova_motion_controller.goto_global_pose(self._pre_grasp_pose)

        self._manipulator_state = ManipulatorState.pre_grasp


    def exe_locate_object(self):
        self._query_str = self._action_interface.retrieve_action('locate_object').remove_key_str(self._query_str)
        self.tell_user("ok! looking for " + self._query_str + ". Hold on!")
        self.find_obj(home_first=False, compute_grasp_type=False, pose_only=True)
        
        
    def exe_pick_up(self):

        if self._manipulator_state == ManipulatorState.idle or self._manipulator_state == ManipulatorState.pre_grasp:
            self._query_str = self._action_interface.retrieve_action('pick_up').remove_key_str(self._query_str)
            self.tell_user("ok! looking for " + self._query_str + ". Hold on!")
            self.find_obj(home_first=True, compute_grasp_type=True)
        else:
            print "\tNot in a state to pick up! Need to be idle or pre grasp"
            self.tell_user("I have something in my hands")
            return False

        return True

    def exe_move_to(self):

        if self._manipulator_state == ManipulatorState.post_grasp:
            self._query_str = self._action_interface.retrieve_action('move_to').remove_key_str(self._query_str)
            self.tell_user("ok! looking for " + self._query_str + ". Hold on!")
            self._move_to_state = True
            self.find_obj(home_first=False, compute_grasp_type=False)
            self._move_to_state = False
            self._manipulator_state = ManipulatorState.post_grasp
        else:
            print "\tNot in a state to move! Need to be in post-grasp state"
            self.tell_user("I don't have anything in my hands")
            return False

        return True


    def exe_open(self):
        
        self._kinova_motion_controller.set_gripper_state('open')

        if self._manipulator_state == ManipulatorState.post_grasp:
            self._manipulator_state = ManipulatorState.idle

        return True


    def exe_close(self):
        self._kinova_motion_controller.set_gripper_state('close')

        if self._manipulator_state == ManipulatorState.pre_grasp or self._manipulator_state == ManipulatorState.idle:
            self._manipulator_state = ManipulatorState.post_grasp


    def exe_yes(self):

        if self._manipulator_state == ManipulatorState.pre_grasp:

            self._manipulator_state = ManipulatorState.exe_grasp
            if self._grasp_type == GraspType.back_forward:
                self.back_forward_grasp()
            else:
                self.top_down_grasp()
            self._manipulator_state = ManipulatorState.post_grasp

        elif self._manipulator_state == ManipulatorState.post_grasp:

            self._kinova_motion_controller.set_gripper_state('open')
            self._manipulator_state = ManipulatorState.idle

        else:
            print "\tNot in pre-grasp state. why did you say yes?"
            return False

        return True


    def exe_no(self):

        if self._manipulator_state == ManipulatorState.pre_grasp:

            self._context_idx += 1

            # failure (out of context objects)
            if self._context_idx > len(self._all_centroids_point_stamped)-1:
                print "Object not found in reduced context space! Need to expand context space"
                self.reset_state()
                return False

            self.pre_grasp(self._context_idx)
            self._manipulator_state = ManipulatorState.pre_grasp

            self.tell_user("do you mean this one?")

        elif self._manipulator_state == ManipulatorState.post_grasp:

            self._context_idx += 1

            # failure (out of context objects)
            if self._context_idx > len(self._all_centroids_point_stamped)-1:
                print "Object not found in reduced context space! Need to expand context space"
                self.reset_state()
                return False

            self.pre_grasp(self._context_idx)
            self._manipulator_state = ManipulatorState.post_grasp

            self.tell_user("do you mean this one?")

        else:
            print "\tNot in pre-grasp state. why did you say no?"
            return False

        return True

    def exe_nothing(self):

        self._manipulator_state = ManipulatorState.idle
        return True

    def exe_reset(self):

        self.reset_state()
        return True

    def abort_cb(self, msg):

        print "RECEIVED ABORT MESSAGE"
        self.tell_user("Aborting!")
        self.reset_state()

    def query_cb(self, msg):

        self._query_str = str(msg.data).lower()

        print "\tAlexa: %s" % (self._query_str) 
        actions = self._action_interface.parse(self._query_str)
        
        for action in actions: 
            if action == None:
                self.tell_user("I'm sorry I don't understand")
            else:
                print "\t\tAction: %s" % (action.name)
                action.execute()


    def reset_state(self):

        if self._manipulator_state == ManipulatorState.searching:
            self._abort_search = True

        if self._all_centroids_point_stamped != None and len(self._all_centroids_point_stamped) > 0:
            self._all_centroids_point_stamped[:] = []
            self._all_obj_widths_heights[:] = []
        else:
            self._all_centroids_point_stamped = []
            self._all_obj_widths_heights = []

        self.goto_home_pose()
        
        self._manipulator_state = ManipulatorState.idle

    def exe_profanities(self):

        # self.tell_user("fuck you! you're a shit cunt")
        self.tell_user("I could use some bad words at you. But this is a formal demo, so I won't")

    def exe_master(self):

        if self._manipulator_state == ManipulatorState.idle:

            pass_pose = PoseStamped()
            pass_pose.header.stamp = rospy.Time.now()
            pass_pose.header.frame_id = WORLD_FRAME
            pass_pose.pose.position.x = PASS_POSE_POS_X
            pass_pose.pose.position.y = PASS_POSE_POS_Y
            pass_pose.pose.position.z = PASS_POSE_POS_Z
            pass_pose.pose.orientation.x = PASS_POSE_QUAT_X
            pass_pose.pose.orientation.y = PASS_POSE_QUAT_Y
            pass_pose.pose.orientation.z = PASS_POSE_QUAT_Z
            pass_pose.pose.orientation.w = PASS_POSE_QUAT_W        

            self._kinova_motion_controller.goto_global_pose(pass_pose)
            self._kinova_motion_controller.set_gripper_state('close')
            self.tell_user("The one and only, the great Mohit.")
            self._manipulator_state = ManipulatorState.idle

        else:
            print 'Cannot exe master when not in idle state'
            return False


if __name__ == '__main__':
    try:
        rospy.init_node('kinect_object_locater')
        refexp_manager = RefexpManager('/kinect2/hd/image_color_rect', '/speech_command', '/bbox_segmentor')    
        rospy.spin()
    except rospy.ROSInterruptException:
        pass