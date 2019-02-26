import rospy
import getopt
import sys
import os

import perception as perception
from perception import PointToPlaneICPSolver, PointToPlaneFeatureMatcher
import rospkg
from autolab_core import YamlConfig
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tf
from autolab_core import Box, PointCloud, NormalCloud
import signal
from autolab_core import RigidTransform
import numpy as np
import copy
from gqcnn import Visualizer as vis
import logging
from grasping_object_list import get_grasping_object_name_list
import socket
import time
import hashlib


class KinectSensor:
    def __init__(self, COLOR_TOPIC,DEPTH_TOPIC, CAMINFO_TOPIC, start_node = True,env = 'simulator'):
        print CAMINFO_TOPIC
        self.rgb = None
        self.depth = None
        self.cam_info = None
        self.bridge = CvBridge()
        self.freeze = False
        self.COLOR_TOPIC = COLOR_TOPIC
        self.DEPTH_TOPIC = DEPTH_TOPIC
        self.CAMINFO_TOPIC = CAMINFO_TOPIC
        self.env = env
        #rospy.init_node('kinect_listener', disable_signals=True)
        if(start_node):
            rospy.init_node('kinect_listener', anonymous=True)
        #self.rgb_sub = rospy.Subscriber(COLOR_TOPIC, Image, self.color_callback)
        #self.depth_sub = rospy.Subscriber(DEPTH_TOPIC, Image, self.depth_callback)
        #self.cameinfo_sub = rospy.Subscriber(CAMINFO_TOPIC, CameraInfo, self.caminfo_callback)


    def color_callback(self, data):
        if not self.freeze:
            self.rgb = data

    def depth_callback(self, data):
        if not self.freeze:
            self.depth = data

    def caminfo_callback(self, data):
        if not self.freeze:
            self.cam_info = data

    def get_color_im(self):
        #while self.rgb is None:
        #    print "Waiting for color"
        #    rospy.sleep(2)
        print "Waiting for color"
        self.rgb = rospy.wait_for_message(self.COLOR_TOPIC, Image)
        if self.cam_info is None:
            self.get_cam_intrinsic()
        #raw_color = self.bridge.imgmsg_to_cv2(self.rgb, 'bgr8')
        #print raw_color.shape
        #color_arr = copy.copy(raw_color)
        #color_arr = np.fliplr(np.flipud(copy.copy(raw_color)))
        #color_arr[:,:,[0,2]] = color_arr[:,:,[2,0]] # convert BGR to RGB
        #color_arr[:,:,0] = np.fliplr(color_arr[:,:,0])
        #color_arr[:,:,1] = np.fliplr(color_arr[:,:,1])
        #color_arr[:,:,2] = np.fliplr(color_arr[:,:,2])
        #color_arr[:,:,3] = np.fliplr(color_arr[:,:,3])
        #color_image = perception.ColorImage(color_arr[:,:,:3], self.cam_info.header.frame_id)
        color_image = perception.ColorImage(self.bridge.imgmsg_to_cv2(self.rgb), frame=self.cam_info.header.frame_id)
        color_image.save('test.jpg')
        return color_image

    def get_depth_im(self):
        #while self.depth is None:
        #    print "Waiting for depth"
        #    rospy.sleep(2)
        print "Waiting for depth"
        self.depth = rospy.wait_for_message(self.DEPTH_TOPIC, Image)
        if self.cam_info is None:
            self.get_cam_intrinsic()
        return self.process_raw_depth_image(self.depth, self.cam_info)

    def process_raw_depth_image(self, depth_im, camera_info = None):
        if camera_info is None:
            camera_info = self.cam_info
        raw_depth = self.bridge.imgmsg_to_cv2(depth_im, 'passthrough')
        if self.env=='simulator':
            #depth_arr = copy.copy(raw_depth)
            depth_arr = np.flipud(copy.copy(raw_depth))
            depth_arr = np.fliplr(copy.copy(depth_arr))
        elif 'object_detection' in self.env:
            print 'Going to ' + self.env
            #depth_arr = np.flipud(copy.copy(raw_depth))
            #depth_arr = np.fliplr(copy.copy(depth_arr))
            depth_arr = copy.copy(raw_depth)
        else:
            print 'Going to ' + self.env
            depth_arr = copy.copy(raw_depth)
        depth_arr = depth_arr * 0.001
        depth_image = perception.DepthImage(depth_arr, camera_info.header.frame_id)

        #depth_image = perception.DepthImage((self.bridge.imgmsg_to_cv2(raw_depth, desired_encoding = "passthrough")).astype('float'), frame=self.cam_info.header.frame_id)
        return depth_image

    def get_cam_intrinsic(self):
        #while self.cam_info is None:
        #    print "Waiting for cam_info"
        #    rospy.sleep(2)
        if self.cam_info is None:
            print "Waiting for cam_info"
            self.cam_info = rospy.wait_for_message(self.CAMINFO_TOPIC, CameraInfo)
        raw_camera_info = self.cam_info
        camera_intrinsics = perception.CameraIntrinsics(raw_camera_info.header.frame_id, raw_camera_info.K[0], raw_camera_info.K[4], raw_camera_info.K[2], raw_camera_info.K[5], raw_camera_info.K[1], raw_camera_info.height, raw_camera_info.width)
        return camera_intrinsics

    def freeze(self):
        self.freeze = True

    def free(self):
        self.freeze = False

    def get_T_cam_world(self, from_frame, to_frame, config_path = './'):
        """ get transformation from camera frame to world frame"""

        transform_filename = config_path + "/"+ from_frame + '_' + to_frame + '.tf'
        if os.path.exists(transform_filename):
            return RigidTransform.load(transform_filename)
        time = 0
        trans = None
        qat = None

        self.tl = tf.TransformListener()
        while not rospy.is_shutdown():
            try:
                time = self.tl.getLatestCommonTime(to_frame, from_frame)
                (trans, qat) = self.tl.lookupTransform(to_frame, from_frame, time)
                break
            except (tf.Exception,tf.LookupException,tf.ConnectivityException, tf.ExtrapolationException):
                print 'try again'
                continue
        RT = RigidTransform()

        #print qat
        qat_wxyz = [qat[-1], qat[0], qat[1], qat[2]]

        #rot = RT.rotation_from_quaternion(qat_wxyz)
        #print trans
        #print rot

        #return RigidTransform(rot, trans, from_frame, to_frame)
        ans = RigidTransform(translation=trans, rotation=qat_wxyz, from_frame=from_frame, to_frame=to_frame)
        ans.save(transform_filename)
        return ans
class GetInitialObjectBelief():
    def __init__(self, obj_filenames = None, debug = False, start_node=True, env = 'simulator'):
        COLOR_TOPIC = '/kinect2/sd/image_color_rect'
        DEPTH_TOPIC = '/kinect2/sd/image_depth_rect'
        CAMINFO_TOPIC = '/kinect2/sd/camera_info'
        self.CAM_FRAME = 'kinect2_ir_optical_frame'
        self.WORLD_FRAME = 'world'
        self.MICO_TARGET_FRAME = 'mico_target_frame'
        self.debug = debug
        self.env = env
        rospack = rospkg.RosPack()
        self.grasping_ro_mico_path = rospack.get_path('grasping_ros_mico')
        self.config_path = self.grasping_ro_mico_path + '/config_files/dexnet_config/'

        if env =='simulator':
            self.config = YamlConfig(self.config_path + 'mico_control_node.yaml')
        else:
            self.config = YamlConfig(self.config_path + 'mico_control_node_' + env + '.yaml')



        if self.config['kinect_sensor_cfg']['color_topic']:
            COLOR_TOPIC = self.config['kinect_sensor_cfg']['color_topic']
        if self.config['kinect_sensor_cfg']['depth_topic']:
            DEPTH_TOPIC = self.config['kinect_sensor_cfg']['depth_topic']
        if self.config['kinect_sensor_cfg']['camera_info_topic']:
            CAMINFO_TOPIC = self.config['kinect_sensor_cfg']['camera_info_topic']
        if self.config['kinect_sensor_cfg']['cam_frame']:
            self.CAM_FRAME = self.config['kinect_sensor_cfg']['cam_frame']
        if 'world_frame' in self.config['kinect_sensor_cfg'].keys():
            self.WORLD_FRAME = self.config['kinect_sensor_cfg']['world_frame']

        self.detector_cfg = self.config['detector']
        if obj_filenames is not None:
            self.obj_filenames = obj_filenames
            #Not loading all point clods to save memory
            #self.load_object_point_clouds()
        # create rgbd sensor
        self.sensor = None
        print "Creating sensor"
        rospy.loginfo('Creating RGBD Sensor')
        self.sensor = KinectSensor(COLOR_TOPIC, DEPTH_TOPIC, CAMINFO_TOPIC, start_node ,env)

        rospy.loginfo('Sensor Running')
        print "Sensor running"


    def get_nearest_pick_point(self, min_z,max_z):
        (camera_intr, point_cloud_world, T_camera_world) = self.get_world_point_cloud()
        cfg = copy.deepcopy(self.detector_cfg )
        cfg['min_pt'][2] = min_z
        cfg['max_pt'][2] = max_z
        seg_point_cloud_world = self.get_segmented_point_cloud_world(cfg, point_cloud_world )

        #get the min x value and corresponding y value in segmented point cloud
        #min_index = np.argmin(seg_point_cloud_world.x_coords)

        #need to get all the min points , otherwise can get random locations for cuboidal objects
        pick_point = [None, None, None]
        if(len(seg_point_cloud_world.x_coords)> 0):
            x_coords_min = seg_point_cloud_world.x_coords.min()
            min_indices = np.where(seg_point_cloud_world.x_coords < x_coords_min + 0.0001)
            min_y_coordinate = np.mean(seg_point_cloud_world.y_coords[min_indices])
            min_z_coordinate = np.mean(seg_point_cloud_world.z_coords[min_indices])
            pick_point = [float(x_coords_min), float(min_y_coordinate), float(min_z_coordinate)]
        return pick_point



    def get_camera_instrinsincs_and_t_cam_world(self, camera_intrinsics = None, T_camera_world = None):
        #sensor = self.sensor
        if camera_intrinsics is None:
            camera_intrinsics = self.sensor.get_cam_intrinsic()
        #T_camera_world = RigidTransform.load('data/calib/primesense_overhead/kinect2_to_world.tf')
        if T_camera_world is None:
            T_camera_world = self.sensor.get_T_cam_world(self.CAM_FRAME, self.WORLD_FRAME, self.config_path)
        return (camera_intrinsics, T_camera_world)

    def get_world_point_cloud(self, depth_image = None, camera_intrinsics = None, T_camera_world = None):
        
        (camera_intrinsics,T_camera_world) = self.get_camera_instrinsincs_and_t_cam_world(camera_intrinsics, T_camera_world)

        if depth_image is None:
            depth_image = self.sensor.get_depth_im()
        inpainted_depth_image = depth_image.inpaint(rescale_factor=self.config['inpaint_rescale_factor'])
        #print camera_intrinsics.rosmsg
        depth_im = inpainted_depth_image
        camera_intr = camera_intrinsics


        #print T_camera_world.translation
        #print T_camera_world.rotation

        # project into 3D
        point_cloud_cam = camera_intr.deproject(depth_im)
        point_cloud_world = T_camera_world * point_cloud_cam


        return (camera_intr, point_cloud_world, T_camera_world)

    def get_world_color_image(self, color_image = None, camera_intrinsics = None, T_camera_world = None):
        (camera_intrinsics,T_camera_world) = self.get_camera_instrinsincs_and_t_cam_world(camera_intrinsics, T_camera_world)
        if color_image is None:
            color_image = self.sensor.get_color_im()
        #inpainted_color_image = color_image.inpaint(rescale_factor=self.config['inpaint_rescale_factor'])
        #color_im = inpainted_color_image
        camera_intr = camera_intrinsics
        
        return (color_image, camera_intr)
        
    def get_segmented_point_cloud_world(self, cfg, point_cloud_world ):
        # read params

        min_pt_box = np.array(cfg['min_pt'])
        max_pt_box = np.array(cfg['max_pt'])



        box = Box(min_pt_box, max_pt_box, self.WORLD_FRAME)
        print min_pt_box
        print max_pt_box

        ch, num = point_cloud_world.data.shape

        print num
        A = point_cloud_world.data
        count = 0
        for i in range(num):
            if min_pt_box[0] < A[0][i] < max_pt_box[0] and \
               min_pt_box[1] < A[1][i] < max_pt_box[1] and \
               min_pt_box[2] < A[2][i] < max_pt_box[2]:
                   count += 1

        print count


        seg_point_cloud_world, _ = point_cloud_world.box_mask(box)
        #seg_point_cloud_world = point_cloud_world
        return seg_point_cloud_world

    def get_object_point_cloud_from_sensor(self, cfg = None, segment = True):

        (camera_intr, point_cloud_world, T_camera_world) = self.get_world_point_cloud()

        if cfg is None:
            cfg = self.detector_cfg
        if segment:
            seg_point_cloud_world = self.get_segmented_point_cloud_world(cfg, point_cloud_world )
        else:
            seg_point_cloud_world = point_cloud_world
        seg_point_cloud_cam = T_camera_world.inverse() * seg_point_cloud_world

        #T_camera_target = self.sensor.get_T_cam_world(self.CAM_FRAME, self.MICO_TARGET_FRAME, self.config_path)
        #print T_camera_world
        #print T_camera_target
        #seg_point_cloud_target = T_camera_target * seg_point_cloud_cam

        """
        import sensor_msgs.point_cloud2 as pcl2
        from sensor_msgs.msg import Image, PointCloud2, PointField
        print point_cloud_world.data.shape
        pc2 = PointCloud2()
        pc2.header.frame_id = self.WORLD_FRAME
        segmented_pc = pcl2.create_cloud_xyz32(pc2.header, np.transpose(seg_point_cloud_world.data))
        pcl_pub = rospy.Publisher('mico_node/pointcloud', PointCloud2, queue_size=10)

        while not rospy.is_shutdown():
           #hello_str = "hello world %s" % rospy.get_time()
           #rospy.loginfo(hello_str)
           pcl_pub.publish(segmented_pc)
           #depth_im_pub.publish(depth_im)
           #rospy.sleep(5)

        #return copy.deepcopy(point_cloud_world)
        """
        print seg_point_cloud_cam.shape
        depth_im_seg = camera_intr.project_to_image(seg_point_cloud_cam)
        #if 'real' in self.env:
        #    vis.figure()
        #    vis.subplot(1,1,1)
        #    vis.imshow(depth_im_seg)
        #    vis.show()
        #camera_intr._frame = self.MICO_TARGET_FRAME
        #depth_im_seg = camera_intr.project_to_image(seg_point_cloud_target)
        #camera_intr._frame = self.WORLD_FRAME
        #depth_im_seg = camera_intr.project_to_image(seg_point_cloud_world)
        return(depth_im_seg, camera_intr)


    def save_point_cloud(self, filename_prefix, depth_im_seg, camera_intr):
        if self.debug:
            vis.figure()
            vis.subplot(1,1,1)
            vis.imshow(depth_im_seg)
            vis.show()
        depth_im_seg.save(filename_prefix + '.npy')
        camera_intr.save(filename_prefix  + '.intr')


    def load_saved_point_cloud(self, filename_prefix):
        camera_intr = perception.CameraIntrinsics.load(filename_prefix  + '.intr')
        camera_intr._frame = camera_intr.frame + "_target"
        depth_im = perception.DepthImage.open(filename_prefix + '.npy', frame=camera_intr.frame)

        return (depth_im, camera_intr)

    def get_non_nan_points(self, point_cloud):
        ans = []
        orig_target_normals = point_cloud.data.T
        print orig_target_normals.shape[0]
        for i in range(0,int(orig_target_normals.shape[0])):
            nan_indices = [x for x in orig_target_normals[i] if str(x)=="nan"]
            if len(nan_indices) !=0:
                #pass
                print i
                print nan_indices
            else:
                ans.append(i)
        return ans
    def get_point_normal_cloud(self, depth_im_seg, camera_intr):
        source_point_normal_cloud = depth_im_seg.point_normal_cloud(camera_intr)
        source_point_cloud = source_point_normal_cloud.points
        source_normal_cloud = source_point_normal_cloud.normals
        if self.debug:
            print source_point_cloud.shape
            print source_normal_cloud.shape
        points_of_interest = np.where(source_point_cloud.z_coords != 0.0)[0]
        source_point_cloud._data = source_point_cloud.data[:, points_of_interest]
        source_normal_cloud._data = source_normal_cloud.data[:, points_of_interest]
        if self.debug:
            print source_point_cloud.shape
            print source_normal_cloud.shape
        points_of_interest = self.get_non_nan_points(source_point_cloud)
        source_point_cloud._data = source_point_cloud.data[:, points_of_interest]
        source_normal_cloud._data = source_normal_cloud.data[:, points_of_interest]
        if self.debug:
            print source_point_cloud.shape
            print source_normal_cloud.shape
        points_of_interest = self.get_non_nan_points(source_normal_cloud)
        source_point_cloud._data = source_point_cloud.data[:, points_of_interest]
        source_normal_cloud._data = source_normal_cloud.data[:, points_of_interest]
        if self.debug:
            print source_point_cloud.shape
            print source_normal_cloud.shape
        return (source_point_cloud, source_normal_cloud)



    def register_point_cloud(self):
        registration_result_array = []
        depth_im_seg, camera_intr = self.get_object_point_cloud_from_sensor() #self.database_objects[1]#
        source_point_cloud, source_normal_cloud = self.get_point_normal_cloud(depth_im_seg, camera_intr)
        source_sample_size = int(source_point_cloud.shape[1])
        if self.debug:
            print source_sample_size
            vis.figure()
            vis.subplot(1,1,1)
            vis.imshow(depth_im_seg)
            vis.show()
        source_sample_size = 1000
        p2pis = PointToPlaneICPSolver(sample_size=source_sample_size)
        p2pfm = PointToPlaneFeatureMatcher()
        for objectFileName in self.obj_filenames:
            self.load_object_point_clouds([objectFileName])
            for (target_depth_im, target_camera_intr) in self.database_objects:
                if self.debug:
                    vis.figure()
                    vis.subplot(1,1,1)
                    vis.imshow(target_depth_im)
                    vis.show()
                target_point_cloud, target_normal_cloud = self.get_point_normal_cloud(target_depth_im, target_camera_intr)
                registration_result = p2pis.register( source_point_cloud, target_point_cloud,
                     source_normal_cloud, target_normal_cloud, p2pfm, num_iterations=10)
                registration_result_array.append(registration_result)

        return registration_result_array

    def get_object_probabilities(self):
        registration_result_array = self.register_point_cloud()
        if self.debug:
            for x in registration_result_array:
                print x.cost
                print x.T_source_target
        #For now only looking at cost. Not checking the translation
        probs_unnormalized = [np.exp(1.0/x.cost) for x in registration_result_array]
        z = sum(probs_unnormalized)
        probs = [x/z for x in probs_unnormalized]
        return probs


    def load_object_point_clouds(self, obj_filenames = None):
        if obj_filenames is None:
            obj_filenames = self.obj_filenames
        ans = []
        for filename in obj_filenames:
            print "Loading " + filename
            depth_im, camera_intr = self.load_saved_point_cloud(filename)
            ans.append((depth_im, camera_intr))
            if self.debug:
                vis.figure()
                vis.subplot(1,1,1)
                vis.imshow(depth_im)
                vis.show()
        self.database_objects = ans
        return ans


def get_current_rgb_image(debug = False, start_node=True, env = 'simulator'):
    giob = GetInitialObjectBelief(None, debug, start_node, env)
    (color_im_seg,camera_intr) = giob.get_world_color_image()
    (depth_im_seg,camera_intr) = giob.get_object_point_cloud_from_sensor(None,False)
    if debug:
        vis.figure()
        vis.subplot(1,2,1)
        vis.imshow(depth_im_seg)
        vis.subplot(1,2,2)
        vis.imshow(color_im_seg)
        vis.show()
    return (depth_im_seg, color_im_seg, camera_intr)
        
def save_current_rgb_image(filename_dir, debug = False, start_node=True, env = 'simulator'):
    (depth_im_seg, color_im_seg, camera_intr) = get_current_rgb_image(debug, start_node, env)
    file_name = filename_dir + socket.gethostname()+ "-" + repr(os.getpid()) + "-" + time.strftime("%Y%m%d-%H%M%S")
    file_name_hash =  hashlib.sha1(file_name).hexdigest()
    file_dir = filename_dir + "/" + file_name_hash[0:2] + "/" + file_name_hash[2:4] + "/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    filename = file_dir + file_name_hash
    color_im_seg.save(filename + '.npz')
    depth_im_seg.save(filename + "_depth" + ".npz")
    camera_intr.save(filename  + '.intr')
    return filename
    
def get_current_point_cloud(debug = False, start_node=True, env = 'simulator'):
    giob = GetInitialObjectBelief(None, debug, start_node, env)
    (depth_im_seg,_) = giob.get_object_point_cloud_from_sensor()
    if debug:
        vis.figure()
        vis.subplot(1,1,1)
        vis.imshow(depth_im_seg)
        vis.show()

def get_current_point_cloud_for_movement(min_x, debug = False, start_node=True, env = 'simulator', min_z = None):

    giob = GetInitialObjectBelief(None, debug, start_node,env)
    (camera_intr, point_cloud_world, T_camera_world) = giob.get_world_point_cloud()
    cfg = copy.deepcopy(giob.detector_cfg )
    cfg['min_pt'][2] = cfg['min_z_for_movement']
    if min_z is not None:
        cfg['min_pt'][2] = min_z
    if min_x > cfg['min_pt'][0]:
        cfg['min_pt'][0] = min_x
    seg_point_cloud_world = giob.get_segmented_point_cloud_world(cfg, point_cloud_world )

    """
    import sensor_msgs.point_cloud2 as pcl2
    from sensor_msgs.msg import Image, PointCloud2, PointField
    print point_cloud_world.data.shape
    pc2 = PointCloud2()
    pc2.header.frame_id = giob.WORLD_FRAME
    segmented_pc = pcl2.create_cloud_xyz32(pc2.header, np.transpose(seg_point_cloud_world.data))
    pcl_pub = rospy.Publisher('mico_node/pointcloud', PointCloud2, queue_size=10)

    i = 0
    while not rospy.is_shutdown():
       #hello_str = "hello world %s" % rospy.get_time()
       #rospy.loginfo(hello_str)
       pcl_pub.publish(segmented_pc)
       #depth_im_pub.publish(depth_im)
       rospy.sleep(5)
       i = i+1
       if i > 5:
           break

    #return copy.deepcopy(point_cloud_world)
    """
    if debug:
        seg_point_cloud_cam = T_camera_world.inverse() * seg_point_cloud_world
        depth_im_seg = camera_intr.project_to_image(seg_point_cloud_cam)
        vis.figure()
        vis.subplot(1,1,1)
        vis.imshow(depth_im_seg)
        vis.show()
    return seg_point_cloud_world

def has_object_moved(point_cloud_1, point_cloud_2):
    if(point_cloud_1.num_points > 0 and point_cloud_2.num_points > 0):
        point_cloud_1_mean = point_cloud_1.mean()
        point_cloud_2_mean = point_cloud_2.mean()
        mean_diff = (point_cloud_1_mean - point_cloud_2_mean).vector;
        movement = np.sqrt(np.dot(mean_diff, mean_diff))
        print movement
        return 1 if movement > 0.01 else 0
    else:
        return 2


def load_object_file(obj_file_names, debug = False, start_node=True):
    giob = GetInitialObjectBelief(obj_file_names, debug, start_node)
    return giob.load_object_point_clouds()

def save_object_file(obj_file_name, debug = False, start_node=True):
    giob = GetInitialObjectBelief(None, debug, start_node)
    (d,c) = giob.get_object_point_cloud_from_sensor()
    giob.save_point_cloud(obj_file_name, d, c)

def get_object_name(object_file):
    object_file_parts = object_file.split('_')
    for object_file_part in object_file_parts:
        if 'cm' in object_file_part:
            object_number = int(filter(str.isdigit, object_file_part))
            if object_number < 10:
                return "Cylinder" + repr(object_number) + 'cm'
            elif object_number < 100:
                return "Cylinder" + repr(object_number) + 'mm'
            elif object_number > 1000:
                g3db_objects = get_grasping_object_name_list('all_g3db')
                g3dbPattern = 'G3DB'+repr(object_number - 1000) + '_'
                for g3db_object in g3db_objects:
                    if g3dbPattern in g3db_object:
                        return g3db_object

def get_object_list(object_group_name):
    if type(object_group_name) is str:
        object_list = get_grasping_object_name_list(object_group_name)
        object_list = [x+'.yaml' for x in object_list]
    else:
        #object_list = [get_object_name(x) for x in object_group_name]
        object_list = [x+'.yaml' for x in object_group_name]
    return object_list

def get_object_filenames(object_group_name, object_file_dir):
    object_list = get_object_list(object_group_name)
    obj_filenames = [object_file_dir + "/" + x for x in object_list]
    return obj_filenames

def get_belief_for_objects_old(object_group_name, object_file_dir, debug = False, start_node=True):
    obj_filenames = get_object_filenames(object_group_name, object_file_dir)
    giob = GetInitialObjectBelief(obj_filenames, debug, start_node)
    ans = giob.get_object_probabilities()
    print "<Object Probabilities>" + repr(ans)
    return ans

def get_belief_for_real_objects(env, object_group_name, object_file_dir, clip_objects = -1, keras_model_name = None, baseline_results_dir = None, debug = False, start_node=True):
    print "Getting Belief for real objects"
    return get_belief_for_objects(object_group_name, object_file_dir, clip_objects , keras_model_name , baseline_results_dir, env, debug, start_node)

#object_file_dir = point clod dir when not using keras model
#object_file_dir = keras model dir when using keras model
def get_belief_for_objects(object_group_name, object_file_dir, clip_objects = -1, keras_model_name = None, baseline_results_dir = None, env = 'simulator', debug = False, start_node=True):
    if keras_model_name is None:
        obj_filenames = get_object_filenames(object_group_name, object_file_dir)
        giob = GetInitialObjectBelief(obj_filenames, debug, start_node, env)
        ans = giob.get_object_probabilities()
        print "<Object Probabilities>" + repr(ans)
        return ans
    else:
        #Load keras model
        giob = GetInitialObjectBelief(None, debug, start_node, env)
        (depth_im,cam_intr) = giob.get_object_point_cloud_from_sensor()
        import object_baseline_classifier as obc
        keras_model_dir = object_file_dir + "/keras_model/"
        ans,object_beliefs = obc.get_object_represention_and_weighted_belief(depth_im,
        object_group_name,keras_model_dir,keras_model_name, baseline_results_dir)
        print "<Object Probabilities>" + repr(ans)
        print "<Object Beliefs>" + repr(list(object_beliefs))
        if clip_objects > 0:
            sort_ind = np.argsort(object_beliefs)
            print sort_ind
            object_beliefs[sort_ind[0:-1*clip_objects]] = 0
            object_beliefs_sum = np.sum(object_beliefs)
            object_beliefs = [x/object_beliefs_sum for x in object_beliefs]
            print "<Object Beliefs Clipped>" + repr(object_beliefs)
        return (list(object_beliefs) + ans)

if __name__ == '__main__':
    object_file_name = None
    object_group_name = None
    debug = False
    opts, args = getopt.getopt(sys.argv[1:],"g:hs:d")
    for opt,arg in opts:
        if opt == '-s':
            object_file_name = arg
        elif opt =='-g':
            object_group_name = arg
        elif opt == '-h':
            print "python get_initial_object_belief.py -s object_name |-g object_group_name object_file_dir"
        elif opt == '-d':
            debug = True
    object_file_dir = args[0]
    #rospy.init_node('Object_belief_node')

    if object_file_name is not None:
        save_object_file(object_file_dir + "/" + object_file_name, debug)
    if object_group_name is not None:
        get_belief_for_objects(object_group_name, object_file_dir, debug)
    #rospy.spin()
