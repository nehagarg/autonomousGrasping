import os
import sys
import getopt
import numpy as np
import yaml
import math
import argparse
import utils as loiv
import perception as perception
import matplotlib.pyplot as plt
from gqcnn import Visualizer as vis
from autolab_core import YamlConfig
from autolab_core import RigidTransform, Point, Box
#import get_initial_object_belief as giob
PICK_ACTION_ID = 10
OPEN_ACTION_ID = 9
CLOSE_ACTION_ID = 8
NUM_PREDICTIONS = 18
GAUSSIAN_VARIANCE = 1

min_x_i = 0.3379;
min_y_i = 0.0816;

initial_gripper_pose_index_x = 0;
initial_gripper_pose_index_y = 7;

class StoredDepthAndColorImageProcesser():
    def __init__(self,config_dir):
        print config_dir
        self.CAM_FRAME = 'kinect2_ir_optical_frame'
        self.WORLD_FRAME = 'world'
        self.config_path = os.path.join(config_dir,'config_files/dexnet_config/')
        print self.config_path
        self.config = YamlConfig(self.config_path + 'mico_control_node.yaml')
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

        transform_filename = self.config_path + "/"+ self.CAM_FRAME + '_' + self.WORLD_FRAME + '.tf'
        self.T_camera_world =  RigidTransform.load(transform_filename)

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

    # def segment_using_giob(self,image_dir,image_file_name):
    #     g = giob.GetInitialObjectBelief(None, True, True)
    #     filename_prefix = os.path.join(image_dir,image_file_name)
    #     (depth_im, cam_intr) = self.load_stored_depth_im(filename_prefix)
    #     (_, point_cloud_world, _) = g.get_world_point_cloud(depth_im,cam_intr, self.T_camera_world)
    #     seg_point_cloud_world= self.get_segmented_point_cloud_world(self.detector_cfg, point_cloud_world )
    #
    #     seg_point_cloud_cam = self.T_camera_world.inverse() * seg_point_cloud_world
    #     depth_im_seg = cam_intr.project_to_image(seg_point_cloud_cam)
    #
    #     vis.figure()
    #     vis.subplot(1,1,1)
    #     vis.imshow(depth_im_seg)
    #     vis.show()

    def load_stored_depth_im(self,filename_prefix):
        camera_intr =  perception.CameraIntrinsics.load(filename_prefix  + '.intr')
        depth_im = perception.DepthImage.open(filename_prefix + '.npy', frame=camera_intr.frame)
        vis.figure()
        vis.subplot(1,1,1)
        vis.imshow(depth_im)
        return (depth_im,camera_intr)



    def mark_gripper_in_depth_image(self,gripper_3D_pose, image_dir, image_file_name, debug = True):
        #self.load_stored_depth_im('/home/neha/WORK_FOLDER/unicorn_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/point_clouds/56_headphones_final-16-Nov-2015-11-29-41_instance0.yaml')
        print gripper_3D_pose
        print image_file_name
        #g = giob.GetInitialObjectBelief(None, True, True)
        filename_prefix = os.path.join(image_dir,image_file_name)
        cropped_depth_filename = filename_prefix + '_depth_cropped.npz'
        if os.path.exists(cropped_depth_filename):
            return #perception.DepthImage.open(cropped_depth_filename)
        camera_intr =  perception.CameraIntrinsics.load(filename_prefix  + '.intr')
        #depth_im = perception.DepthImage.open(filename_prefix + '.npy', frame=camera_intr.frame)
        #depth_im_inpainted = depth_im.inpaint(rescale_factor=self.config['inpaint_rescale_factor'])
        #depth_im = depth_im_inpainted
        depth_im = perception.DepthImage.open(filename_prefix + '_depth.npz', frame=camera_intr.frame)
        color_im = perception.ColorImage.open(filename_prefix + '.npz', frame=camera_intr.frame)
        gripper_point_world = Point(np.array(gripper_3D_pose), self.WORLD_FRAME)
        gripper_point_cam = self.T_camera_world.inverse() * gripper_point_world
        image_point = camera_intr.project(gripper_point_cam)
        #vis.figure()
        #vis.subplot(1,1,1)
        #vis.imshow(depth_im)


        #(_, point_cloud_world, _) = g.get_world_point_cloud(depth_im,camera_intr, self.T_camera_world)
        #TODO:Remove gripper points using mask RCNN
        # project into 3D
        point_cloud_cam = camera_intr.deproject(depth_im)
        point_cloud_world = self.T_camera_world * point_cloud_cam

        x_diff = min_x_i - gripper_3D_pose[0]
        #x_diff = -0.03
        print x_diff
        if x_diff > 0:
            x_diff = 0.0

        y_diff = min_y_i + 0.01*initial_gripper_pose_index_y - gripper_3D_pose[1]
        #y_diff = -0.03
        print y_diff

        print point_cloud_world.x_coords.shape
        for i in range(0,point_cloud_world.x_coords.shape[0]):
            point_cloud_world.x_coords[i] = point_cloud_world.x_coords[i] + x_diff
            point_cloud_world.y_coords[i] = point_cloud_world.y_coords[i] + y_diff

        point_cloud_cam_translated = self.T_camera_world.inverse()* point_cloud_world
        new_depth_im = camera_intr.project_to_image(point_cloud_cam_translated, False)
        new_depth_im_inpainted = new_depth_im.inpaint()
        new_depth_im = new_depth_im_inpainted

        new_point_cloud_cam = camera_intr.deproject(new_depth_im)
        new_point_cloud_world = self.T_camera_world * new_point_cloud_cam



        #seg_point_cloud_world= self.get_segmented_point_cloud_world(self.detector_cfg, point_cloud_world )

        #seg_point_cloud_cam = self.T_camera_world.inverse() * seg_point_cloud_world
        #depth_im_seg = camera_intr.project_to_image(seg_point_cloud_cam)
        #Taking point in bounding box
        gripper_offset = 0.018
        self.detector_cfg['min_pt'][0] = min_x_i - gripper_offset
        #self.detector_cfg['min_pt'][2] = self.detector_cfg['min_pt'][2] + 0.01
        self.detector_cfg['max_pt'][0] = 0.85
        point_cloud_world_segmented = self.get_segmented_point_cloud_world( self.detector_cfg, new_point_cloud_world )
        point_cloud_cam_translated_and_segmented = self.T_camera_world.inverse()* point_cloud_world_segmented
        new_depth_im_seg = camera_intr.project_to_image(point_cloud_cam_translated_and_segmented)

        new_depth_cropped = new_depth_im_seg.crop(184,208,184/2,161 + (208/2)) #crop area determined by crop points below
        #new_depth_im_seg_inpainted = new_depth_im_seg.inpaint()

        #vis.figure()
        #vis.subplot(1,1,1)
        #vis.imshow(depth_im_seg)
        #vis.show()

        if debug:
            crop_points_world = []

            # crop_points_world.append([gripper_3D_pose[0] - gripper_offset, gripper_3D_pose[1], gripper_3D_pose[2]])
            # crop_points_world.append([gripper_3D_pose[0] -gripper_offset+ 0.4, gripper_3D_pose[1], gripper_3D_pose[2]])
            # crop_points_world.append([gripper_3D_pose[0]-gripper_offset, gripper_3D_pose[1] - 0.16, gripper_3D_pose[2]])
            # crop_points_world.append([gripper_3D_pose[0]-gripper_offset, gripper_3D_pose[1] + 0.16, gripper_3D_pose[2]])

            crop_points_world.append([min_x_i, min_y_i + 0.01*initial_gripper_pose_index_y, gripper_3D_pose[2]])
            crop_points_world.append([0.85, min_y_i + 0.01*initial_gripper_pose_index_y, gripper_3D_pose[2]])
            crop_points_world.append([min_x_i-gripper_offset, min_y_i + 0.01*initial_gripper_pose_index_y - 0.16, gripper_3D_pose[2]])
            crop_points_world.append([min_x_i-gripper_offset, min_y_i + 0.01*initial_gripper_pose_index_y + 0.16, gripper_3D_pose[2]])


            image_points = []
            for crop_point_world in crop_points_world:
                point_world = Point(np.array(crop_point_world), self.WORLD_FRAME)
                point_cam = self.T_camera_world.inverse() * point_world
                image_points.append(camera_intr.project(point_cam))
                print image_points[-1]

            vis.figure()
            vis.subplot(1,3,1)
            vis.imshow(color_im)
            plt.plot(image_point.x, image_point.y, c='r', marker='+', mew=2.5, ms=7.5)
            #for box_point in image_points:
            #    plt.plot(box_point.x, box_point.y, c='g', marker='*', mew=2.5, ms=7.5)
            vis.subplot(1,3,2)
            vis.imshow(new_depth_im)
            for box_point in image_points:
                plt.plot(box_point.x, box_point.y, c='g', marker='*', mew=2.5, ms=7.5)
            #vis.imshow(color_im)
            #vis.imshow(new_depth_im_seg_inpainted)
            vis.subplot(1,3,3)
            #vis.imshow(color_im)
            vis.imshow(new_depth_cropped)
            vis.show()
        new_depth_cropped.save(cropped_depth_filename)
        return #new_depth_cropped

def get_float_array(a):
    return [float(x) for x in a]

#Lines contain name of vision file also
def get_data_from_line(line):
    #print line
    ans = {}
    sasor = line.strip().split('*')
    init_state = sasor[0].split('|')
    next_state = sasor[2].split('|')
    ans['index'] = get_float_array(init_state[0].split(' ')[0:2])
    ans['init_gripper'] = get_float_array(init_state[0].split(' ')[2:])
    ans['init_object'] = get_float_array(init_state[1].split(' '))
    init_joint_values = get_float_array(init_state[2].split(' '))
    ans['init_joint_values'] = [init_joint_values[0], init_joint_values[2]]
    ans['next_gripper'] = get_float_array(next_state[0].split(' '))
    ans['next_object'] = get_float_array(next_state[1].split(' '))
    next_joint_values =  get_float_array(next_state[2].split(' '))
    ans['next_joint_values'] = [next_joint_values[0], next_joint_values[2]]
    ans['action'] = int(sasor[1])
    ans['reward'] = float(sasor[-1 ])
    ans['touch'] = get_float_array(sasor[-2].split('|')[-3].split(' '))
    ans['vision_movement'] = float(sasor[-2].split('|')[-2].split(' ')[0])
    ans['image_file_name'] = sasor[-2].split('|')[-1].split(' ')[0]
    return ans


def load_data(object_name_list, data_dir, object_id_mapping_file, debug = True):
    object_id_mapping = yaml.load(file(object_id_mapping_file, 'r'))
    ans = {}
    for object_name in object_name_list:
        final_data_dir = os.path.join(data_dir , object_name)
        print final_data_dir
        files = [os.path.join(final_data_dir, f) for f in os.listdir(final_data_dir) if object_name in f and f.endswith('.txt') and '_24_' in f]
        for file_ in files:
            print file_
            if debug:
                if len(ans) > 1:
                    break
            #if  file not in bad_list:
            #    continue
            file_nan_count = 0
            with open(file_, 'r') as f:
                for line in f:
                    if 'nan' in line:
                        file_nan_count = file_nan_count + 1
                        continue
                    sasor = get_data_from_line(line.strip())
                    sasor['object_id'] = object_id_mapping[object_name]
                    if('openAction' not in file_):
                        if sasor['action'] != OPEN_ACTION_ID:
                            if sasor['action']==PICK_ACTION_ID:
                                #Assuming pick action will always be after a close action
                                sasor['touch_prev'] = ans[CLOSE_ACTION_ID][-1]['touch']
                            if(sasor['reward'] > -999):
                                if sasor['action'] not in ans:
                                    ans[sasor['action']]= []
                                ans[sasor['action']].append(sasor)
                    else:
                        if sasor['action'] == OPEN_ACTION_ID:
                            if(sasor['reward'] > -999):
                                if sasor['action'] not in ans:
                                    ans[sasor['action']]= []
                                ans[sasor['action']].append(sasor)

            print file_nan_count
            assert(file_nan_count < 5)
        #print ans
    return ans


class LoadTransitionData():
    def __init__(self):
        self.ans = None
        self.imProc = None
    def get_data(self,object_name_list, data_dir, object_id_mapping_file, debug = True):
        if self.ans is None:
            self.ans = load_data(object_name_list, data_dir, object_id_mapping_file, debug)
        return self.ans

    def get_depth_image(self,griper_3D_pose, image_dir,image_file_name):
        if self.imProc is None:
            self.imProc = StoredDepthAndColorImageProcesser(image_dir)
        self.imProc.mark_gripper_in_depth_image(griper_3D_pose, image_dir,image_file_name, False)
        #return depth_im
        #image_file_prefix = os.path.join(image_dir, image_file_name)
        #camera_file = image_file_prefix + '.intr'
        #depth_file = image_file_prefix + '_depth.npz'
        #depth_image = perception.

    def get_training_data(self,action_, object_name_list, data_dir, object_id_mapping_file, image_dir=None, debug = False):
        ans = self.get_data(object_name_list, data_dir, object_id_mapping_file, debug)
        input_s = []
        expected_outcome = []
        image_input = []
        pick_success = []
        for action in ans.keys():
            #if(action % 2 == 0 or action > 8):
            if(action ==action_ or (action_ < 0 and action != PICK_ACTION_ID)):
                for i in range(0,len(ans[action])):
                    if debug:
                        if i>2:
                            break
                    #print ans[action][i]
                    input_s_entry = ans[action][i]['init_gripper'][0:2]
                    input_s_entry = input_s_entry +  ans[action][i]['init_object'][0:2]
                    (theta_x, theta_y, theta_z) = loiv.quaternion_to_euler_angle(
                    ans[action][i]['init_object'][6],
                    ans[action][i]['init_object'][3],
                    ans[action][i]['init_object'][4],
                    ans[action][i]['init_object'][5])
                    input_s_entry = input_s_entry +  [math.radians(theta_z)]
                    input_s_entry = input_s_entry +  ans[action][i]['init_joint_values']
                    input_s_entry = input_s_entry +  [ ans[action][i]['object_id']]
                    if image_dir is not None:
                        input_s_entry = input_s_entry +  [action]
                    input_s.append(input_s_entry)
                    gripper_pos_change = np.array(ans[action][i]['next_gripper'][0:2]) - np.array(ans[action][i]['init_gripper'][0:2])
                    object_pos_change =  np.array(ans[action][i]['next_object'][0:2]) - np.array(ans[action][i]['init_object'][0:2])
                    (theta_x, theta_y, theta_z) = loiv.quaternion_to_euler_angle(
                    ans[action][i]['next_object'][6],
                    ans[action][i]['next_object'][3],
                    ans[action][i]['next_object'][4],
                    ans[action][i]['next_object'][5])
                    theta_z_change = np.array([math.radians(theta_z) - input_s_entry[4]])
                    joint_angle_change = np.array(ans[action][i]['next_joint_values']) - np.array(ans[action][i]['init_joint_values'])
                    touch_values = np.array(ans[action][i]['touch'])
                    expected_outcome_entry = np.concatenate((gripper_pos_change, object_pos_change, theta_z_change, joint_angle_change, touch_values))
                    expected_outcome.append(expected_outcome_entry)
                    if image_dir is not None : #and action != PICK_ACTION_ID:
                        #image_input_entry = self.get_depth_image(ans[action][i]['next_gripper'][0:3], image_dir, ans[action][i]['image_file_name'])
                        #image_input.append(image_input_entry.raw_data)
                        self.get_depth_image(ans[action][i]['next_gripper'][0:3], image_dir, ans[action][i]['image_file_name'])
                        image_input.append(ans[action][i]['image_file_name'])
                    if action== PICK_ACTION_ID:
                        pick_success_entry = 0
                        if ans[action][i]['reward'] > 0:
                            pick_success_entry = 1
                        pick_success.append(pick_success_entry)

        #print input_s
        #print expected_outcome
        if(action_ == PICK_ACTION_ID):
            return np.array(input_s),np.array(pick_success)
        else:
            return np.array(input_s), np.array(expected_outcome), np.array(image_input)

    def generate_next_state_entry(self,input_si, expected_outcomei):
        gripper_pos = np.array(expected_outcomei[0:2] + input_si[0:2])
        object_pos = np.array(expected_outcomei[2:4] + input_si[2:4])
        theta_z = np.array([expected_outcomei[4] + input_si[4]])
        joint_values =np.array(expected_outcomei[5:7] + input_si[5:7])
        object_id = np.array([input_si[7]])
        #action = np.array([input_si[8]])
        next_state_entry = np.concatenate((gripper_pos,object_pos,theta_z,joint_values,object_id))
        return next_state_entry

    def load_cropped_depth_image(self,image_dir,image_file_name):
        filename_prefix = os.path.join(image_dir,image_file_name)
        cropped_depth_filename = filename_prefix + '_depth_cropped.npz'
        depth_im =  perception.DepthImage.open(cropped_depth_filename)
        return depth_im.raw_data

    def observation_data_generator(self, file_name, image_dir, batch_size):
        #data_file = open(file_name, 'r')
        while True:
            input_s_gen = []
            image_input_gen = []
            input_s_existing = []
            probability = []
            with open(file_name, 'r') as f:
                for line in f:
                    stripped_line = line.strip()
                    line_components = stripped_line.split('|')
                    input_s_gen_entry = [float(x) for x in line_components[0].split(' ')]
                    image_input_gen_entry = self.load_cropped_depth_image(image_dir, line_components[1])
                    input_s_existing_entry = [float(x) for x in line_components[2].split(' ')]
                    prob = float(line_components[3])
                    input_s_gen.append(input_s_gen_entry)
                    image_input_gen.append(image_input_gen_entry)
                    input_s_existing.append(input_s_existing_entry)
                    probability.append(prob)
                    if len(probability) == batch_size:
                        yield ([np.array(input_s_gen), np.array(image_input_gen), np.array(input_s_existing), np.array(probability)], None)
                        input_s_gen = []
                        image_input_gen = []
                        input_s_existing = []
                        probability = []
                if len(probability) > 0 : #yield last batch
                    yield ([np.array(input_s_gen), np.array(image_input_gen), np.array(input_s_existing), np.array(probability)], None)
                    input_s_gen = []
                    image_input_gen = []
                    input_s_existing = []
                    probability = []

    def write_training_data_for_transition_model(self,action,object_name_list,data_dir, object_id_mapping_file, debug = False):
        input_s_, expected_outcome_, image_input_ = self.get_training_data(action, object_name_list, data_dir, object_id_mapping_file, None, debug)
        num_samples = input_s_.shape[0]
        arr = np.arange(num_samples)
        np.random.shuffle(arr)
        output_file_name = '../grasping_ros_mico/scripts/transition_model/data/' + repr(action) + '.data'
        output_file = open(output_file_name, 'w')
        input_s_shuffled = input_s_[arr[0:num_samples]]
        expected_outcome_shuffled = expected_outcome_[arr[0:num_samples]]
        for i in range(0,num_samples):
            output_file.write(' '.join(str(x) for x in input_s_[i]))
            output_file.write('|')
            output_file.write(' '.join(str(x) for x in expected_outcome_[i]))
            output_file.write('\n')
        output_file.close()

    def write_training_data_for_observation_model(self,action, object_name_list, data_dir, object_id_mapping_file, image_dir, debug = False):
        input_s_, expected_outcome_, image_input_ = self.get_training_data(action, object_name_list, data_dir, object_id_mapping_file, image_dir, debug)
        num_samples = input_s_.shape[0]
        arr = np.arange(num_samples)
        np.random.shuffle(arr)
        output_file_name = '../grasping_ros_mico/scripts/observation_model/data_cluster_size_10/' + repr(action) + '.data'
        output_file = open(output_file_name, 'w')
        input_s_shuffled = input_s_[arr[0:num_samples]]
        expected_outcome_shuffled = expected_outcome_[arr[0:num_samples]]
        image_input_shuffled = image_input_[arr[0:num_samples]]
        num_samples = min(150000, num_samples)
        cluster_size = 10
        num_clusters = int(1.0*num_samples/cluster_size)
        if(num_clusters*cluster_size < num_samples):
            num_clusters = num_clusters + 1
        input_s_gen = []
        image_input_gen = []
        input_s_existing = []
        #image_input_existing = []
        probability = []
        new_image_input = []

        for k in range(0,num_clusters):
            for i in range(k*cluster_size,min(num_samples,(k+1)*cluster_size)):
                if input_s_shuffled[i][-1] != PICK_ACTION_ID: #Pick is terminal action
                    input_s_gen_entry = self.generate_next_state_entry(input_s_shuffled[i], expected_outcome_shuffled[i])
                    image_input_gen_entry = self.load_cropped_depth_image(image_dir, image_input_shuffled[i])
                    new_image_input_entry = image_input_shuffled[i]
                    for j in range(k*cluster_size,min(num_samples,(k+1)*cluster_size)):
                        if input_s_shuffled[j][-1] == input_s_shuffled[i][-1]: #Same action
                            input_s_existing_entry = self.generate_next_state_entry(input_s_shuffled[j], expected_outcome_shuffled[j])
                            image_input_existing_entry = self.load_cropped_depth_image(image_dir, image_input_shuffled[j])
                            #input_s_gen.append(input_s_gen_entry)
                            #image_input_gen.append(image_input_gen_entry)
                            #input_s_existing.append(input_s_existing_entry)
                            #image_input_existing.append(image_input_existing_entry)
                            distance =  np.square(np.subtract(image_input_gen_entry,image_input_existing_entry)).mean()
                            prob = math.exp(-0.5*distance/(GAUSSIAN_VARIANCE*GAUSSIAN_VARIANCE))
                            #probability.append(prob)
                            print len(probability)
                            output_file.write(' '.join(str(x) for x in input_s_gen_entry))
                            output_file.write('|')
                            output_file.write(image_input_shuffled[i])
                            output_file.write('|')
                            output_file.write(' '.join(str(x) for x in input_s_existing_entry))
                            output_file.write('|')
                            output_file.write(str(prob))
                            output_file.write('\n')
        #return np.array(input_s_gen), np.array(image_input_gen), np.array(input_s_existing), np.array(probability)
        output_file.close()

def main():
    object_id_mapping_file = '../grasping_ros_mico/ObjectNameToIdMapping.yaml'
    data_dir = '../grasping_ros_mico/data_low_friction_table_exp_wider_object_workspace_ver8/data_for_regression'
    image_dir = '../grasping_ros_mico'
    object_id_mapping = yaml.load(file(object_id_mapping_file, 'r'))
    object_name_list = object_id_mapping.keys()
    parser = argparse.ArgumentParser()
    help_ = "Action name"
    parser.add_argument("-a",
                        "--action",
                        help=help_)
    help_ = "Task transition or observation"
    parser.add_argument("-t",
                        "--task",
                        help=help_)
    args = parser.parse_args()
    #LoadTransitionData().get_training_data(int(args.action), object_name_list, data_dir, object_id_mapping_file, image_dir, True)
    if args.task == 'o':
        LoadTransitionData().write_training_data_for_observation_model(int(args.action), object_name_list, data_dir, object_id_mapping_file, image_dir)
    if args.task == 't':
        LoadTransitionData().write_training_data_for_transition_model(int(args.action), object_name_list, data_dir, object_id_mapping_file)
    #imProc = StoredDepthAndColorImageProcesser(image_dir)
    #imProc.segment_using_giob('./test_images','sample_depth_im')
    #imProc.mark_gripper_in_depth_image([0.3379, 0.1516, 1.0833], './test_images','sample_depth_im')
    #get_depth_image([0.3379, 0.1516, 1.0833], './test_images','sample_depth_im')

if __name__ == '__main__':
    main()
