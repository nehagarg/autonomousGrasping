
from log_file_parser import ParseLogFile
from grasping_object_list import get_grasping_object_name_list
import math
import load_objects_in_vrep as ol
import os

def check_pick(file_name, diff_values):
    fullData = ParseLogFile(file_name, '', 0, 'vrep/ver5').getFullDataWithoutBelief()
    final_state = fullData['stepInfo'][-1]['state']
    final_reward = fullData['stepInfo'][-1]['reward']
    final_action = ("").join(fullData['stepInfo'][-1]['action'][:-1].split(" "))
    distance = 0
    if(final_action == '-Action=ActionisPICK'):
        
        distance = math.pow(final_state.g_x - final_state.o_x + diff_values[0], 2);
        distance = distance + math.pow(final_state.g_y - final_state.o_y + diff_values[1], 2);
        distance = distance + math.pow(final_state.g_z - final_state.o_z + diff_values[2], 2);
        distance = math.pow(distance, 0.5);
        if(distance > 0.12):
            calculated_reward = -10
        else:
            calculated_reward = 100
        
    else:
        calculated_reward = final_reward
    return [calculated_reward, final_reward, distance]
    
def get_diff_values(pattern):
    pick_point_x_diff = -0.03;
    pick_point_y_diff = 0.0;
    default_initial_object_x = 0.4919;
    default_initial_object_y = 0.1562;
    default_initial_object_z = 1.0998;
    
    
    object_property_dir = '../grasping_ros_mico/g3db_object_labels/object_instances/object_instances_updated/'
    if 'Cylinder' in pattern:
        object_property_dir = '../grasping_ros_mico/pure_shape_labels/'
    mesh_properties = ol.get_object_properties(pattern, object_property_dir)
    pick_point = mesh_properties['pick_point']
    expected_pick_x = default_initial_object_x + pick_point_x_diff
    expected_pick_y = default_initial_object_y + pick_point_y_diff
    x_diff = expected_pick_x - pick_point[0];
    y_diff = expected_pick_y - pick_point[1];
    
    object_pose_z = mesh_properties['object_initial_pose_z']
    z_diff = object_pose_z - default_initial_object_z
    return [x_diff, y_diff, z_diff]

def correct_pick_values(dir_name, pattern, diff_values):
    files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if pattern+'_' in f and f.endswith('.log')]
    
    for file in files:
        [calculated_reward, actual_reward, distance] = check_pick(file, diff_values)
        if calculated_reward !=actual_reward:
            print file + " " + repr(calculated_reward) + " " +repr(actual_reward) + " " + repr(distance)
            
    
    
    
def main():
    base_dir = '/home/neha/WORK_FOLDER/unicorn_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/low_friction_table/vrep_scene_ver6/multiObjectType/'
    dir_list = []
    for i in range(0,7):
        dir_list.append('belief_uniform_baseline_' + repr(i) + '_reward100_penalty10/simulator/fixed_distribution/')
    dir_list.append('belief_uniform_cylinder_7_8_9_reward100_penalty10/use_discretized_data/simulator/fixed_distribution/t5_n40')
    dir_list.append('belief_uniform_cylinder_7_8_9_reward100_penalty10/use_discretized_data/simulator/fixed_distribution/learning/version14')
    patterns = get_grasping_object_name_list('cylinder_and_g3db_instances')
    
    for pattern in patterns:
        print '#############################'
        print pattern
        diff_values = get_diff_values(pattern)
        print diff_values
        for dir_name in dir_list:
            correct_pick_values(os.path.join(base_dir,dir_name), pattern, diff_values)

if __name__ == '__main__':
    main()    