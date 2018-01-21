import plot_despot_results as pdr

def get_dir_list():
    dir_list = []
    for i in range(0,7):
        dir_list.append('belief_uniform_baseline_' + repr(i) + '_reward100_penalty10/simulator/fixed_distribution/')
    #dir_list.append('belief_uniform_cylinder_7_8_9_reward100_penalty10/use_discretized_data/simulator/fixed_distribution/')
    dir_list.append('belief_uniform_g3db_instances_train1_reward100_penalty10/use_discretized_data/use_weighted_belief/simulator/fixed_distribution/horizon90/')
    return dir_list
    
def main():
    base_dir = "/home/neha/WORK_FOLDER/unicorn_dir_mount/neha_github/autonomousGrasping/python_scripts/"
    base_dir = base_dir + './unicorn_csv_files/grasping_ros_mico/results/despot_logs/low_friction_table/vrep_scene_ver6/multiObjectType/'
    
    dir_list = get_dir_list()
    #patterns = ['g3db_instances|', 'all_cylinders|']
    patterns = ['g3db_instances_train1|', 'g3db_instances_validation1|']
    out_dir = './unicorn_csv_files/grasping_ros_mico/results/despot_logs/low_friction_table/vrep_scene_ver6/multiObjectType/basline_and_cylinder_belief'
    pdr.PROBLEM_NAME = "vrep"
    for pattern in patterns:
        pdr.generate_combined_csv(base_dir, dir_list, pattern, out_dir)
if __name__ == '__main__':
    main()    