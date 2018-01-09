import plot_despot_results as pdr

def main():
    base_dir = './unicorn_csv_files/grasping_ros_mico/results/despot_logs/low_friction_table/vrep_scene_ver6/multiObjectType/'
    dir_list = []
    for i in range(0,7):
        dir_list.append('belief_uniform_baseline_' + repr(i) + '_reward100_penalty10/simulator/fixed_distribution/')
    dir_list.append('belief_uniform_cylinder_7_8_9_reward100_penalty10/use_discretized_data/simulator/fixed_distribution/')
    patterns = ['g3db_instances|', 'all_cylinders|']
    out_dir = './unicorn_csv_files/grasping_ros_mico/results/despot_logs/low_friction_table/vrep_scene_ver6/multiObjectType/basline_and_cylinder_belief'
    pdr.PROBLEM_NAME = "vrep"
    for pattern in patterns:
        pdr.generate_combined_csv(base_dir, dir_list, pattern, out_dir)
if __name__ == '__main__':
    main()    