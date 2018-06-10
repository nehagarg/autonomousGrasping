
from log_file_parser import ParseLogFile
from grasping_object_list import get_grasping_object_name_list
#from adaboost_data_generator import get_label_string



def createActionHash(state_type = 'toy'):
    action_string_hash = {}
    if state_type == 'toy':
        action_string_hash['-Action=ActionisINCREASEXby1'] = 0
        action_string_hash['-Action=ActionisINCREASEXby16'] = 1
        action_string_hash['-Action=ActionisDECREASEXby1'] = 2
        action_string_hash['-Action=ActionisDECREASEXby16'] = 3
        action_string_hash['-Action=ActionisINCREASEYby1'] = 4
        action_string_hash['-Action=ActionisINCREASEYby16'] = 5
        action_string_hash['-Action=ActionisDECREASEYby1'] = 6
        action_string_hash['-Action=ActionisDECREASEYby16'] = 7
        action_string_hash['-Action=ActionisCLOSEGRIPPER'] = 8
        action_string_hash['-Action=ActionisOPENGRIPPER'] = 9
    if state_type == 'vrep_old':
        action_string_hash['-Action=ActionisINCREASEXby0.01'] = 0
        action_string_hash['-Action=ActionisINCREASEXby0.02'] = 1
        action_string_hash['-Action=ActionisINCREASEXby0.04'] = 2
        action_string_hash['-Action=ActionisINCREASEXby0.08'] = 3
        action_string_hash['-Action=ActionisDECREASEXby0.01'] = 4
        action_string_hash['-Action=ActionisDECREASEXby0.02'] = 5
        action_string_hash['-Action=ActionisDECREASEXby0.04'] = 6
        action_string_hash['-Action=ActionisDECREASEXby0.08'] = 7
        action_string_hash['-Action=ActionisINCREASEYby0.01'] = 8
        action_string_hash['-Action=ActionisINCREASEYby0.02'] = 9
        action_string_hash['-Action=ActionisINCREASEYby0.04'] = 10
        action_string_hash['-Action=ActionisINCREASEYby0.08'] = 11
        action_string_hash['-Action=ActionisDECREASEYby0.01'] = 12
        action_string_hash['-Action=ActionisDECREASEYby0.02'] = 13
        action_string_hash['-Action=ActionisDECREASEYby0.04'] = 14
        action_string_hash['-Action=ActionisDECREASEYby0.08'] = 15
        action_string_hash['-Action=ActionisCLOSEGRIPPER'] = 16
        action_string_hash['-Action=ActionisOPENGRIPPER'] = 17
        action_string_hash['-Action=ActionisPICK'] = 18
    if state_type == 'vrep':
        action_string_hash['-Action=ActionisINCREASEXby0.01'] = 0
        action_string_hash['-Action=ActionisINCREASEXby0.08'] = 1
        action_string_hash['-Action=ActionisDECREASEXby0.01'] = 2
        action_string_hash['-Action=ActionisDECREASEXby0.08'] = 3
        action_string_hash['-Action=ActionisINCREASEYby0.01'] = 4
        action_string_hash['-Action=ActionisINCREASEYby0.08'] = 5
        action_string_hash['-Action=ActionisDECREASEYby0.01'] = 6
        action_string_hash['-Action=ActionisDECREASEYby0.08'] = 7
        action_string_hash['-Action=ActionisCLOSEGRIPPER'] = 8
        action_string_hash['-Action=ActionisOPENGRIPPER'] = 9
        action_string_hash['-Action=ActionisPICK'] = 10
    return action_string_hash

#for cpp
def get_log_file_parser(log_filename):
    lfp =  ParseLogFile(log_filename, 'vrep', 0, 'vrep')
    return lfp
def get_next_step_from_log(lfp,step):
    action_string_hash = createActionHash('vrep')
    state_value = lfp.stepInfo_[step]['state']
    touch_0 = lfp.stepInfo_[step]['obs'].sensor_obs[0]
    touch_1 = lfp.stepInfo_[step]['obs'].sensor_obs[1]
    vision_movement = 0
    if hasattr(lfp.stepInfo_[step]['obs'], 'vision_movement'):
        vision_movement = lfp.stepInfo_[step]['obs'].vision_movement
    values = [state_value.g_x, state_value.g_y, state_value.g_z]
    values.append(state_value.g_xx)
    values.append(state_value.g_yy)
    values.append(state_value.g_zz)
    values.append(state_value.g_w)
    values.append(state_value.o_x)
    values.append(state_value.o_y)
    values.append(state_value.o_z)
    values.append(state_value.fj1)
    values.append(state_value.fj2)
    values.append(state_value.fj3)
    values.append(state_value.fj4)
    values.append(touch_0)
    values.append(touch_1)
    values.append(vision_movement)
    values.append(action_string_hash[("").join(lfp.stepInfo_[step]['action'][:-1].split(" "))])
    values.append(lfp.stepInfo_[step]['reward'])

    return ' '.join([str(x) for x in values])



def process_full_data(fullData,seqs, state_type = 'toy', isTraining = True):
    action_string_hash = createActionHash(state_type.split('/')[0])
    max_steps = 49
    max_reward = 100
    if state_type == 'toy':
        max_reward = 20
        max_steps = 89
    #print action_string_hash
    #print fullData
    num_steps = len(fullData['stepInfo'])


    num_actions = len(action_string_hash.keys())
    successTraj = False
    if len(fullData['stepInfo']) > 0 and fullData['stepInfo'][-1]['reward'] == max_reward:
        successTraj = True
    if (successTraj or not isTraining):
        if num_steps > max_steps:
            print num_steps
        seq = []
        j_range = len(fullData['stepInfo'])
        #if not isTraining:
        #    j_range = j_range - 1
        for j in range(0,j_range):
            act = num_actions
            obs = None
            if 'action' in fullData['stepInfo'][j]:
                act = action_string_hash[("").join(fullData['stepInfo'][j]['action'][:-1].split(" "))]
            if 'obs' in fullData['stepInfo'][j]:
                obs = fullData['stepInfo'][j]['obs'].convert_to_array(state_type)
                if 'weighted' in state_type:
                    obs = obs + fullData['roundInfo']['initial_object_probs']
                #obs = fullData['stepInfo'][j]['obs'].sensor_obs
                #obs.append(fullData['stepInfo'][j]['obs'].gripper_l_obs)
                #obs.append(fullData['stepInfo'][j]['obs'].gripper_r_obs)
                #obs.append(fullData['stepInfo'][j]['obs'].x_w_obs)
                #obs.append(fullData['stepInfo'][j]['obs'].y_w_obs)
            seq.append((act,obs))
        if 'weighted' in state_type:
	    obs = fullData['roundInfo']['state'].get_obs().convert_to_array(state_type)
            obs = obs + fullData['roundInfo']['initial_object_probs']
            seq = [('$', obs) ] + seq
        if not isTraining:
            seq.append((num_actions,None))
        seqs.append(seq)

def parse_file(file_name, belief_type = '', isTraining = True, round_no = 0, state_type = 'toy'):

    if not isTraining:
        round_no = -1
    fullData =  ParseLogFile(file_name, belief_type, round_no, state_type).getFullDataWithoutBelief()
    seqs = []
    process_full_data(fullData,seqs, state_type, isTraining)
    return seqs

def parse(fileName, belief_type = '', isTraining = False):
    seqs = []
    if fileName =='test':

        for i in range(0,100):
            #logfileName = '/home/neha/WORK_FOLDER/neha_github/apc/rosmake_ws/despot_vrep_glue/results/despot_logs/VrepData_gaussian_belief_with_state_in_belief_t5_n10_trial_' + repr(i) +'.log'
            #logfileName = '/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/separate_close_reward/singleObjectType/cylinder_9cm_reward100_penalty10/t1_n320_withoutLCAP/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t1_n320_trial_' + repr(i) +'.log'
            #logfileName = '/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/low_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/t5_n80/Table_scene_low_friction_9cm_cylinder_belief_gaussian_with_state_in_t5_n80_trial_' + repr(i) +'.log'
            #logfileName = '/home/neha/WORK_FOLDER/unicorn_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n20/Table_scene_low_friction_7cm_cylinder_belief_gaussian_with_state_in_t5_n20_trial_' + repr(i) +'.log'
            #logfileName = '/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n160/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t5_n160_trial_' + repr(i) +'.log'
            #logfileName = '/home/neha/WORK_FOLDER/unicorn_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/simulator/learning/version8/Table_scene_low_friction_7cm_cylinder_v8_trial_' + repr(i) +'.log'
            #logfileName = '/home/neha/WORK_FOLDER/unicorn_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/simulator/fixed_distribution/learning/version8/Table_scene_low_friction_7cm_cylinder_v8_trial_' + repr(i) +'.log'
            t = '5'
            scenario = '80'

            #object = '84_yogurtcup_final-13-Nov-2015-10-32-33_instance0'

            object = 'Cylinder_7'
            #object='G3DB84_yogurtcup_final'
            #object = 'G3DB1_Coffeecup_final-20-dec-2015'
            #object = '9cm'
            logfileName = '/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/low_friction_table/vrep_scene_ver7/multiObjectType/'
            #logfileName = logfileName + 'belief_uniform_g3db_single_reward100_penalty10/simulator/fixed_distribution/t' + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
            logfileName = logfileName + 'belief_uniform_cylinder_7_8_9_reward100_penalty10/use_discretized_data/'
            #logfileName = logfileName + 'belief_uniform_g3db_instances_train1_reward100_penalty10/use_discretized_data/use_weighted_belief/fixed_distribution/horizon90/'
            logfileName = logfileName +'t' + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
            #logfileName = logfileName + 'belief_uniform_g3db_1_84_reward100_penalty10/icp_score_weighted/simulator/fixed_distribution/learning/version13/'
            #logfileName = logfileName + 'Table_scene_' + object + '_trial_' + repr(i) +'.log'
            #object='8cm'
            #logfileName = '/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_uniform_cylinder_7_8_9_reward100_penalty100/fixed_distribution/t' + t + '_n' + scenario + '/Table_scene_low_friction_'+ object + '_cylinder_belief_uniform_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'

            #print i
            #seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            #logfileName = '/home/neha/WORK_FOLDER/unicorn_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/simulator/fixed_distribution/t1_n40/Table_scene_low_friction_7cm_cylinder_belief_gaussian_with_state_in_t1_n40_trial_' + repr(i) +'.log'

            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver7')

            #logfileName = '../../grasping_ros_mico/results/despot_logs/singleObjectType/cylinder_9cm_reward100_penalty10/t5_n20/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t5_n20_trial_' + repr(i) +'.log'
            #print i
            #seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            #logfileName = '../../grasping_ros_mico/results/despot_logs/singleObjectType/cylinder_9cm_reward100_penalty10/t1_n20/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t1_n20_trial_' + repr(i) +'.log'
            #print i
            #seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
    elif fileName == 'toy/version1':
       for i in range(0,2000):
           for t in ['5']:
               for scenario in ['10','20']:
                   logfileName = '../../graspingV4/results/despot_logs/t' + t + "_n" + scenario + "/Toy_train_belief_default_t" + t + "_n" + scenario+ "_trial_" + repr(i) + ".log"
                   seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'toy')
    elif fileName in ['vrep/version25']:
        object_class_names = ['g3db_instances_train3_version7_label_3_class_0',
                                'g3db_instances_train3_version7_label_3_class_1',
                                'g3db_instances_train3_version7_label_3_class_2',
                                'g3db_instances_train3_version7_label_3_class_3',
                                'g3db_instances_train3_version7_label_3_class_4']
        object_class_samples_mapping = {}
        object_class_samples_mapping['g3db_instances_train3_version7_label_3_class_0'] = 2000
        object_class_samples_mapping['g3db_instances_train3_version7_label_3_class_1'] = 500
        object_class_samples_mapping['g3db_instances_train3_version7_label_3_class_2'] = 4000
        object_class_samples_mapping['g3db_instances_train3_version7_label_3_class_3'] = 3000
        object_class_samples_mapping['g3db_instances_train3_version7_label_3_class_4'] = 4000

        for object_class_name in object_class_names:
            object_list = get_grasping_object_name_list(object_class_name)

            for i in range(0,object_class_samples_mapping[object_class_name]):
                for t in ['5']:
                    for scenario in ['160']:
                       for object in object_list:
                            logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/'
                            logfileName = logfileName + 'vrep_scene_ver7/multiObjectType/'
                            logfileName = logfileName + 'belief_uniform_g3db_instances_train3_reward100_penalty10/'
                            logfileName = logfileName + 'use_discretized_data/use_binary_touch'
                            logfileName = logfileName + '/use_wider_workspace/use_probabilistic_neighbour_step'
                            logfileName = logfileName + '/use_discrete_observation_in_update/'
                            logfileName = logfileName + 'use_discrete_observation_in_step/'
                            logfileName = logfileName + 'use_weighted_belief/'
                            logfileName = logfileName + 'use_classifier/kmeans-extra-data_label_3_20180419-123116/horizon90/t'
                            logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                            #print i
                            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver7/weighted_5')


    elif fileName in ['vrep/version24']:
        object_class_names = ['g3db_instances_train2_version7_label_3_class_0',
                                'g3db_instances_train2_version7_label_3_class_1',
                                'g3db_instances_train2_version7_label_3_class_2',
                                'g3db_instances_train2_version7_label_3_class_3',
                                'g3db_instances_train2_version7_label_3_class_4']
        object_class_samples_mapping = {}
        object_class_samples_mapping['g3db_instances_train2_version7_label_3_class_0'] = 2000
        object_class_samples_mapping['g3db_instances_train2_version7_label_3_class_1'] = 500
        object_class_samples_mapping['g3db_instances_train2_version7_label_3_class_2'] = 4000
        object_class_samples_mapping['g3db_instances_train2_version7_label_3_class_3'] = 3000
        object_class_samples_mapping['g3db_instances_train2_version7_label_3_class_4'] = 4000

        for object_class_name in object_class_names:
            object_list = get_grasping_object_name_list(object_class_name)

            for i in range(0,object_class_samples_mapping[object_class_name]):
                for t in ['5']:
                    for scenario in ['160']:
                       for object in object_list:
                            logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/'
                            logfileName = logfileName + 'vrep_scene_ver7/multiObjectType/'
                            logfileName = logfileName + 'belief_uniform_g3db_instances_train2_reward100_penalty10/'
                            logfileName = logfileName + 'use_discretized_data/use_binary_touch'
                            logfileName = logfileName + '/use_wider_workspace/use_probabilistic_neighbour_step'
                            logfileName = logfileName + '/use_discrete_observation_in_update/'
                            logfileName = logfileName + 'use_discrete_observation_in_step/'
                            logfileName = logfileName + 'use_weighted_belief/'
                            logfileName = logfileName + 'use_classifier/kmeans-extra-data_label_3_20180419-123116/horizon90/t'
                            logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                            #print i
                            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver7/weighted_5')

    elif fileName in ['vrep/version23']:
        object_list = get_grasping_object_name_list('cylinders_train')
        for i in range(0,4000):
            for t in ['5']:
                for scenario in ['320']:
                   for object in object_list:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/'
                        logfileName = logfileName + 'vrep_scene_ver7/multiObjectType/'
                        logfileName = logfileName + 'belief_uniform_cylinder_7_8_9_reward100_penalty10/'
                        logfileName = logfileName + 'use_discretized_data/use_binary_touch/'
                        logfileName = logfileName + 'use_probabilistic_neighbour_step/'
                        logfileName = logfileName + 'use_discrete_observation_in_update/'
                        logfileName = logfileName + 'use_discrete_observation_in_step/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver7')

    elif fileName in ['vrep/version22']:
        object_list = get_grasping_object_name_list('g3db_instances_train2_version7')
        for i in range(0,500):
            for t in ['5']:
                for scenario in ['160']:
                   for object in object_list:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/'
                        logfileName = logfileName + 'vrep_scene_ver7/multiObjectType/'
                        logfileName = logfileName + 'belief_uniform_g3db_instances_train2_reward100_penalty10/'
                        logfileName = logfileName + 'use_discretized_data/use_binary_touch'
                        logfileName = logfileName + '/use_wider_workspace/use_probabilistic_neighbour_step_temperature_bug'
                        logfileName = logfileName + '/use_discrete_observation_in_update/'
                        logfileName = logfileName + 'use_discrete_observation_in_step/'
                        logfileName = logfileName + 'use_weighted_belief/'
                        logfileName = logfileName + 'use_classifier/kmeans-extra-data_label_4_20180504-200455/horizon90/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver7/weighted_4')

    elif fileName in ['vrep/version21']:
        object_list = get_grasping_object_name_list('g3db_instances_train2_version7')
        for i in range(0,500):
            for t in ['5']:
                for scenario in ['80']:
                   for object in object_list:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/'
                        logfileName = logfileName + 'vrep_scene_ver7/multiObjectType/'
                        logfileName = logfileName + 'belief_uniform_g3db_instances_train2_reward100_penalty10/'
                        logfileName = logfileName + 'use_discretized_data/use_binary_touch'
                        logfileName = logfileName + '/use_wider_workspace/use_probabilistic_neighbour_step_temperature_bug'
                        logfileName = logfileName + '/use_discrete_observation_in_update/'
                        logfileName = logfileName + 'use_discrete_observation_in_step/'
                        logfileName = logfileName + 'use_weighted_belief/'
                        logfileName = logfileName + 'use_classifier/kmeans-extra-data_label_3_20180419-123116/horizon90/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver7/weighted_5')

    elif fileName in ['vrep/version20']:
        object_list = get_grasping_object_name_list('g3db_instances_train1_version7')
        for i in range(0,500):
            for t in ['5']:
                for scenario in ['160']:
                   for object in object_list:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/'
                        logfileName = logfileName + 'vrep_scene_ver7/multiObjectType/'
                        logfileName = logfileName + 'belief_uniform_g3db_instances_train1_reward100_penalty10/'
                        logfileName = logfileName + 'use_discretized_data/use_weighted_belief/'
                        logfileName = logfileName + 'use_classifier/kmeans_20180215-115011/horizon90/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver7/weighted_3')

    elif fileName in ['vrep/version19']:
        object_list = get_grasping_object_name_list('g3db_instances_train1_version7')
        for i in range(0,243):
            for t in ['5']:
                for scenario in ['160']:
                   for object in object_list:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/'
                        logfileName = logfileName + 'vrep_scene_ver7/multiObjectType/'
                        logfileName = logfileName + 'belief_uniform_g3db_instances_train1_reward100_penalty10/'
                        logfileName = logfileName + 'use_discretized_data/fixed_distribution/horizon90/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver7')
    elif fileName in ['vrep/version18']:
        object_list = get_grasping_object_name_list('cylinders_train')
        for i in range(0,4000):
            for t in ['5']:
                for scenario in ['80']:
                   for object in object_list:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/'
                        logfileName = logfileName + 'vrep_scene_ver7/multiObjectType/'
                        logfileName = logfileName + 'belief_uniform_cylinder_7_8_9_reward100_penalty10/'
                        logfileName = logfileName + 'use_discretized_data/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver7')
    elif fileName in ['vrep/version17']:
        #data with number of objects = 10 bug
        object_list = get_grasping_object_name_list('g3db_instances_train1')
        for i in range(0,1000):
            for t in ['5']:
                for scenario in ['40']:
                   for object in object_list:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/'
                        logfileName = logfileName + 'vrep_scene_ver6/multiObjectType/'
                        logfileName = logfileName + 'belief_uniform_g3db_instances_train1_reward100_penalty10/'
                        logfileName = logfileName + 'use_discretized_data/use_weighted_belief/horizon90/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver5/weighted_15')

    elif fileName in ['vrep/version16']:
        #Had point cloud bug
        object_list = get_grasping_object_name_list('g3db_instances_train1')
        for i in range(0,2000):
            for t in ['5']:
                for scenario in ['320']:
                   for object in object_list:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/'
                        logfileName = logfileName + 'vrep_scene_ver6/multiObjectType/'
                        logfileName = logfileName + 'belief_uniform_g3db_instances_train1_reward100_penalty10/'
                        logfileName = logfileName + 'use_discretized_data/use_weighted_belief/horizon90/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver5/weighted_15')

    elif fileName in ['vrep/version14','vrep/version15']:
        #version 14 and 15 differ only in training params
        object_list = get_grasping_object_name_list('cylinders_train')
        for i in range(0,4000):
            for t in ['5']:
                for scenario in ['40']:
                   for object in object_list:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/'
                        logfileName = logfileName + 'vrep_scene_ver6/multiObjectType/'
                        logfileName = logfileName + 'belief_uniform_cylinder_7_8_9_reward100_penalty10/'
                        logfileName = logfileName + 'use_discretized_data/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver5')
    elif fileName == 'vrep/version13':
        object_list = get_grasping_object_name_list('coffee_yogurt_cup')
        for i in range(0,4000):
            for t in ['5']:
                for scenario in ['80']:
                   for object in object_list:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_uniform_g3db_1_84_reward100_penalty10/icp_score_weighted/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver5/weighted_2')

    elif fileName == 'vrep/version12':
        for i in range(0,4000):
            for t in ['5']:
                for scenario in ['320']:
                   for object in ['7cm', '8cm', '9cm']:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_uniform_cylinder_7_8_9_reward100_penalty10/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep/ver5')
    elif fileName == 'vrep/version11':
        for i in range(0,4000):
            for t in ['5']:
                for scenario in ['320']:
                   for object in ['7cm', '8cm', '9cm']:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_uniform_cylinder_7_8_9_reward100_penalty10/t'
                        logfileName = logfileName + t + '_n' + scenario + '/Table_scene_'+ object + '_belief_uniform_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')

    elif fileName in ['vrep/version9', 'vrep/version10']:
        for i in range(0,500):
            for t in ['5']:
                for scenario in ['40', '80', '160']:
                    for object in ['7cm', '8cm', '9cm']:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty100/t' + t + '_n' + scenario + '/Table_scene_low_friction_'+ object + '_cylinder_belief_gaussian_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')

    elif fileName == 'vrep/version8':
        for i in range(0,500):
            for t in ['5']:
                for scenario in ['40', '80', '160']:
                    for object in ['7cm', '8cm', '9cm']:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t' + t + '_n' + scenario + '/Table_scene_low_friction_'+ object + '_cylinder_belief_gaussian_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')

    elif fileName == 'vrep/version8_pick_bug':
        for i in range(0,500):
            for t in ['5']:
                for scenario in ['40', '80', '160']:
                    for object in ['7cm', '8cm', '9cm']:
                        logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t' + t + '_n' + scenario + '/Table_scene_low_friction_'+ object + '_cylinder_belief_gaussian_with_state_in_t' + t + '_n' + scenario + '_trial_' + repr(i) +'.log'
                        #print i
                        seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')


    elif fileName == 'vrep/version7':
        for i in range(0,1000):
            logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/t5_n40/Table_scene_low_friction_9cm_cylinder_belief_gaussian_with_state_in_t5_n40_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/t5_n80/Table_scene_low_friction_9cm_cylinder_belief_gaussian_with_state_in_t5_n80_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/t5_n160/Table_scene_low_friction_9cm_cylinder_belief_gaussian_with_state_in_t5_n160_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/low_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/t5_n320/Table_scene_low_friction_9cm_cylinder_belief_gaussian_with_state_in_t5_n320_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
    elif fileName == 'vrep/version6':
        for i in range(0,500):
            #logfileName = '/home/neha/WORK_FOLDER/neha_github/apc/rosmake_ws/despot_vrep_glue/results/despot_logs/VrepData_gaussian_belief_with_state_in_belief_t5_n10_trial_' + repr(i) +'.log'
            ################################################### v6 begins#################################################
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n160/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t5_n160_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n160/TableScene_cylinder_8cm_gaussian_belief_with_state_in_belief_t5_n160_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n160/TableScene_cylinder_7cm_gaussian_belief_with_state_in_belief_t5_n160_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n320/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t5_n320_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n320/TableScene_cylinder_8cm_gaussian_belief_with_state_in_belief_t5_n320_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n320/TableScene_cylinder_7cm_gaussian_belief_with_state_in_belief_t5_n320_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n640/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t5_n640_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n640/TableScene_cylinder_8cm_gaussian_belief_with_state_in_belief_t5_n640_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n640/TableScene_cylinder_7cm_gaussian_belief_with_state_in_belief_t5_n640_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            ##############################################################v6 ends##########################################################################
    elif fileName in ['vrep/verion4', 'vrep/version5']:
        for i in range(0,1000):
            #####################v4 begins####################################################################
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/t5_n80/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t5_n80_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/t5_n40/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t5_n40_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/t5_n20/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t5_n20_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            logfileName = '../../grasping_ros_mico/results/despot_logs/high_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/t1_n20/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t1_n20_trial_' + repr(i) +'.log'
            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
            ####################v4 ends#################################################################

        #for i in range(0,400):
        #    for round_no in range(0,4):
        #        logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + repr(i) + '_multi_runs_t10_n10_obs_prob_change_particles_as_state_4objects.log'
        #        seqs = seqs + parse_file(logfileName, belief_type, True, round_no, 'toy')
                #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/graspingV3_state_' + repr(i) + '_t20_obs_prob_change_particles_as_state_4objects.log'
                #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + repr(i) + '_t10_n10_obs_prob_change_particles_as_state_4objects.log'
                #seqs = seqs + parse_file(logfileName, True, True)
                #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_same_objects/version6/state_' + repr(i) + '.log'
                #seqs = seqs + parse_file(logfileName, False, True)
                #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_same_objects/version7/dagger_data/graspingV4_state_' + repr(i) + '_t10_n10_obs_prob_change_particles_as_state_4objects.log'
                #seqs = seqs + parse_file(logfileName, False, True)
                #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_same_objects/version7/dagger_data/graspingV4_state_' + repr(i) + '_t1_n10_obs_prob_change_particles_as_state_4objects.log'
                #seqs = seqs + parse_file(logfileName, False, True)


    else:
       # seqs = seqs + parse_file(fileName, '', isTraining, -1, 'toy')
         seqs = seqs + parse_file(fileName, '', isTraining, -1, 'vrep')
         #print seqs
    return seqs


def test_parsing_methods(filename):

    fullData1 =  ParseLogFile(filename, '', -1,'toy', 1).getFullDataWithoutBelief()
    seqs1 = []
    process_full_data(fullData1,seqs1)


    fullData2 =  ParseLogFile(filename, '', -1, 'toy', 2).getFullDataWithoutBelief()
    seqs2 = []
    process_full_data(fullData2,seqs2)

    assert len(seqs1[0]) == len(seqs2[0])
    for i in range(0,len(seqs1[0])):
        assert seqs1[0][i][0] == seqs2[0][i][0]
        for j in range(0,len(seqs1[0][i][1])):
            assert seqs1[0][i][1][j] == seqs2[0][i][1][j]
    print seqs2
    print seqs1

def test_parser(filename = 'test'):
    seqs = parse(filename, '', False)
    action_seqs = []
    for seq in seqs :
        action_seq = [x[1] for x in seq]
        action_seqs = action_seqs + [action_seq]
    print action_seqs
    print len(action_seqs)

def main():
    i = 0
    import sys
    #filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + repr(i) + '_multi_runs_t10_n10_obs_prob_change_particles_as_state_4objects.log'
    #filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/'

    #filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/TableScene_cylinder_10cm_gaussian_belief_with_state_in_belief_t5_n10_trial_0.log'
    filename = 'test'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    #test_parser(filename)
    #test_parsing_methods(filename)
    test_parser(filename)

if __name__=="__main__":
    main()
