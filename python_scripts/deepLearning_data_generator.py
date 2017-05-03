
from log_file_parser import ParseLogFile
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

def process_full_data(fullData,seqs, state_type = 'toy', isTraining = True):
    action_string_hash = createActionHash(state_type)
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
                obs = fullData['stepInfo'][j]['obs'].convert_to_array()
                #obs = fullData['stepInfo'][j]['obs'].sensor_obs
                #obs.append(fullData['stepInfo'][j]['obs'].gripper_l_obs)
                #obs.append(fullData['stepInfo'][j]['obs'].gripper_r_obs)
                #obs.append(fullData['stepInfo'][j]['obs'].x_w_obs)
                #obs.append(fullData['stepInfo'][j]['obs'].y_w_obs)
            seq.append((act,obs))
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
        
        for i in range(0,20):
            #logfileName = '/home/neha/WORK_FOLDER/neha_github/apc/rosmake_ws/despot_vrep_glue/results/despot_logs/VrepData_gaussian_belief_with_state_in_belief_t5_n10_trial_' + repr(i) +'.log'
            #logfileName = '/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/separate_close_reward/singleObjectType/cylinder_9cm_reward100_penalty10/t1_n320_withoutLCAP/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t1_n320_trial_' + repr(i) +'.log'
            #logfileName = '/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/low_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/t5_n80/Table_scene_low_friction_9cm_cylinder_belief_gaussian_with_state_in_t5_n80_trial_' + repr(i) +'.log'
            logfileName = '/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/low_friction_table/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n20/Table_scene_low_friction_9cm_cylinder_belief_gaussian_with_state_in_t5_n20_trial_' + repr(i) +'.log'
            #logfileName = '/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10/t5_n160/TableScene_cylinder_9cm_gaussian_belief_with_state_in_belief_t5_n160_trial_' + repr(i) +'.log'

            #print i
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
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
    
    elif fileName == 'vrep/version8':
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

def test_parser(filename = None):
    seqs = parse('test', '', True)
    action_seqs = []
    for seq in seqs :
        action_seq = [x[0] for x in seq]
        action_seqs = action_seqs + [action_seq]
    print action_seqs
    
def main():
    i = 0
    #filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + repr(i) + '_multi_runs_t10_n10_obs_prob_change_particles_as_state_4objects.log'
    #filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/'
    
    filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/TableScene_cylinder_10cm_gaussian_belief_with_state_in_belief_t5_n10_trial_0.log'
    
    #test_parser(filename)
    #test_parsing_methods(filename)
    test_parser()

if __name__=="__main__":
    main()

    
