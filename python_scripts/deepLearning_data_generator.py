
from log_file_parser import ParseLogFile
from adaboost_data_generator import get_label_string



def createActionHash(state_type = 'toy'):
    action_string_hash = {}
    if state_type == 'toy':
        action_string = get_label_string()
        action_string_array = action_string[:-2].split(", ")        
        for i in range(0,len(action_string_array)):
            action_string_hash[action_string_array[i]] = i
    if state_type == 'vrep':
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
    return action_string_hash

def process_full_data(fullData,seqs, state_type = 'toy', isTraining = True):
    action_string_hash = createActionHash(state_type)
    #print action_string_hash
    #print fullData
    num_steps = len(fullData['stepInfo'])
    num_actions = len(action_string_hash.keys())
    successTraj = False
    if len(fullData['stepInfo']) > 0 and fullData['stepInfo'][-1]['reward'] == 20:
        successTraj = True
    if (successTraj or not isTraining):
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
    if fileName is None:
        for i in range(0,100):
            logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/ros/apc/rosmake_ws/despot_vrep_glue/results/despot_logs/VrepData_single_particle_belief_t5_n1_state_'+ repr(i) +'.log'
            seqs = seqs + parse_file(logfileName, belief_type, True, 0, 'vrep')
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

def test_parser(filename):
    seqs = parse(filename)
    print seqs
    
def main():
    i = 0
    filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + repr(i) + '_multi_runs_t10_n10_obs_prob_change_particles_as_state_4objects.log'
    #filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/'
    
    
    #test_parser(filename)
    test_parsing_methods(filename)
    

if __name__=="__main__":
    main()

    
