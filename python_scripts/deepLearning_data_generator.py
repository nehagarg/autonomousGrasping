
from log_file_parser import ParseLogFile
from adaboost_data_generator import get_label_string



def createActionHash():
    action_string = get_label_string()
    action_string_array = action_string[:-2].split(", ")
    action_string_hash = {}
    for i in range(0,len(action_string_array)):
        action_string_hash[action_string_array[i]] = i
    return action_string_hash

def parse_file(file_name, isPlanningLog = True, isTraining = True, round_no = 0):
    action_string_hash = createActionHash()
    #print action_string_hash
    seqs = []
    if not isTraining:
        round_no = -1
    fullData =  ParseLogFile(file_name, isPlanningLog, round_no).getFullDataWithoutBelief()
    #print fullData
    num_steps = len(fullData['stepInfo'])
    if (num_steps < 90 or not isTraining):
        seq = []
        j_range = len(fullData['stepInfo'])
        #if not isTraining:
        #    j_range = j_range - 1
        for j in range(0,j_range):
            act = 11
            obs = None
            if 'action' in fullData['stepInfo'][j]:
                act = action_string_hash[("").join(fullData['stepInfo'][j]['action'][:-1].split(" "))]
            if 'obs' in fullData['stepInfo'][j]:
                obs = fullData['stepInfo'][j]['obs'].sensor_obs
                obs.append(fullData['stepInfo'][j]['obs'].gripper_l_obs)
                obs.append(fullData['stepInfo'][j]['obs'].gripper_r_obs)
                obs.append(fullData['stepInfo'][j]['obs'].x_w_obs)
                obs.append(fullData['stepInfo'][j]['obs'].y_w_obs)
            seq.append((act,obs))
        seqs.append(seq)
    return seqs
 
def parse(fileName, isPlanningLog = False, isTraining = False):
    seqs = []
    if fileName is None:
        for i in range(0,400):
            for round_no in range(0,4):
                logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + repr(i) + '_multi_runs_t10_n10_obs_prob_change_particles_as_state_4objects.log'
                seqs = seqs + parse_file(logfileName, True, True, round_no)
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
        seqs = seqs + parse_file(fileName, isPlanningLog, isTraining)
    return seqs

def main():
    i = 0
    filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + repr(i) + '_multi_runs_t10_n10_obs_prob_change_particles_as_state_4objects.log'

    seqs = parse(filename)
    print seqs

if __name__=="__main__":
    main()

    
