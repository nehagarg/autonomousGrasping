
import re
import sys
import os
from plot_despot_results import generate_reward_file, get_mean_std_for_numbers_in_file, get_mean_std_for_array
import operator

def get_mean_std_for_pocman(filename):
    import numpy as np
    import math
    a = []
    sum2 = 0.0
    with open(filename, 'r') as f:
        for line in f:
          a.append(float(line)) 
          sum2 = sum2+ a[-1]*a[-1]
    mean = np.mean(a)
    std = np.std(a)
    std2 = math.sqrt((sum2/(len(a)*len(a))) - (mean*mean/len(a)))
    print mean
    print std
    print std2
    
    
def parse_pocman_trace(filename, round=-1, isTraining = True):
    numeric_const_pattern = r"""
    [-+]? # optional sign
    (?:
       (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
           |
            (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
       )
     # followed by optional exponent part if desired
     (?: [Ee] [+-]? \d+ ) ?
     """
    rx =  re.compile(numeric_const_pattern, re.VERBOSE)
    
    f = open(filename, 'r')
    seqs = []
    parse_act_obs = False
    parse_round = True
    reward = 0
    act = 5
    obs = None
    while True:
        line = f.readline()
        if not line:
            if isTraining and reward < -1: #Criteria for selecting trace in full pocman
                    print "Removing Seq " + repr(round_no - 1)
                    del(seqs[-1])
	    if not isTraining:
		seqs[-1].append((5,None))
            break;
            
        regular_expression = 'Round (\d+) Step (\d+)'
        step_start = re.search(regular_expression, line)
            
        if step_start:
            parse_act_obs = True
            round_no = int(step_start.group(1))
            step_no = int(step_start.group(2))
            if round==round_no:
                break;
            if step_no == 0:
                #print round_no
                if round_no > 0:
                    #print len(seqs[-1])
                    #print reward
                    if isTraining and reward < -1: #Criteria for selecting trace in full pocman
                        #print "Removing Seq " + repr(round_no - 1)
                        del(seqs[-1])
                seqs.append([])
                
        if parse_act_obs:
            if re.search('- Reward', line):
                values = rx.findall(line)
                reward = int(values[0])

            if not isTraining: #Slightly different file format
                if re.search('- Action', line):
                    values = rx.findall(line)
                    act = int(values[0])
                if re.search('- Observation', line):
                    values = rx.findall(line)
                    obs = int(values[0])
                    seqs[-1].append((act,[obs])) 
                    parse_act_obs = False
            else: #Original despot trace
            
                regular_expression = 'history and root with action (\d+), observation (\d+)'
                act_obs_found = re.search(regular_expression, line)
                if(act_obs_found):
                    act = int(act_obs_found.group(1))
                    obs = int(act_obs_found.group(2))
                    seqs[-1].append((act,[obs]))
                    parse_act_obs = False

    return seqs    

def get_high_reward_filenames(filenames):
    #Assuming each file contains one run
    reward_file_name = repr(os.getpid()) + '.txt'
    reward_dict = {}
    for filename in filenames:
        generate_reward_file('./', [filename], 1, reward_file_name)
        (reward,_,_) = get_mean_std_for_numbers_in_file(reward_file_name)
        reward_dict[filename] = reward
    sorted_x = sorted(reward_dict.items(), key=operator.itemgetter(1), reverse = True)
    print len(sorted_x)
    
    print sorted_x[0:5]
    print sorted_x[(len(sorted_x)/2) - 5:len(sorted_x)/2]
    high_reward_filenames = []
    high_rewards = []
    for filename,reward in sorted_x[0:len(sorted_x)/2]:
	high_reward_filenames.append(filename)
        high_rewards.append(reward)
    (high_reward_mean,_,_) = get_mean_std_for_array(high_rewards)
    print high_reward_mean
    return high_reward_filenames
        

def parse(filename=None, round = -1):
    seqs = []
    if filename is None:
        filename = '../pocman_t10_100runs_access9.log'
	
    	seqs = seqs + parse_pocman_trace(filename)
    elif filename == 'pocman/version2':
        seqs = []
        filenames = []
        for i in range(0,1000):
            for t in ['1','5','10']:
                for n in ['100', '500', '1000']:
                    logfilename = '../../pocman/results/t' + t + '_n' + n + '/full_pocman_t' + t + '_n' + n + '_trial_' + repr(i) + '.log'
                    filenames.append(logfilename)
                    logfilename = '../../pocman/results/learning/version1/combined_2/t' + t + '_n' + n + '/full_pocman__belief_default_t' + t + '_n' + n + '_trial_' + repr(i) + '.log'
                    
            logfilename = '../../pocman/results/learning/version1/full_pocman__trial_' + repr(i) + '.log'
            filenames.append(logfilename)
        high_reward_filenames = get_high_reward_filenames(filenames)
        for logfilename in high_reward_filenames:
            seqs = seqs + parse_pocman_trace(logfilename, round, True)            
                    
    elif filename == 'pocman/version1':
        seqs = []
        filenames = []
        for i in range(0,1000):
            for t in ['1','5','10']:
                for n in ['100', '500', '1000']:
                    logfilename = '../../pocman/results/t' + t + '_n' + n + '/full_pocman_t' + t + '_n' + n + '_trial_' + repr(i) + '.log'
                    filenames.append(logfilename)
        high_reward_filenames = get_high_reward_filenames(filenames)
        for logfilename in high_reward_filenames:
            seqs = seqs + parse_pocman_trace(logfilename, round, True)
        
            
            
    else:
	seqs = seqs + parse_pocman_trace(filename, -1, False)
    return seqs

def test_parser(filename):
    seqs = parse_pocman_trace(filename, 11)
    print seqs
    print len(seqs)
    
def main():
    i = 0
    #filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + repr(i) + '_multi_runs_t10_n10_obs_prob_change_particles_as_state_4objects.log'
    #filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/'
    
    #filename = 'pocman_t10_100runs_access9.log'
    filename = 't10_n100_rewards.txt'
    if sys.argv[1]:
        filename = sys.argv[1]
    #test_parser(filename)
    #test_parsing_methods(filename)
    get_mean_std_for_pocman(filename)
    

if __name__=="__main__":
    main()

    
