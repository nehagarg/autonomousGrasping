
import re
import sys


def get_mean_std_for_pocman(filename):
    import numpy as np
    import math
    a = []
    sum2 = 0.0
    with open(filename, 'r') as f:
        for line in f:
          a.append(int(line)) 
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
                    seqs[-1].append((act,obs)) 
                    parse_act_obs = False
            else: #Original despot trace
            
                regular_expression = 'history and root with action (\d+), observation (\d+)'
                act_obs_found = re.search(regular_expression, line)
                if(act_obs_found):
                    act = int(act_obs_found.group(1))
                    obs = int(act_obs_found.group(2))
                    seqs[-1].append((act,obs))
                    parse_act_obs = False

    return seqs    
    


def parse(filename=None, round = -1):
    if filename is None:
        filename = '../pocman_t10_100runs_access9.log'
	
    	return parse_pocman_trace(filename)
    else:
	return parse_pocman_trace(filename, -1, False)

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

    
