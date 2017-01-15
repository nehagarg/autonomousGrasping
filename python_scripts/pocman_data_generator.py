
import re

def parse(filename, round=-1):
    f = open(filename, 'r')
    seqs = []
    parse_act_obs = False
    parse_round = True
    while True:
        line = f.readline()
        if not line:
                break;
        if parse_round:
            regular_expression = 'Round (\d+) Step (\d+)'
            step_start = re.search(regular_expression, line)
            if step_start:
               parse_act_obs = True
               parse_round = False
               round_no = int(step_start.group(1))
               step_no = int(step_start.group(2))
               if round==round_no:
                   break;
               if step_no == 0:
                   seqs.append([])
        if parse_act_obs:
            regular_expression = 'history and root with action (\d+), observation (\d+)'
            act_obs_found = re.search(regular_expression, line)
            if(act_obs_found):
                act = int(act_obs_found.group(1))
                obs = int(act_obs_found.group(2))
                seqs[-1].append((act,obs))
                parse_act_obs = False
                parse_round = True
    return seqs    
    




def test_parser(filename):
    seqs = parse(filename, 2)
    print seqs
    
def main():
    i = 0
    #filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/4_objects_obs_prob_change_particles_as_state/graspingV4_state_' + repr(i) + '_multi_runs_t10_n10_obs_prob_change_particles_as_state_4objects.log'
    #filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/'
    
    filename = 'pocman_t10_100runs_access9.log'
    
    test_parser(filename)
    #test_parsing_methods(filename)
    

if __name__=="__main__":
    main()

    
