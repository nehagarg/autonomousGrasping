
import os
import getopt
import yaml
import sys
try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper
    
def generate_commands(yaml_config):
    all_commands = []
    
    solver = yaml_config['solver']
    config_file = yaml_config['config_file']
    output_dir = yaml_config['output_dir'] 
    file_name = yaml_config['file_name']
    planning_time = yaml_config['planning_time'] 
    number_scenarios = yaml_config['number_scenarios'] 
    horizon = yaml_config['horizon'] 
    belief_type = yaml_config['belief_type'] 
    additional_params = yaml_config['additional_params']
    begin_index = yaml_config['begin_index'] 
    end_index = yaml_config['end_index']
    
    for i in range(begin_index, end_index):
        command = "./bin/despot_without_display -v3 -t " + repr(planning_time) + " -n "
        command = command + repr(number_scenarios)+ ' -s ' + repr(horizon) + ' --solver=' + solver
        command = command + ' ' + additional_params + ' -m ' + config_file + ' --belief=' + belief_type + ' > '
        command = command + output_dir
        command = command + "/" + file_name + '_trial_' + repr(i) + '.log 2>&1' 
        all_commands.append(command)
    return all_commands

def get_default_params(yaml_file = None):
    ans = {}
    if yaml_file is not None:
        with open(yaml_file,'r') as stream:
            ans = yaml.load(stream)
    else:
        ans['solver'] = 'DESPOT'
        ans['config_file'] = 'config_files/VrepDataInterface.yaml'
        ans['planning_time'] = 1
        ans['number_scenarios'] = 5
        ans['horizon'] = 50
        ans['belief_type'] = 'GAUSSIAN_WITH_STATE_IN'
        ans['additional_params'] = '--number=-1 -l CAP'
        ans['begin_index'] = 0
        ans['end_index'] = 1000
        ans['file_name_prefix'] = ''
    

    return ans

def generate_params_file(file_name):
    ans = {}
    ans['solver'] = 'DESPOT'
    ans['config_file'] = 'config_files/VrepDataInterface.yaml'
    ans['planning_time'] = 1
    ans['number_scenarios'] = 5
    ans['horizon'] = 50
    ans['belief_type'] = 'GAUSSIAN_WITH_STATE_IN'
    ans['additional_params'] = '--number=-1 -l CAP'
    ans['begin_index'] = 0
    ans['end_index'] = 1000
    ans['file_name_prefix'] = ''
    
    if file_name == 'data_model_9cm_combined_automatic.yaml':
        ans['solver'] = 'LEARNINGPLANNING'
        ans['config_file'] = 'config_files/VrepDataInterface_v4_automatic.yaml'
        ans['file_name_prefix'] = 'Table_scene_9cm_cylinder_v4_automatic'
        ans['output_dir'] = './results/despot_logs/high_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/learning/version4/combined_1'

    if file_name == 'data_model_9cm_despot_low_friction.yaml':
        ans['config_file'] = 'config_files/VrepDataInterface_low_friction.yaml'
        ans['file_name_prefix'] = 'Table_scene_9cm_cylinder'
        ans['output_dir'] = './results/despot_logs/low_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10'

    output = yaml.dump(ans, Dumper = Dumper)
    f = open(file_name, 'w')
    f.write(output)
    
    
if __name__ == '__main__':
    
    opts, args = getopt.getopt(sys.argv[1:],"hegt:n:d:y:",["dir=","yaml_file="])
    output_dir = None
    yaml_file = None
    execute_command = False
    genarate_yaml = False
    planning_time = None
    number_scenarios = None
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'experiment_v2.py -d <directory_name> -f <file_prefix>'
         sys.exit()
      elif opt == '-e':
         execute_command = True
      elif opt == '-g':
         genarate_yaml = True
      elif opt == '-t':
         planning_time = int(arg)
      elif opt == '-n':
         number_scenarios = int(arg)
      elif opt in ("-d", "--dir"):
         output_dir = arg
      elif opt in ("-y", "--yaml"):
         yaml_file = arg

    
    if genarate_yaml:
        generate_params_file(yaml_file)
        sys.exit()
        
    ans = get_default_params(yaml_file)
    if output_dir is not None:
        ans['output_dir'] = output_dir
        
    if ans['solver'] == 'DEEPLEARNING':
        ans['file_name'] = ans['file_name_prefix']
    else:
        
        ans['planning_time'] = planning_time
        ans['number_scenarios'] = number_scenarios
        ans['output_dir'] = ans['output_dir'] + '/t' + repr(ans['planning_time']) + '_n' + repr(ans['number_scenarios'])
        ans['file_name'] = ans['file_name_prefix'] + '_belief_' + ans['belief_type'].lower() + '_t' + repr(ans['planning_time']) + '_n' + repr(ans['number_scenarios'])
   
    all_commands = generate_commands(ans)
    for command in all_commands:
        print command
        if execute_command:
            print "Executing...."
            os.system(command)
        
      
    