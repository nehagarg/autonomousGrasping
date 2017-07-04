
import os
import getopt
import yaml
import sys
try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper
from generate_grasping_ros_mico_yaml_config_file import get_learning_version_from_filename, get_switching_threshold, LEARNED_MODEL_NAME
import re
    
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
    problem_type = yaml_config['problem_type']
    modified_additional_params = additional_params
    for i in range(begin_index, end_index):
        #if problem_type == 'graspingV4':
        #    modified_additional_params = additional_params + repr(i)
        if additional_params.endswith('='):
             modified_additional_params = additional_params + repr(i)
        command = "./bin/" + problem_type + " -v3 -t " + repr(planning_time) + " -n "
        command = command + repr(number_scenarios)+ ' -s ' + repr(horizon) + ' --solver=' + solver
        command = command + ' ' + modified_additional_params + ' -m ' + config_file + ' --belief=' + belief_type + ' > '
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

def get_config_file_name_from_experiment_file(file_name):
    config_file_name = 'config_files/'
    if 'data-model' in file_name:
        config_file_name = config_file_name + 'VrepDataInterface_'
    elif 'vrep-model' in file_name:
        config_file_name = config_file_name + 'VrepInterface_'
    
    
def generate_params_file(file_name, problem_type):
    ans = {}
    ans['solver'] = 'DESPOT'
    ans['planning_time'] = 1
    ans['number_scenarios'] = 5
    ans['horizon'] = 50
    ans['begin_index'] = 0
    ans['end_index'] = 1000
    ans['file_name_prefix'] = ''
    ans['config_file'] = 'config_files/' + file_name.replace('learning', 'combined_0').replace('despot', 'combined_0')
    
    if problem_type == 'despot_without_display':
        #ans['config_file'] = 'config_files/VrepDataInterface.yaml'
        ans['belief_type'] = 'GAUSSIAN_WITH_STATE_IN'
        ans['additional_params'] = '--number=-1 -l CAP'
    if problem_type == 'graspingV4':
        
        ans['belief_type'] = 'DEFAULT'
        ans['additional_params'] = '--number='
        ans['output_dir'] = './results/despot_logs'
        ans['end_index'] = 400
        ans['horizon'] = 90
        
        
        for filetype in ['train', 'test']:
            if filetype in file_name:
                ans['file_name_prefix'] = 'Toy_' + filetype
        

    if problem_type == 'pocman':
        #ans['config_file'] = 'config_files/' + file_name.replace('learning', 'combined_0')
        ans['belief_type'] = 'DEFAULT'
        ans['additional_params'] = ''
        ans['output_dir'] = './results'
        ans['horizon'] = 90
        ans['file_name_prefix'] = 'full_pocman_'
    
    
    learning_version, model_name = get_learning_version_from_filename(file_name)
    
    if 'combined' in file_name:
        ans['solver'] = 'LEARNINGPLANNING'
        if 'output_dir' in ans:
            m = re.search('combined_[0-9]+', file_name)
            switching_threshold = get_switching_threshold(file_name)
            switching_threshold_string = ""
            if switching_threshold != 10:
                switching_threshold_string = "-" + repr(switching_threshold)
            ans['output_dir'] = ans['output_dir'] + "/learning/version" + learning_version + '/'+ m.group() + switching_threshold_string
    if 'learning' in file_name:
        ans['solver'] = 'DEEPLEARNING'
        if 'output_dir' in ans:
            ans['output_dir'] = ans['output_dir'] + "/learning/version" + learning_version
        
    for filetype in ['combined_0', 'combined_1', 'combined_2', 'combined_0-15', 'combined_0-20']:
        for interface_type in ["vrep_model", "data_model"]:
            file_prefix = interface_type + "_9cm_low_friction_"
            if file_name == file_prefix + filetype + ".yaml" :
                ans = get_default_params(file_prefix + "learning.yaml")
                ans['output_dir'] = ans['output_dir'] + "/" + filetype
                ans['config_file'] = (ans['config_file'].split('.'))[0] + '_' + filetype + ".yaml"
            for object_type in ['7cm', '8cm', '9cm', '75mm', '85mm']:
                file_prefix =  interface_type + "_multi_object_" + object_type + "_low_friction_"
                if file_name == file_prefix  + filetype + '.yaml':
                    ans = get_default_params(file_prefix + 'learning.yaml')
                    ans['output_dir'] = ans['output_dir'] + "/" + filetype
                    ans['config_file'] = (ans['config_file'].split('.'))[0] + '_' + filetype + ".yaml"
    
    if 'combined' in file_name:
        ans['solver'] = 'LEARNINGPLANNING'
    
    if 'learning' in file_name:
        ans['solver'] = 'DEEPLEARNING'

    if 'fixed_distribution' in file_name:
        ans = get_default_params(file_name.replace('_fixed_distribution', '') )
        ans['additional_params'] = '-l CAP --number='
        if('simulator' in ans['output_dir']):
            ans['output_dir'] = ans['output_dir'].replace("simulator","simulator/fixed_distribution")
        else:
            ans['output_dir'] = ans['output_dir'].replace("penalty10","penalty10/fixed_distribution")
        ans['end_index'] = 245
                
    if file_name == 'data_model_9cm_combined_automatic.yaml':
        ans['config_file'] = 'config_files/VrepDataInterface_v4_automatic.yaml'
        ans['file_name_prefix'] = 'Table_scene_9cm_cylinder_v4_automatic'
        ans['output_dir'] = './results/despot_logs/high_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10/learning/version4/combined_1'

    if file_name == 'data_model_9cm_low_friction.yaml':
        ans['config_file'] = 'config_files/VrepDataInterface_low_friction.yaml'
        ans['file_name_prefix'] = 'Table_scene_9cm_cylinder'
        ans['output_dir'] = './results/despot_logs/low_friction_table/singleObjectType/cylinder_9cm_reward100_penalty10'

    output = yaml.dump(ans, Dumper = Dumper)
    f = open(file_name, 'w')
    f.write(output)
    
def generate_fixed_distribution_commands():
    for filetype in ['', '_learning', '_combined_0', '_combined_1', '_combined_2', '_combined_0-15', '_combined_0-20']:
        for interface_type in ["vrep_model_fixed_distribution", "data_model_fixed_distribution"]:
            generate_params_file(interface_type + "_9cm_low_friction" + filetype + ".yaml", 'despot_without_display')
            for object_type in ['7cm', '8cm', '9cm', '75mm', '85mm']:
                  generate_params_file(interface_type + "_multi_object_" + object_type + "_low_friction" + filetype + ".yaml", 'despot_without_display')       

    
def main():
    global LEARNED_MODEL_NAME
    opts, args = getopt.getopt(sys.argv[1:],"hegt:n:d:s:c:p:m:",["dir="])
    output_dir = None
    yaml_file = None
    execute_command = False
    genarate_yaml = False
    planning_time = None
    number_scenarios = None
    start_index = None
    end_index = None
    problem_type = None
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'experiment_v2.py -d <directory_name> yaml_file'
         sys.exit()
      elif opt == '-e':
         execute_command = True
      elif opt == '-g':
         genarate_yaml = True
      elif opt == '-t':
         planning_time = int(arg)
      elif opt == '-n':
         number_scenarios = int(arg)
      elif opt == '-s':
         start_index = int(arg)
      elif opt == '-c':
         end_index = int(arg)
      elif opt in ("-d", "--dir"):
         output_dir = arg
      elif opt == '-p':
          problem_type = arg
      elif opt == '-m':
            LEARNED_MODEL_NAME = arg
    if len(args) > 0:
        yaml_file = args[0]
        
    if genarate_yaml:
        generate_params_file(yaml_file, problem_type)
        sys.exit()
        
    ans = get_default_params(yaml_file)
    ans['problem_type'] = 'despot_without_display'
    if problem_type is not None:
        ans['problem_type'] = problem_type
    if output_dir is not None:
        ans['output_dir'] = output_dir
    if start_index is not None:
        ans['begin_index'] = start_index
    if end_index is not None:
        ans['end_index'] = end_index
    
        
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
            if not os.path.exists(ans['output_dir']):
                print "Creating path " + ans['output_dir']
                os.mkdir(ans['output_dir'])
            #TODO add automatic directory creation
            os.system(command)
        
if __name__ == '__main__':
    main()
    