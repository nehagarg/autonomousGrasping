
import os
import getopt
import yaml
import sys
try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper
from generate_grasping_ros_mico_yaml_config_file import get_learning_version_from_filename, get_switching_threshold, LEARNED_MODEL_NAME, get_grasping_object_name_list
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
    
    object_list = ['7cm', '8cm', '9cm', '75mm', '85mm'];
    for filetype in ['combined_0', 'combined_1', 'combined_2', 'combined_0-15', 'combined_0-20', 'combined_3-50', 'combined_4']:
        for interface_type in ["vrep_model", "data_model"]:
            file_prefix = interface_type + "_9cm_low_friction_"
            if file_name == file_prefix + filetype + ".yaml" :
                ans = get_default_params(file_prefix + "learning.yaml")
                ans['output_dir'] = ans['output_dir'] + "/" + filetype
                ans['config_file'] = (ans['config_file'].split('.'))[0] + '_' + filetype + ".yaml"
            for object_type in object_list:
                file_prefix =  interface_type + "_multi_object_" + object_type + "_low_friction_"
                if file_name == file_prefix  + filetype + '.yaml':
                    ans = get_default_params(file_prefix + 'learning.yaml')
                    ans['output_dir'] = ans['output_dir'] + "/" + filetype
                    ans['config_file'] = (ans['config_file'].split('.'))[0] + '_' + filetype + ".yaml"
    
    if 'penalty_100' in file_name:
        new_file_name = file_name
        if 'G3DB' in file_name:
            object_list = get_grasping_object_name_list()
            for object_type in object_list:
                if object_type in file_name:
                    G3DB_object_type = object_type
                    new_file_name = file_name.replace(G3DB_object_type, '75mm')
        if 'v8' in file_name:
            ans = get_default_params(new_file_name.replace('_penalty_100_v8', '') )
            ans['output_dir'] = ans['output_dir'].replace("penalty10","penalty100")
            #ans['output_dir'] = ans['output_dir'].replace("version8","version9")
            ans['config_file'] = ans['config_file'].replace('Vrep','VrepPenalty100V8')
        elif 'v10' in file_name:
            ans = get_default_params(new_file_name.replace('_penalty_100_v10', '') )
            ans['output_dir'] = ans['output_dir'].replace("penalty10","penalty100")
            ans['output_dir'] = ans['output_dir'].replace("version8","version10")
            ans['config_file'] = ans['config_file'].replace('Vrep','VrepPenalty100V10')
        else:
            ans = get_default_params(new_file_name.replace('_penalty_100', '') )
            ans['output_dir'] = ans['output_dir'].replace("penalty10","penalty100")
            ans['output_dir'] = ans['output_dir'].replace("version8","version9")
            ans['config_file'] = ans['config_file'].replace('Vrep','VrepPenalty100')
        if 'G3DB' in file_name:
            ans['config_file'] = ans['config_file'].replace('75mm', G3DB_object_type)
            ans['file_name_prefix'] = ans['file_name_prefix'].replace('75mm', G3DB_object_type)
            
    if 'fixed_distribution' in file_name:
        new_file_name = file_name
        if 'G3DB' in file_name:
            object_list = get_grasping_object_name_list()
            for object_type in object_list:
                if object_type in file_name:
                    G3DB_object_type = object_type
                    new_file_name = file_name.replace(G3DB_object_type, '75mm')
        ans = get_default_params(new_file_name.replace('_fixed_distribution', '') )
        ans['additional_params'] = '-l CAP --number='
        if('simulator' in ans['output_dir']):
            ans['output_dir'] = ans['output_dir'].replace("simulator","simulator/fixed_distribution")
        else:
            if 'penalty100' in ans['output_dir']:
                ans['output_dir'] = ans['output_dir'].replace("penalty100","penalty100/fixed_distribution")
            else:
                ans['output_dir'] = ans['output_dir'].replace("penalty10","penalty10/fixed_distribution")
        ans['end_index'] = 245
        if 'G3DB' in file_name:
            ans['config_file'] = ans['config_file'].replace('75mm', G3DB_object_type)
            ans['file_name_prefix'] = ans['file_name_prefix'].replace('75mm', G3DB_object_type)
         
    if 'combined' in file_name:
        ans['solver'] = 'LEARNINGPLANNING'
        if 'combined_4' in file_name:
            ans['solver'] = 'DESPOTWITHLEARNEDPOLICY'
    
        #m = re.search('combined_[0-9]+', file_name)
        #switching_version = int(m.group().split('_')[-1])
        #if switching_version > 2:
        #    ans['additional_params'] = '--max-policy-simlen=10 ' + ans['additional_params'] 
        
    
    if 'learning' in file_name:
        ans['solver'] = 'DEEPLEARNING'
    if 'baseline' in file_name:
        ans = get_default_params(file_name.replace('_baseline', '') )
        ans['output_dir'] = ans['output_dir']+ "/baseline"
        ans['solver'] = 'USERDEFINED'
        
        
                
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

def generate_penalty_100_v10_commands(type = 'G3DB'):
    object_list = ['7cm', '8cm', '9cm', '75mm', '85mm']
    if type == 'G3DB':
        object_list = get_grasping_object_name_list()
    for filetype in ['', '_learning', '_combined_0', '_combined_1', '_combined_2', '_combined_0-15', '_combined_0-20', '_combined_3-50', '_combined_4', '_baseline']:
        for interface_type in ["vrep_model_penalty_100_v10", "data_model_penalty_100_v10", "vrep_model_penalty_100_v10_fixed_distribution", "data_model_penalty_100_v10_fixed_distribution"]:
            #generate_params_file(interface_type + "_9cm_low_friction" + filetype + ".yaml", 'despot_without_display')
            for object_type in object_list:
                  generate_params_file(interface_type + "_multi_object_" + object_type + "_low_friction" + filetype + ".yaml", 'despot_without_display')       

def generate_penalty_100_v8_commands(type = 'G3DB'):
    object_list = ['7cm', '8cm', '9cm', '75mm', '85mm']
    if type == 'G3DB':
        object_list = get_grasping_object_name_list()
    for filetype in ['', '_learning', '_combined_0', '_combined_1', '_combined_2', '_combined_0-15', '_combined_0-20', '_combined_3-50', '_combined_4', '_baseline']:
        for interface_type in ["vrep_model_penalty_100_v8", "data_model_penalty_100_v8", "vrep_model_penalty_100_v8_fixed_distribution", "data_model_penalty_100_v8_fixed_distribution"]:
            #generate_params_file(interface_type + "_9cm_low_friction" + filetype + ".yaml", 'despot_without_display')
            for object_type in object_list:
                  generate_params_file(interface_type + "_multi_object_" + object_type + "_low_friction" + filetype + ".yaml", 'despot_without_display')       

def generate_penalty_100_commands(type = 'G3DB'):
    object_list = ['7cm', '8cm', '9cm', '75mm', '85mm']
    if type == 'G3DB':
        object_list = get_grasping_object_name_list()
    for filetype in ['', '_learning', '_combined_0', '_combined_1', '_combined_2', '_combined_0-15', '_combined_0-20', '_combined_3-50', '_combined_4', '_baseline']:
        for interface_type in ["vrep_model_penalty_100", "data_model_penalty_100", "vrep_model_penalty_100_fixed_distribution", "data_model_penalty_100_fixed_distribution"]:
            #generate_params_file(interface_type + "_9cm_low_friction" + filetype + ".yaml", 'despot_without_display')
            for object_type in object_list:
                  generate_params_file(interface_type + "_multi_object_" + object_type + "_low_friction" + filetype + ".yaml", 'despot_without_display')       

def generate_fixed_distribution_commands(type = 'G3DB'):
    object_list = ['7cm', '8cm', '9cm', '75mm', '85mm']
    if type == 'G3DB':
        object_list = get_grasping_object_name_list()
    for filetype in ['_baseline', '_combined_4']: #['', '_learning', '_combined_0', '_combined_1', '_combined_2', '_combined_0-15', '_combined_0-20', '_combined_3-50', '_combined_4']:
        for interface_type in ["vrep_model_fixed_distribution", "data_model_fixed_distribution"]:
            generate_params_file(interface_type + "_9cm_low_friction" + filetype + ".yaml", 'despot_without_display')
            for object_type in object_list:
                  generate_params_file(interface_type + "_multi_object_" + object_type + "_low_friction" + filetype + ".yaml", 'despot_without_display')       

def generate_fixed_distribution_3_commands():
    for filetype in  ['_baseline', '_combined_4']: #['_combined_3-50']:
        for interface_type in ["vrep_model", "data_model", "vrep_model_fixed_distribution", "data_model_fixed_distribution"]:
            generate_params_file(interface_type + "_9cm_low_friction" + filetype + ".yaml", 'despot_without_display')
            for object_type in ['7cm', '8cm', '9cm', '75mm', '85mm']:
                  generate_params_file(interface_type + "_multi_object_" + object_type + "_low_friction" + filetype + ".yaml", 'despot_without_display')       


def generate_sample_input_command(dir,error_files):
    object_list = ['7cm', '8cm', '9cm', '75mm', '85mm']
    command = 'data'
    if 'simulator' in dir:
        command = 'vrep'
    command = command + '_model_fixed_distribution'
    if 'singleObjectType' in dir:
        command = command + "_9cm"
    else:
        command = command + "_multi_object_pattern"
    command = command + '_low_friction'
    ans = []
    for error_file in error_files:
        o = ''
        for object_type in object_list:
            if object_type in error_file:
                o = object_type
        trial_no = int(error_file.split('.')[0].split('_')[-1])
        m = re.search('t([0-9]+)_', dir)
        learning_version = 'None'
        if m:
            t = m.groups(0)[0]
        else:
            t = '1'
            learning_version = 'empty'
        m = re.search('_n([0-9]+)', dir)
        if m:
            n = m.groups(0)[0]
        else:
            n = '1'
        combined_version = 'None'
        m = re.search('combined_([0-9]+)', dir)
        if m:
            combined_version = m.groups(0)[0]
        m = re.search('combined_[0-9]+-([0-9]+)', dir)
        if m:
            combined_version = combined_version + '-' + m.groups(0)[0]
        if o:
            ans.append(' '.join([o , command, t, n, learning_version, combined_version, repr(trial_no), repr(trial_no + 1), '1']))
    #print ans
    return ans

def generate_run_commands_for_error_files(dir):
    cur_dir = os.getcwd()
    os.chdir(dir)
    file_list = [f for f in os.listdir('.') if os.path.isfile(f)]
    error_files = []
    for file_name in file_list:
        if '.log' in file_name:
            with open(file_name,'r') as f:
                all_text = f.read()
                isErrorFile = False
                if 'ERROR' in all_text:
                    isErrorFile = True
                if 'failed' in all_text:
                    isErrorFile = True
                if 'Segmentation fault' in all_text:
                    isErrorFile = True
                if 'Simulation terminated in' not in all_text:
                    isErrorFile = True
                if isErrorFile:
                    error_files.append(file_name)
    os.system("grep 'Simulation terminated in' *.log | grep Binary > binary_files.txt")
    with open('binary_files.txt', 'r') as f:
        for line in f:
            file_name = line.split(' ')[2]
            error_files.append(file_name)
    os.chdir(cur_dir)
    command_list = generate_sample_input_command(dir,error_files)
    print command_list
    return command_list

def generate_fixed_fistribution_sample_input(dir_name = None, output_file = None):
    dir_iterator = get_dir(dir_name)
    all_commands = []
    for dir in dir_iterator:
	print dir
        all_commands = all_commands + generate_run_commands_for_error_files(dir)
    if output_file is None:
        output_file = 'sample_input.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(all_commands))
def correct_fixed_distribution_log_file_numbering(dir_name=None, e = False):
    dir_iterator = get_dir(dir_name)
    for dir in dir_iterator:
	#if 'combined_3' in dir:
            print dir
            correct_log_file_numbering(dir, e)

def correct_log_file_numbering(dir, e = False):
    cur_dir = os.getcwd()
    os.chdir(dir)
    file_list = [f for f in os.listdir('.') if os.path.isfile(f)]
    new_file_list = []
    for file_name in file_list:
        if '.log' in file_name:
            """ 
            trial_no = int((file_name.split('_')[-1]).split('.')[0])
            if(trial_no < 245):
                new_trial_no = (trial_no/49)*81 + (trial_no % 49);
            else:
                new_trial_no = ((trial_no - 245)% 32) + 49 + ((trial_no -245)/32)*81
            new_file_name = file_name.replace('_' + repr(trial_no)+'.log', '_' + repr(new_trial_no)+'.log')
            new_file_name = new_file_name + '_'
            """
            new_file_name = file_name
            new_file_name = new_file_name.replace('107mm', '75mm')
            new_file_name = new_file_name.replace('117mm', '85mm')
            new_file_name = new_file_name.replace('_n112_', '_n80_')
            new_file_name = new_file_name.replace('_n192_', '_n160_')
            new_file_name = new_file_name.replace('_n256_', '_n160_')
            new_file_name = new_file_name.replace('_n222_', '_n320_')
            new_file_name = new_file_name.replace('_n960_', '_n640_')
            new_file_name = new_file_name.replace('_n1920_', '_n1280_')
            new_file_name = new_file_name.replace('_n12112_', '_n1280_')
            new_file_name = new_file_name.replace('_n1133_', '_n1280_')
            new_file_list.append(new_file_name)
            
        else:
            new_file_list.append(file_name) #To keep same numbering
            
            
    for i in range(0,len(file_list)):
        if file_list[i] != new_file_list[i]:
            command = 'mv ' + file_list[i] + ' ' + new_file_list[i]
            if e:
                os.system(command)
            else:
                print command
    """        
    for i in range(0,len(new_file_list)):
        new_name = new_file_list[i].replace('.log_', '.log')
        command = 'mv ' + new_file_list[i] + ' ' + new_name
        if e:
            os.system(command)
        else:
            print command
    """
            
    os.chdir(cur_dir)
        
def add_learning_pattern(dir, pattern):
    cur_dir = os.getcwd()
    os.chdir(dir)
    file_list = [f for f in os.listdir('.') if os.path.isfile(f)]
    #print file_list
    for file_name in file_list:
        if '.log' in file_name:
            with open(file_name,'r') as f:
                all_text = f.read()
            new_pattern = 'Before calling exec '
            if('Learning' in pattern):
                index = 0
                while index < len(all_text):
                    index = all_text.find(pattern, index)
                    if index == -1:
                        break
                    new_index = index
                    while not all_text[new_index -1].isdigit():
                        new_index = new_index - 1
                        #if new_index < index -50:
                        #    break;
                    print all_text[new_index -1 : index]   
                    new_index2 = new_index -1
                    while all_text[new_index2].isdigit():
                        new_index2 = new_index2 - 1
                        #if new_index2 < new_index -50:
                        #    break;
                    print all_text[new_index2:new_index -1]
                    #new_index3 = new_index2 - 4;
                    print all_text[new_index2 -4:new_index2]
                    if all_text.find('/', new_index2 -4, new_index2) == -1:
                    #new_index = all_text.find("\d+\.\d+",index)
                    #print repr(index)
                    #print repr(new_index)
                    #print ":" + all_text[index:index+len(pattern)] + ":"
                    #if len(all_text[new_index2+1:new_index]) == 1:
                        all_text = all_text[:index] + '@' + all_text[index + 1 :]
                    index = index + len(pattern) + 10
                new_text = all_text.replace('@', new_pattern)
            else:
                new_text = all_text.replace(pattern, new_pattern)
            with open(file_name,'w') as f:
                f.write(new_text)
    
    
    os.chdir(cur_dir)
    
def correct_log_files_percent_learning_calculation(dir_name = None):
    dir_iterator = get_dir(dir_name)
    for dir in dir_iterator:
        for switch_dir in ['combined_1', 'combined_2']:
            if switch_dir in dir:
                add_learning_pattern(dir,"22LearningPlanningSolver::Search()")
            else:
                add_learning_pattern(dir,"[['$'")
def get_dir(dir_name = None):
    root_dir = '~/WORK_FOLDER/add something'
    if dir_name is not None:
        root_dir = dir_name
    dict1 = {}
    dict1['single'] = 'cylinder_9cm_reward100_penalty10'
    dict1['multi'] = 'belief_cylinder_7_8_9_reward100_penalty10'
    dict1['data'] = 'fixed_distribution'
    dict1['vrep'] =  'simulator/fixed_distribution'
    dict2 = {}
    dict2['single'] = [5,10,20,40,80,160,320]
    dict2['multi'] = [5,10,20,40,80,160,320,640,1280]
    dict3 = {}
    dict3['single'] = 'learning/version7'
    dict3['multi'] = 'learning/version8'
    for experiment_type in ['single', 'multi']:
        dir1 = root_dir +"/" + experiment_type + "ObjectType/" + dict1[experiment_type]
        for interface_type in ['data','vrep']:
            dir2 = dir1 + "/" + dict1[interface_type]
            for t in [1,5]:
                for n in dict2[experiment_type]:
                    dir = dir2 + "/t" + repr(t) + '_n' + repr(n)
                    #add_learning_pattern(dir,"[['$'")
                    yield dir
            dir3 = dir2 + "/" + dict3[experiment_type]
            #add_learning_pattern(dir3,"[['$'")
            yield dir3
            for switch_dir in [ 'combined_1', 'combined_2', 'combined_0-15', 'combined_0-20', 'combined_3-50']:
                for t in [1,5]:
                    for n in dict2[experiment_type]:
                        dir = dir3 +"/" + switch_dir + "/t" + repr(t) + '_n' + repr(n)
                        yield dir
                        #if switch_dir in [ 'combined_1', 'combined_2']:
                        #    add_learning_pattern(dir,"22LearningPlanningSolver::Search()")
                        #else:
                        #    add_learning_pattern(dir,"[['$'")
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
    
        
    if ans['solver'] == 'DEEPLEARNING' or ans['solver'] == 'USERDEFINED':
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
    
