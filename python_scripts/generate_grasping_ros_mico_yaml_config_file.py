import sys
import re
import getopt
import yaml
LEARNED_MODEL_NAME = None
SVM_MODEL_NAME = None

def get_grasping_object_name_list():
    pattern_list = ['G3DB11_cheese_final-10-mar-2016']
    pattern_list.append('G3DB39_beerbottle_final-11-mar-2016')
    pattern_list.append('G3DB40_carafe_final-11-mar-2016')
    pattern_list.append('G3DB41_jar_and_lid_final-11-mar-2016')
    pattern_list.append('G3DB43_wineglass2_final')
    pattern_list.append('G3DB48_bottle_and_plug_final')
    pattern_list.append('G3DB50_carton_final')
    pattern_list.append('G3DB54_candlestick_final')
    pattern_list.append('G3DB5_bottle_final-19-jan-2016')
    pattern_list.append('G3DB65_coffeemaker_final')
    pattern_list.append('G3DB66_chocolatebox_final')
    pattern_list.append('G3DB67_jug_final')
    pattern_list.append('G3DB73_juicebottle_final')
    pattern_list.append('G3DB84_yogurtcup_final')
    pattern_list.append('G3DB91_peppershaker_final')
    pattern_list.append('G3DB94_weight_final')
    assert(len(pattern_list)==16)
    return pattern_list

def load_config_from_file(yaml_file):
    ans = None
    with open(yaml_file,'r') as stream:
            ans = yaml.load(stream)
    return ans
            
def get_low_friction_table_config(ans):
    ans["object_mapping"] = ["data_low_friction_table_exp/SASOData_Cylinder_9cm_"]
    ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_8cm_")
    ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_7cm_")
    ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_75mm_")
    ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_85mm_")
    ans["object_min_z"] = [1.0950]*(len(ans["object_mapping"])+ 1) # 1 element extra for test object
    ans["object_initial_pose_z"] = [1.0998]*(len(ans["object_mapping"])+ 1) # 1 element extra for test object
    ans["low_friction_table"] = True
    
def create_basic_config():
    ans = {}
    #ans["start_state_index"] = -1
    ans["num_belief_particles"] = 1000
    ans["interface_type"] = 1
    ans["pick_reward"] = 100
    ans["pick_penalty"] = -10
    ans["invalid_state_penalty"] = -10
    ans["object_mapping"] = ["data_table_exp/SASOData_Cylinder_9cm_", "data_table_exp/SASOData_Cylinder_8cm_", "data_table_exp/SASOData_Cylinder_7cm_"]
    ans["object_mapping"].append("data_table_exp/SASOData_Cuboid_9cm_")
    ans["object_mapping"].append("data_table_exp/SASOData_Cuboid_8cm_")
    ans["object_mapping"].append("data_table_exp/SASOData_Cuboid_7cm_")
    ans["object_mapping"].append("data_table_exp/SASOData_Cylinder_75mm_")
    ans["object_mapping"].append("data_table_exp/SASOData_Cylinder_85mm_")
    ans["object_min_z"] = [1.1200]*(len(ans["object_mapping"])+ 1) # 1 element extra for test object
    ans["object_initial_pose_z"] = [1.1248]*(len(ans["object_mapping"])+ 1) # 1 element extra for test object
    
    ans["low_friction_table"] = False
    ans["test_object_id"] = len(ans["object_mapping"])
    ans["belief_object_ids"] = [0]
    ans["separate_close_reward"] = True
    ans["switching_method"] = 0 #0 for threshold based switching 1 for automatic one class svm based switching
    ans["svm_model_prefix"] = ""
    ans["learned_model_name"] = ""
    ans["switching_threshold"] = 10
        
    return ans

def get_toy_config(filename):
    ans = get_learning_config(filename, 'toy')
    ans["test_object_id"] = 0
    if 'test' in filename:
        ans["test_object_id"] = 1
    return ans


def get_learning_version_from_filename(filename):
    ans1 = '1'
    m = re.search('_v([0-9]+)', filename)
    if m:
        ans1 = m.groups(0)[0]
    ans2 = LEARNED_MODEL_NAME
    if ans2 is None:
        ans2 = 'model.ckpt-823'
    #TODO use pattern matching to get learned model ckpt no
    return (ans1, ans2)
    
def get_svm_model_name(filename):
    ans = SVM_MODEL_NAME
    if ans is None:
        ans =  "nu_0_1_kernel_rbf_gamma_0_1_"
    #TODO use pattern matchind to get svm veriosn from config filename
    return ans
def get_switching_threshold(filename):
    ans = 10
    m = re.search('_combined_[0-9]+-([0-9]+)', filename)
    if m:
       ans = m.groups(0)[0] 
    return int(ans)

def get_learning_config(filename, problem_type):
    ans= {}
    filename_filetype = None
    i = 0;
    for filetype in ['combined_0','combined_1', 'combined_2','combined_3', 'combined_4', 'combined_5']:
        if filetype in filename:
            filename_filetype = filetype
            ans["switching_method"] = i
        i =i+1
    if filename_filetype is not None:
        (learning_version, model_name) = get_learning_version_from_filename(filename)
        ans["learned_model_name"] = problem_type + "/version" + learning_version + "/" + model_name
        ans["switching_threshold"] = get_switching_threshold(filename)
        if ans["switching_method"] > 0:
            svm_model_name = get_svm_model_name(filename)
            ans["svm_model_prefix"] = "output/"+ problem_type +"/version" + learning_version + "/" + svm_model_name
    return ans    
    
def get_pocman_config(filename):
    return get_learning_config(filename, 'pocman')
        
    
def modify_basic_config(filename, ans):
    
    if 'toy' in filename:
        return get_toy_config(filename)
    
    if 'pocman' in filename:
        return get_pocman_config(filename)
    
    
    
    
    if filename == "VrepDataInterface.yaml" :
        ans["interface_type"] = 1
        ans["test_object_id"] = 0
    
    object_list = ['7cm', '8cm', '9cm', '75mm', '85mm']
    if 'G3DB' in filename:
        object_list = get_grasping_object_name_list()
    for filetype in ['combined_1', 'combined_2', 'combined_0-15', 'combined_0-20', 'combined_3-50', 'combined_4']:
        for interface_type in ["", "Data"]:
            file_prefix = "Vrep" + interface_type + "Interface_low_friction"
            if filename == file_prefix + "_"+ filetype + ".yaml" :
                ans = load_config_from_file(file_prefix + ".yaml")
                ans['svm_model_prefix'] = 'output/vrep/version7/nu_0_1_kernel_rbf_gamma_0_1_'
                ans['switching_method'] = int(filetype.split('_')[1].split('-')[0])
            for object_type in object_list:
                file_prefix = "Vrep" + interface_type + "InterfaceMultiCylinderObjectTest" + object_type + "_low_friction_table"
                if filename == file_prefix + '_' + filetype + '.yaml':
                    ans = load_config_from_file(file_prefix + '.yaml')
                    ans['svm_model_prefix'] = 'output/vrep/version8/nu_0_1_kernel_rbf_gamma_0_1_'
                    ans['switching_method'] = int(filetype.split('_')[1].split('-')[0])
    
    ans["switching_threshold"] = get_switching_threshold(filename)    
    
    if filename == "VrepDataInterface_low_friction.yaml" :
        get_low_friction_table_config(ans)
        ans["interface_type"] = 1
        ans["test_object_id"] = 0
        
    
    if filename == "VrepDataInterfaceMultiCylinderObjectTest9cm_low_friction_table.yaml" :
        get_low_friction_table_config(ans)
        ans["interface_type"] = 1
        ans["belief_object_ids"] = [0, 1, 2]
        ans["test_object_id"] = 0
        
        
        
    if filename == "VrepDataInterfaceMultiCylinderObjectTest8cm_low_friction_table.yaml" :
        get_low_friction_table_config(ans)
        ans["interface_type"] = 1
        ans["belief_object_ids"] = [0, 1, 2]
        ans["test_object_id"] = 1
        
    if filename == "VrepDataInterfaceMultiCylinderObjectTest7cm_low_friction_table.yaml" :
        get_low_friction_table_config(ans)
        ans["interface_type"] = 1
        ans["belief_object_ids"] = [0, 1, 2]
        ans["test_object_id"] = 2
     
    if filename == "VrepDataInterfaceMultiCylinderObjectTest75mm_low_friction_table.yaml" :
        get_low_friction_table_config(ans)
        ans["interface_type"] = 1
        ans["belief_object_ids"] = [0, 1, 2]
        ans["test_object_id"] = 3
        
    if filename == "VrepDataInterfaceMultiCylinderObjectTest85mm_low_friction_table.yaml" :
        get_low_friction_table_config(ans)
        ans["interface_type"] = 1
        ans["belief_object_ids"] = [0, 1, 2]
        ans["test_object_id"] = 4
        
        
        
    if filename == "VrepDataInterfaceMultiCylinderObjectTest9cm.yaml" :
        ans["interface_type"] = 1
        ans["belief_object_ids"] = [0, 1, 2]
        ans["test_object_id"] = 0
    if filename == "VrepDataInterfaceMultiCylinderObjectTest8cm.yaml" :
        ans["interface_type"] = 1
        ans["belief_object_ids"] = [0, 1, 2]
        ans["test_object_id"] = 1
    if filename == "VrepDataInterfaceMultiCylinderObjectTest7cm.yaml" :
        ans["interface_type"] = 1
        ans["belief_object_ids"] = [0, 1, 2]
        ans["test_object_id"] = 2
    if filename == "VrepDataInterfaceMultiCylinderObjectTest75mm.yaml" :
        ans["interface_type"] = 1
        ans["belief_object_ids"] = [0, 1, 2]
        ans["test_object_id"] = 6
    if filename == "VrepDataInterfaceMultiCylinderObjectTest85mm.yaml" :
        ans["interface_type"] = 1
        ans["belief_object_ids"] = [0, 1, 2]
        ans["test_object_id"] = 7
        
    if filename == "VrepInterfaceMultiCylinderObject.yaml" :
        ans["interface_type"] = 0
        ans["belief_object_ids"] = [0, 1, 2]
        
    if filename == "VrepInterface.yaml" :
        ans["interface_type"] = 0
        ans["test_object_id"] = 0
        
    if filename == "VrepInterface_cup.yaml" :
        ans["interface_type"] = 0
        ans["object_initial_pose_z"][-1] = 1.0950
        ans["object_min_z"][-1] = 1.0900
        
        
    if filename == "RealArmInterface.yaml" :
        ans["interface_type"] = 2
        
    return ans 
def write_config_in_file(filename, ans):
    
    from yaml import dump
    try:
        from yaml import CDumper as Dumper
    except ImportError:
        from yaml import Dump
    output = dump(ans, Dumper=Dumper)
    f = open(filename, 'w')
    f.write(output)

def generate_config_files_for_penalty100(type='G3DB'):
    object_list = ['7cm', '8cm', '9cm', '75mm', '85mm']
    interface_types = ["", "Data"]
    if type=='G3DB':
        object_list = get_grasping_object_name_list()
        interface_types = [""]
    for filetype in ['', '_combined_1', '_combined_2', '_combined_0-15', '_combined_0-20', '_combined_3-50', '_combined_4']:
        for interface_type in interface_types:    
            for object_type in object_list:
                file_prefix = "Vrep" +interface_type + "InterfaceMultiCylinderObjectTest" + object_type + "_low_friction_table"
                filename = file_prefix + filetype + '.yaml'
                ans = load_config_from_file(filename)
                ans["pick_penalty"] = -100
                ans["invalid_state_penalty"] = -100
                ans["learned_model_name"] = "vrep/version9/model.ckpt-976"
                write_config_in_file(filename.replace('Vrep','VrepPenalty100'), ans)
    
def generate_combined_config_files_for_G3DB(type='G3DB'):
    object_list = ['7cm', '8cm', '9cm', '75mm', '85mm']
    interface_types = ["", "Data"]
    if type=='G3DB':
        object_list = get_grasping_object_name_list()
        interface_types = [""]
    for filetype in ['combined_4']: #, 'combined_2', 'combined_0-15', 'combined_0-20', 'combined_3-50', 'combined_4']:
        for interface_type in interface_types:    
            for object_type in object_list:
                file_prefix = "Vrep" +interface_type + "InterfaceMultiCylinderObjectTest" + object_type + "_low_friction_table"
                filename = file_prefix + '_' + filetype + '.yaml'
                ans = create_basic_config()
                ans = modify_basic_config(filename, ans) 
                write_config_in_file(filename, ans)
def main():
    opts, args = getopt.getopt(sys.argv[1:],"g:hm:s:")
    global LEARNED_MODEL_NAME
    global SVM_MODEL_NAME
    for opt,arg in opts:    
        if opt == '-m':
            LEARNED_MODEL_NAME = arg
        elif opt == '-s':
            SVM_MODEL_NAME = arg
        elif opt =='-g':
            generate_config_files_for_penalty100(arg)
            #generate_combined_config_files_for_G3DB(arg)
        elif opt == '-h':
            print "python generate_grasping_ros_mico_yaml_config.py -m <learning model name> -s <joint model_name> <config filename>"
    
    filename = "VrepInterface.yaml"
    if len(args) > 0:
        filename = args[0]
    ans = create_basic_config()
    ans = modify_basic_config(filename, ans)
    write_config_in_file(filename, ans)
    

if __name__ == "__main__" :
    main()
