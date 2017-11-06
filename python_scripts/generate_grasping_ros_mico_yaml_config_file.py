import sys
import re
import getopt
import yaml
import grasping_object_list 
LEARNED_MODEL_NAME = None
SVM_MODEL_NAME = None

#Keeping this function for backward compatibility
def get_grasping_object_name_list(type='used'):
    return grasping_object_list.get_grasping_object_name_list(type)

def load_config_from_file(yaml_file):
    ans = None
    with open(yaml_file,'r') as stream:
            ans = yaml.load(stream)
    return ans

    
def get_initial_object_pose_z(id):
    
    if id==84:
       return  1.1406
    if id==1 :
       return 1.0899;


    
def get_min_z_o(id):
    
    ans = get_initial_object_pose_z(id) -0.0048;
    return ans;
    


def get_g3db_belief_ver5_low_friction_table_config(ans):
    ans["object_mapping"] = ["data_low_friction_table_exp_ver5/SASOData_Cylinder_1001cm_"]
    ans["object_mapping"].append("data_low_friction_table_exp_ver5/SASOData_Cylinder_1084cm_")
    ans["object_mapping"].append("data_low_friction_table_exp_ver5/SASOData_Cylinder_9cm_")
    ans["object_mapping"].append("data_low_friction_table_exp_ver5/SASOData_Cylinder_8cm_")
    ans["object_mapping"].append("data_low_friction_table_exp_ver5/SASOData_Cylinder_7cm_")
    ans["object_mapping"].append("data_low_friction_table_exp_ver5/SASOData_Cylinder_75cm_")
    ans["object_mapping"].append("data_low_friction_table_exp_ver5/SASOData_Cylinder_85cm_")
    
    #ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_7cm_")
    #ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_75mm_")
    #ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_85mm_")
    ans["object_min_z"] = [get_min_z_o(1), get_min_z_o(84)]
    ans["object_min_z"] = ans["object_min_z"] + [1.0950]*(len(ans["object_mapping"]) -2 + 1) 
    ans["object_initial_pose_z"] = [get_initial_object_pose_z(1), get_initial_object_pose_z(84)]
    ans["object_initial_pose_z"] =ans["object_initial_pose_z"]+ [1.0998]*(len(ans["object_mapping"])-2 + 1)
    ans["low_friction_table"] = True
    ans["belief_object_ids"] = [0,1]
    ans["version5"] = True
    
def get_g3db_belief_low_friction_table_config(ans):
    ans["object_mapping"] = ["data_low_friction_table_exp/SASOData_Cylinder_1001cm_"]
    ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_1084cm_")
    ans["object_min_z"] = [get_min_z_o(1), get_min_z_o(84)]
    ans["object_initial_pose_z"] = [get_initial_object_pose_z(1), get_initial_object_pose_z(84)]
    ans["low_friction_table"] = True
    ans["belief_object_ids"] = [0,1]
    
def get_low_friction_table_config(ans):
    ans["object_mapping"] = ["data_low_friction_table_exp/SASOData_Cylinder_9cm_"]
    ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_8cm_")
    ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_7cm_")
    ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_75mm_")
    ans["object_mapping"].append("data_low_friction_table_exp/SASOData_Cylinder_85mm_")
    ans["object_min_z"] = [1.0950]*(len(ans["object_mapping"])+ 1) # 1 element extra for test object
    ans["object_initial_pose_z"] = [1.0998]*(len(ans["object_mapping"])+ 1) # 1 element extra for test object
    ans["low_friction_table"] = True
    
def create_basic_config(filename):
    ans = get_learning_config(filename, 'vrep')
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
    #ans["switching_method"] = 0 #0 for threshold based switching 1 for automatic one class svm based switching
    #ans["svm_model_prefix"] = ""
    #ans["learned_model_name"] = ""
    #ans["switching_threshold"] = 10
        
    return ans

def get_toy_config(filename):
    ans = get_learning_config(filename, 'toy')
    ans["test_object_id"] = 0
    if 'test' in filename:
        ans["test_object_id"] = 1
    return ans


def get_learning_version_from_filename(filename):
    global LEARNED_MODEL_NAME
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
    
    if 'Ver5' in filename:
            get_g3db_belief_ver5_low_friction_table_config(ans)
            return ans
        
    if '1001-84' in filename:
        if 'Ver5' in filename:
            get_g3db_belief_ver5_low_friction_table_config(ans)
        else:
            get_g3db_belief_low_friction_table_config(ans)
        return ans
    
    
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

def generate_config_files_for_penalty100_v10(type='G3DB'):
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
                if 'svm_model_prefix' in ans.keys():
                    ans["svm_model_prefix"] = ans["svm_model_prefix"].replace('version8', 'version10')
                ans["learned_model_name"] = "vrep/version10/model.ckpt-826"
                write_config_in_file(filename.replace('Vrep','VrepPenalty100V10'), ans)
                
def generate_config_files_for_penalty100_v8(type='G3DB'):
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
                #if 'svm_model_prefix' in ans.keys():
                #    ans["svm_model_prefix"] = ans["svm_model_prefix"].replace('version8', 'version9')
                #ans["learned_model_name"] = "vrep/version9/model.ckpt-976"
                write_config_in_file(filename.replace('Vrep','VrepPenalty100V8'), ans)
                
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
                if 'svm_model_prefix' in ans.keys():
                    ans["svm_model_prefix"] = ans["svm_model_prefix"].replace('version8', 'version9')
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

def generate_G3DB_belief_files():
    object_list = get_grasping_object_name_list('coffee_yogurt_cup')
    interface_types = ["", "Data"]
    for filetype in ['']:
       for interface_type in interface_types:
           for object_type in object_list:
                file_prefix = "Vrep" +interface_type + "InterfaceMulti1001-84Test" + object_type + "_low_friction_table"
                filename = file_prefix + filetype + '.yaml'
                ans = create_basic_config(filename)
                ans = modify_basic_config(filename, ans)
                ans["interface_type"] = 0
                if interface_type == 'Data':
                    ans["interface_type"] = 1
                ans["test_object_id"] = object_list.index(object_type)                
                write_config_in_file(filename, ans)
                
def generate_G3DB_ver5_belief_files(weighted = 'false'):
    global LEARNED_MODEL_NAME
    LEARNED_MODEL_NAME = 'model.ckpt-867' #for version 13
    object_list = get_grasping_object_name_list('coffee_yogurt_cup')
    weighted_prefix = ""
    if weighted != 'false':
        weighted_prefix = "weighted_belief/"
    interface_types = ["", "Data"]
    for filetype in ['', '_combined_0_v13']:
       for interface_type in interface_types:
           for object_type in object_list:
                file_prefix = "low_friction_table/vrep_scene_ver5/penalty10/yoghurtCup/" + weighted_prefix + "Vrep" +interface_type + "InterfaceVer5Multi1001-84Test" + object_type + "_low_friction_table"
                filename = file_prefix + filetype + '.yaml'
                ans = create_basic_config(filename)
                ans = modify_basic_config(filename, ans)
                ans["interface_type"] = 0
                if interface_type == 'Data':
                    if weighted != 'false':
                        ans["use_data_step"] = True
                    else:
                        ans["interface_type"] = 1
                ans["test_object_id"] = object_list.index(object_type)
                ans["version5"] = True
                if weighted != 'false':
                    ans["get_object_belief"] = True
                write_config_in_file(filename, ans)
                
def generate_G3DB_ver5_single_belief_files():
    object_list = get_grasping_object_name_list('coffee_yogurt_cup')
    interface_types = ["", "Data"]
    for filetype in ['']:
       for interface_type in interface_types:
           for object_type in object_list:
                file_prefix = "Vrep" +interface_type + "InterfaceVer5Single1001-84Test" + object_type + "_low_friction_table"
                filename = file_prefix + filetype + '.yaml'
                ans = create_basic_config(filename)
                ans = modify_basic_config(filename, ans)
                ans["interface_type"] = 0
                if interface_type == 'Data':
                    ans["interface_type"] = 1
                ans["test_object_id"] = object_list.index(object_type)
                ans["belief_object_ids"] = [object_list.index(object_type)]
                
                write_config_in_file(filename, ans)

def generate_G3DB_ver5_cylinder_belief_files():
    global LEARNED_MODEL_NAME
    #LEARNED_MODEL_NAME = 'model.ckpt-693' #for version 11
    LEARNED_MODEL_NAME = 'model.ckpt-965' #for version 12
    object_list = get_grasping_object_name_list('coffee_yogurt_cup')
    object_list = object_list+['9cm', '8cm', '7cm', '75mm', '85mm']
    interface_types = ["", "Data"]
    for filetype in ['_v12_combined_0']: #['', '_v11_combined_0']:
       for interface_type in interface_types:
           for object_type in object_list:
                file_prefix = "low_friction_table/vrep_scene_ver5/penalty10/cylinder/Vrep" +interface_type + "InterfaceVer5MultiCylinderObjectTest" + object_type + "_low_friction_table"
                filename = file_prefix + filetype + '.yaml'
                ans = create_basic_config(filename)
                ans = modify_basic_config(filename, ans)
                ans["interface_type"] = 0
                if interface_type == 'Data':
                    ans["interface_type"] = 1
                ans["test_object_id"] = object_list.index(object_type)
                ans["belief_object_ids"] = [2,3,4]
                
                write_config_in_file(filename, ans)

def generate_G3DB_ver5_cylinder_cup_belief_files():
    object_list = get_grasping_object_name_list('coffee_yogurt_cup')
    object_list = object_list+['9cm', '8cm']
    interface_types = ["", "Data"]
    for filetype in ['']:
       for interface_type in interface_types:
           for object_type in object_list:
                file_prefix = "Vrep" +interface_type + "InterfaceVer5MultiCylinder-1001Test" + object_type + "_low_friction_table"
                filename = file_prefix + filetype + '.yaml'
                ans = create_basic_config(filename)
                ans = modify_basic_config(filename, ans)
                ans["interface_type"] = 0
                if interface_type == 'Data':
                    ans["interface_type"] = 1
                ans["test_object_id"] = object_list.index(object_type)
                ans["belief_object_ids"] = [0,2,3]
                
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
            #generate_config_files_for_penalty100_v10(arg)
            #generate_combined_config_files_for_G3DB(arg)
            #generate_G3DB_belief_files()
            generate_G3DB_ver5_belief_files(arg)
            #generate_G3DB_ver5_single_belief_files()
            #generate_G3DB_ver5_cylinder_belief_files()
            #generate_G3DB_ver5_cylinder_cup_belief_files()
            return
        elif opt == '-h':
            print "python generate_grasping_ros_mico_yaml_config.py -m <learning model name> -s <joint model_name> <config filename>"
    
    filename = "VrepInterface.yaml"
    if len(args) > 0:
        filename = args[0]
    ans = create_basic_config(filename)
    ans = modify_basic_config(filename, ans)
    write_config_in_file(filename, ans)
    

if __name__ == "__main__" :
    main()
