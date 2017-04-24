import sys
import yaml
from yaml import dump
try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dump

    
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
    ans["start_state_index"] = -1
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
        
    return ans

def modify_basic_config(filename, ans):
    if filename == "VrepDataInterface.yaml" :
        ans["interface_type"] = 1
        ans["test_object_id"] = 0
        
    for filetype in ['combined_1', 'combined_2']:
        for interface_type in ["", "Data"]:
            file_prefix = "Vrep" + interface_type + "Interface_low_friction"
            if filename == file_prefix + "_"+ filetype + ".yaml" :
                ans = load_config_from_file(file_prefix + ".yaml")
                ans['svm_model_prefix'] = 'output/vrep/version7/nu_0_1_kernel_rbf_gamma_0_1_'
                ans['switching_method'] = int(filetype.split('_')[1])
            for object_type in ['7cm', '8cm', '9cm', '75mm', '85mm']:
                file_prefix = "Vrep" + interface_type + "InterfaceMultiCylinderObjectTest" + object_type + "_low_friction_table"
                if filename == file_prefix + '_' + filetype + '.yaml':
                    ans = load_config_from_file(file_prefix + '.yaml')
                    ans['svm_model_prefix'] = 'output/vrep/version8/nu_0_1_kernel_rbf_gamma_0_1_'
                    ans['switching_method'] = int(filetype.split('_')[1])
    
        
    
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
def main():
    filename = "VrepInterface.yaml"
    if sys.argv[1]:
        filename = sys.argv[1]
    ans = create_basic_config()
    ans = modify_basic_config(filename, ans)
    output = dump(ans, Dumper=Dumper)
    f = open(filename, 'w')
    f.write(output)
    

if __name__ == "__main__" :
    main()
