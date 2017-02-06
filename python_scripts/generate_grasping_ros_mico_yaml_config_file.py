import sys
from yaml import dump
try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper
    
def create_basic_config():
    ans = {}
    ans["start_state_index"] = -1
    ans["num_belief_particles"] = 1000
    ans["interface_type"] = 1
    ans["pick_reward"] = 100
    ans["pick_penalty"] = -100
    ans["invalid_state_penalty"] = -100
    ans["object_mapping"] = ["data_table_exp/SASOData_Cylinder_9cm_", "data_table_exp/SASOData_Cylinder_8cm_", "data_table_exp/SASOData_Cylinder_7cm_"]
    ans["object_mapping"].append("data_table_exp/SASOData_Cuboid_9cm_")
    ans["object_mapping"].append("data_table_exp/SASOData_Cuboid_8cm_")
    ans["object_mapping"].append("data_table_exp/SASOData_Cuboid_7cm_")
    ans["test_object_id"] = 0
    ans["belief_object_ids"] = [0]
    
    return ans

def modify_basic_config(filename, ans):
    if filename == "VrepDataInterface.yaml" :
        ans["interface_type"] = 1
    if filename == "VrepInterface.yaml" :
        ans["interface_type"] = 0
        
        
def main():
    filename = "VrepInterface.yaml"
    if sys.argv[1]:
        filename = sys.argv[1]
    ans = create_basic_config()
    modify_basic_config(filename, ans)
    output = dump(ans, Dumper=Dumper)
    f = open(filename, 'w')
    f.write(output)
    

if __name__ == "__main__" :
    main()