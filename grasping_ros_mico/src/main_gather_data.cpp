#include "VrepInterface.h"
#include "ros/ros.h"
#include "log_file_reader.h"







void GatherSimulationData(int val)
{
    //GraspingRealArm* model = new GraspingRealArm(-1);
    RobotInterface::low_friction_table = true;
    VrepInterface* vrepInterfacePointer = new VrepInterface();
    vrepInterfacePointer->min_z_o.push_back(vrepInterfacePointer->default_min_z_o);
    vrepInterfacePointer->initial_object_pose_z.push_back(vrepInterfacePointer->default_initial_object_pose_z);
    
    std::cout<< "Gathering data" << std::endl;
    vrepInterfacePointer->GatherData(val);
    //vrepInterfacePointer->GatherJointData(0);
    //model->GatherJointData(0);
    //model->GatherGripperStateData(0);
}

void test_python()
{
    log_file_reader lfr;
    lfr.testPythonCall();
}


int main(int argc, char* argv[]) {
    
    std::istringstream iss( argv[1] );
        int val = 0;

        if (iss >> val)
        {
            std::cout << "Object id is  : " << val << std::endl;
        }
        else{
            std::cout << "Object id is : " << val << std::endl;
        }
    std::cout << "In main" << std::endl;
    ros::init(argc,argv,"gather_data" + to_string(getpid()));
    
    GatherSimulationData(val);
    return 0;
    
    //test_python();
    //return 0;
   
 //return RosWithoutDisplayTUI().run(argc, argv);
  
 // return RosTUI().run(argc, argv);
}


