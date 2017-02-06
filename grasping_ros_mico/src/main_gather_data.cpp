#include "VrepInterface.h"
#include "ros/ros.h"
#include "log_file_reader.h"







void GatherSimulationData()
{
    //GraspingRealArm* model = new GraspingRealArm(-1);
    VrepInterface* vrepInterfacePointer = new VrepInterface();
    std::cout<< "Gathering data" << std::endl;
    vrepInterfacePointer->GatherData(1);
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
    std::cout << "In main" << std::endl;
    ros::init(argc,argv,"gather_data" + to_string(getpid()));
    
    GatherSimulationData();
    return 0;
    
    //test_python();
    //return 0;
   
 //return RosWithoutDisplayTUI().run(argc, argv);
  
 // return RosTUI().run(argc, argv);
}


