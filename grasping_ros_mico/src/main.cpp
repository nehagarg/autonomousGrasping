#include "grasping_ros_tui.h"
#include "grasping_real_arm.h"
#include "ros/ros.h"
#include "log_file_reader.h"

using namespace despot;



DSPOMDP* RosTUI::InitializeModel(option::Option* options) {
     DSPOMDP* model;
     
      /*if (options[E_DATA_FILE])
            {
                if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        model = new GraspingRealArm(options[E_DATA_FILE].arg, number);
                }
                else
                {
                    model = new GraspingRealArm(options[E_DATA_FILE].arg, -1);
                }
            }
            else
            {
        */        if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        model = new GraspingRealArm( number);
                }
                else
                {
                    model = new GraspingRealArm(-1);
                }
        //    }
   
     
    return model;
  }



class RosWithoutDisplayTUI: public TUI {
public:
  RosWithoutDisplayTUI() {
  }

    DSPOMDP* InitializeModel(option::Option* options) {
     DSPOMDP* model;
    /*if (options[E_DATA_FILE])
            {
                if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        
                        model = new GraspingV4(options[E_DATA_FILE].arg, number);
                }
                else
                {
                    model = new GraspingV4(options[E_DATA_FILE].arg, -1);
                }
            }
            else
    */
     //{
       if (options[E_NUMBER]) {
                        int number = atoi(options[E_NUMBER].arg);
                        model = new GraspingRealArm( number);
                }
                else
                {
                    model = new GraspingRealArm(-1);
                }
             
     //       }
     
    return model;
  }
};




void GatherSimulationData()
{
    //GraspingRealArm* model = new GraspingRealArm(-1);
    VrepInterface* vrepInterfacePointer = new VrepInterface();
    std::cout<< "Gathering data" << std::endl;
    vrepInterfacePointer->GatherData(1);
    //model->GatherJointData(0);
    //model->GatherGripperStateData(0);
}

void test_python()
{
    log_file_reader lfr;
    lfr.testPythonCall();
}


int main(int argc, char* argv[]) {
    ros::init(argc,argv,"despot");
    //GatherSimulationData();
    //return 0;
    
    //test_python();
    //return 0;
   
  return RosWithoutDisplayTUI().run(argc, argv);
}


