#include "VrepInterface.h"
#include "ros/ros.h"
#include "log_file_reader.h"

class G3DB_z_values
{
public:
    double get_min_z_o(double id)
    {
        double ans = get_initial_object_pose_z(id) -0.0048;
        return ans;
    }
    double get_initial_object_pose_z(double id)
    {
        if(id>=84 && id < 85)
        {
           return  1.1406;
        }
        if(id>=1 && id < 2)
        {
            return 1.0899;
        }
    }
};




void GatherSimulationData(std::string val, double epsi, int action_type, 
        int min_x, int max_x, int min_y, int max_y, int object_state_id, bool generate_default)
{
    //GraspingRealArm* model = new GraspingRealArm(-1);
    RobotInterface::low_friction_table = true;
    RobotInterface::version5 = true;
    RobotInterface::use_data_step = false;
    RobotInterface::get_object_belief = false;
    int start_index = -10;
    if(epsi==0.005)
    {
        start_index = -10000;
    }
    VrepInterface* vrepInterfacePointer = new VrepInterface(start_index);    
    //vrepInterfacePointer->epsilon = epsi;
    if(!generate_default)
    {
        vrepInterfacePointer->LoadObjectInScene(val);
    }
    else
    {
        //TODO replace with loading object properties
        vrepInterfacePointer->min_z_o.push_back(vrepInterfacePointer->default_min_z_o);
        vrepInterfacePointer->initial_object_pose_z.push_back(vrepInterfacePointer->default_initial_object_pose_z);
    }
    //Expanding valid state for object for data collection
    vrepInterfacePointer->min_x_o = vrepInterfacePointer->min_x_o - 0.1;
    vrepInterfacePointer->max_x_o = vrepInterfacePointer->max_x_o + 0.1;
    vrepInterfacePointer->min_y_o = vrepInterfacePointer->min_y_o - 0.1;
    vrepInterfacePointer->max_y_o = vrepInterfacePointer->max_y_o + 0.1;
    /*vrepInterfacePointer->min_z_o.push_back(vrepInterfacePointer->default_min_z_o);
    vrepInterfacePointer->initial_object_pose_z.push_back(vrepInterfacePointer->default_initial_object_pose_z);
    if (val > 1000)
    {
        G3DB_z_values z_values;
        double g3db_object_id = val -1000;
        vrepInterfacePointer->min_z_o.push_back(z_values.get_min_z_o(g3db_object_id));
        vrepInterfacePointer->initial_object_pose_z.push_back(z_values.get_initial_object_pose_z(g3db_object_id));
    }
    */
    std::cout<< "Gathering data" << std::endl;
    vrepInterfacePointer->GatherData(val, action_type, epsi, min_x, max_x, min_y, max_y, object_state_id, generate_default);
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
    std::string val;
    double epsilon = 0.01;
    int action_type = 0;
    int min_x = -1;
    int max_x = -1;
    int min_y=-1;
    int max_y = -1;
    bool generate_default = false;
    //6 27 48
    //3 24 45
    //0 21 42
    int object_state_id = 24;
    if(argc >=2)
    {
        std::istringstream iss( argv[1] );
        

        if (iss >> val)
        {
            std::cout << "Object id is  : " << val << std::endl;
        }
        else{
            std::cout << "Object id is : " << val << std::endl;
        }
    }
    
    if(argc >=3)
    {
       std::istringstream iss( argv[2] );
       iss >> action_type;
       std::cout << "Action type is  : " << action_type << std::endl;
    }
    if(argc >=4)
    {
        std::istringstream iss( argv[3] );
        iss >> epsilon;
        std::cout << "Epsilon is  : " << epsilon << std::endl;
        
    }
    if(argc >=5)
    {
        std::istringstream iss( argv[4] );
        char c;
        iss >> min_x >> c >> max_x >> c >> min_y >> c >> max_y;
        std::cout << "Minmax is  : " << min_x << "," << max_x 
                << "," << min_y << "," << max_y << std::endl;
        
    }
    if(argc >=6)
    {
        std::istringstream iss( argv[5] );
        iss >> object_state_id;
       std::cout << "Object state is  : " << object_state_id << std::endl;
        
    }
    
    if(argc >=7)
    {
        generate_default = true;
       std::cout << "Generating defualt " << std::endl;
        
    }
    std::cout << "In main" << std::endl;
    ros::init(argc,argv,"gather_data" + to_string(getpid()));
    
    GatherSimulationData(val, epsilon, action_type, min_x, max_x, min_y, max_y, object_state_id, generate_default);
    return 0;
    
    //test_python();
    //return 0;
   
 //return RosWithoutDisplayTUI().run(argc, argv);
  //./bin/gather_data Cylinder_9 0 0.005 0,1,-1,-1 24 true
 // return RosTUI().run(argc, argv);
}


