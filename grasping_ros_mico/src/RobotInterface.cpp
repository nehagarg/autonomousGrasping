/* 
 * File:   RobotInterface.cpp
 * Author: neha
 * 
 * Created on May 5, 2015, 2:11 PM
 */

#include "RobotInterface.h"
#include <chrono>


std::vector <int> RobotInterface::objects_to_be_loaded;
std::vector<std::string> RobotInterface::object_id_to_filename;
bool RobotInterface::low_friction_table;


RobotInterface::RobotInterface() {
    std::ifstream infile;
    infile.open("data/sensor_mean_std_max.txt");
    double sensor_mean, sensor_std , sensor_max;
    int count = 0;
    while (infile >> sensor_mean >> sensor_std >> sensor_max)
    {
        touch_sensor_mean[count] = sensor_mean;
        touch_sensor_std[count] = sensor_std;
        touch_sensor_max[count] = sensor_max;
       // std::cout << count << ": "<< touch_sensor_mean[count] << " " << touch_sensor_std[count] << std::endl;
        count++;

    }
    infile.close();
    
    //Read touch sensor reading when gripper closed
    infile.open("data/gripper_closed_without_object_touch_values.txt");
    for(int i = 0; i < 48; i++)
    //for(int i = 0; i < 2; i++)
    {
        infile >> touch_sensor_mean_closed_without_object[i];
    }
    infile.close();
    
    //Read touch sensor reading when gripper closed with
    infile.open("data/gripper_closed_with_object_touch_values.txt");
    for(int i = 0; i < 48; i++)
    //for(int i = 0; i < 2; i++)
    {
        infile >> touch_sensor_mean_closed_with_object[i];
    }
    infile.close();
    
    if(low_friction_table)
    {
        min_x_o = min_x_o_low_friction_table;
        default_min_z_o = default_min_z_o_low_friction_table;
        default_initial_object_pose_z = default_initial_object_pose_z_low_friction_table;
        initial_gripper_pose_z = initial_gripper_pose_z_low_friction_table;
        initial_object_x = initial_object_x_low_friction_table;
    }
    //Load simulation data for belief object
    for(int i = 0; i < objects_to_be_loaded.size(); i++)
    {
        int object_id = objects_to_be_loaded[i];
        std::cout << "Loading object " << object_id << " with filename " << object_id_to_filename[object_id] << std::endl;
        getSimulationData( object_id);
        std::cout << simulationDataCollectionWithObject[object_id][0].size() << " entries for action 0" << std::endl;
    }
    
}

RobotInterface::RobotInterface(const RobotInterface& orig) {
}

RobotInterface::~RobotInterface() {
}

void RobotInterface::getSimulationData(int object_id) {

   
    //Read simualtion data with object
    SimulationDataReader simDataReader;
    std::ifstream simulationDataFile;
    
    //simulationDataFile.open("data/simulationData1_allParts.txt");
    std::string simulationFileName = object_id_to_filename[object_id]+"allActions.txt";
    simulationDataFile.open(simulationFileName);
    //int t_count = 0;    
    while(!simulationDataFile.eof())
    {
        SimulationData simData; double reward; int action;
        //simData.current_gripper_pose.pose.position.x = temp_read;
       // simDataReader.parseSimulationDataLine(simulationDataFile, simData, action, reward);
        simDataReader.parseSimulationDataLineTableData(simulationDataFile, simData, action, reward);
        /*t_count++;
        if(t_count > 10)
        {
            exit(0);
        }*/
        //std::cout << reward << " " << action << "*";
        if(reward != -1000 && reward != -2000)
        {
            if(action!= A_OPEN)
            {
                //TODO also filter the states where the nexxt state and observation is same as given by defulat state
                simulationDataCollectionWithObject[object_id][action].push_back(simData);
                
                simulationDataIndexWithObject[object_id][action].push_back(simulationDataIndexWithObject[object_id][action].size());
                
            }
        }  
    }
    //std::cout << std::endl;
    simulationDataFile.close();
    
    //simulationDataFile.open("data/simulationData_1_openAction.txt");
    simulationDataFile.open(object_id_to_filename[object_id]+"openAction.txt");
   
    
    while(!simulationDataFile.eof())
    {
        SimulationData simData; double reward; int action;
        //simData.current_gripper_pose.pose.position.x = temp_read;
        //simDataReader.parseSimulationDataLine(simulationDataFile, simData, action, reward);
        simDataReader.parseSimulationDataLineTableData(simulationDataFile, simData, action, reward);
        
        //std::cout << reward << " ";
        if(reward != -1000 && reward != -2000)
        {
            
            //TODO also filter the states where the nexxt state and observation is same as given by defulat state
            simulationDataCollectionWithObject[object_id][action].push_back(simData);
            simulationDataIndexWithObject[object_id][action].push_back(simulationDataIndexWithObject[object_id][action].size());
                
            
        }  
    }
    //std::cout << std::endl;
    simulationDataFile.close();
}


bool RobotInterface::Step(GraspingStateRealArm& grasping_state, double random_num, int action, double& reward, GraspingObservation& grasping_obs, bool debug) const {
GraspingStateRealArm initial_grasping_state = grasping_state;
//debug = true;
    //PrintState(grasping_state, std::cout);
    // PrintAction(action);
   // GraspingStateRealArm nearest_grasping_state = GetNearestGraspingStates(grasping_state,0);
    //Get next state
    
    //Check if gripper is closed using finger joint values
    int gripper_status = GetGripperStatus(grasping_state.finger_joint_state);
    
    //Get next state and observation
    if(action < A_CLOSE)
    {
        if(gripper_status == 0) //gripper is open
        {
            GetNextStateAndObsFromData(grasping_state, grasping_state, grasping_obs, action, debug);
            
        }
        else
        {
            //State remains same
            GetObsFromData(grasping_state, grasping_obs, random_num, action, debug);
        }
    }
    else if (action == A_CLOSE) //Close gripper
    { 
        if(gripper_status == 0) //gripper is open
        {
            GetNextStateAndObsFromData(grasping_state, grasping_state, grasping_obs, action, debug);
           
        }
        else
        {
            //State remains same
            GetObsFromData(grasping_state, grasping_obs, random_num, action, debug);
        }
    }
    else if (action == A_OPEN) // Open gripper
    {
        if(gripper_status > 0) // gripper is closed
        {
           GetNextStateAndObsFromData(grasping_state, grasping_state, grasping_obs, action, debug);
           int new_gripper_status = GetGripperStatus(grasping_state.finger_joint_state);
           while(new_gripper_status > 0)
           {
               if(debug) {
                std::cout << "Moving gripper back by 1 cm to let it open" << std::endl;
               }
               grasping_state.gripper_pose.pose.position.x = grasping_state.gripper_pose.pose.position.x - 0.01;
               GetNextStateAndObsFromData(grasping_state, grasping_state, grasping_obs, action, debug);
               new_gripper_status = GetGripperStatus(grasping_state.finger_joint_state);
           }
        }
        else
        {
           //State remains same. 
            //Cannot get observation from data for open action as observation is for open action after close action. Gripper might not open correctly and give wrong observation
            //Using dummy action A_INCREASE_X to get observation as the gripper will be open in this action
            GetObsFromData(grasping_state, grasping_obs, random_num, A_INCREASE_X, debug); 
            
        }
    }
    else if (action == A_PICK) // Pick object
    {
        if(gripper_status > 0)
        {
            GetNextStateAndObsFromData(grasping_state, grasping_state, grasping_obs,  action, debug);
        }
        else
        {
            GetNextStateAndObsUsingDefaulFunction(grasping_state, grasping_obs, action, debug);
        }
       /* grasping_state.gripper_pose.pose.position.z = grasping_state.gripper_pose.pose.position.z + pick_z_diff;
        grasping_state.gripper_pose.pose.position.x =  pick_x_val; 
        grasping_state.gripper_pose.pose.position.y =  pick_y_val;
        
        if(gripper_status == 2) //Object is inside gripper and gripper is closed
        {
            //Change position of object too
            //Currently pick always succeeds
            //TODO : Gather cases when pick fails even when object is grasped 
            //and compare current state against those cases to decide if pick will succeed
            grasping_state.object_pose.pose.position.x = grasping_state.gripper_pose.pose.position.x + 0.03;
            grasping_state.object_pose.pose.position.y = grasping_state.gripper_pose.pose.position.y;
            grasping_state.object_pose.pose.position.z = grasping_state.gripper_pose.pose.position.z;
        }
        
        GetObsFromData(grasping_state, grasping_obs, random_num, action);
        */
    }
    else
    {
        std::cout << "Invalid Action " << action << std::endl;
        assert(false);
    }
        
    //PrintState(grasping_state, std::cout);
   //PrintObs(grasping_obs, std::cout);
    
    
    bool validState = IsValidState(grasping_state);
    
    
    //Decide Reward
    GetReward(initial_grasping_state, grasping_state, grasping_obs, action, reward);
    
   // std::cout << "Reward " << reward << std::endl;
    
    //Update next state parameters dependent on previous state
    UpdateNextStateValuesBasedAfterStep(grasping_state,grasping_obs,reward, action);
     if(action == A_PICK || !validState) //Wither pick called or invalid state reached
    {
        return true;
    }
    return false;
}


double RobotInterface::ObsProb(GraspingObservation grasping_obs, const GraspingStateRealArm& grasping_state, int action) const {
        GraspingObservation grasping_obs_expected;
    
    GetObsFromData(grasping_state, grasping_obs_expected, despot::Random::RANDOM.NextDouble(), action);
    
   // PrintObs(grasping_obs_expected);
    double total_distance = 0;
    double finger_weight = 1;
    if(action == A_CLOSE)
    {
        finger_weight = 2;
    }
    double sensor_weight = 4;
    double gripper_position_weight = 1;
    double gripper_orientation_weight = 1;
    
    double tau = finger_weight + sensor_weight + gripper_position_weight + gripper_orientation_weight;
    
    double finger_distance = 0;
    for(int i = 0; i < 4; i++)
    {
        finger_distance = finger_distance + abs(grasping_obs.finger_joint_state[i] - grasping_obs_expected.finger_joint_state[i]);
    }
    finger_distance = finger_distance/4.0;
    
    double sensor_distance = 0;
    for(int i = 0; i < 2; i++)
    {
        sensor_distance = sensor_distance + abs(grasping_obs.touch_sensor_reading[i] - grasping_obs_expected.touch_sensor_reading[i]);
    }
    sensor_distance = sensor_distance/2.0;
    
    double gripper_distance = 0;
    gripper_distance = gripper_distance + pow(grasping_obs.gripper_pose.pose.position.x - grasping_obs_expected.gripper_pose.pose.position.x, 2);
    gripper_distance = gripper_distance + pow(grasping_obs.gripper_pose.pose.position.y - grasping_obs_expected.gripper_pose.pose.position.y, 2);
    gripper_distance = gripper_distance + pow(grasping_obs.gripper_pose.pose.position.z - grasping_obs_expected.gripper_pose.pose.position.z, 2);
    gripper_distance = pow(gripper_distance, 0.5);
    
    double gripper_quaternion_distance = 1 - pow(((grasping_obs.gripper_pose.pose.orientation.x*grasping_obs_expected.gripper_pose.pose.orientation.x)+
                                                  (grasping_obs.gripper_pose.pose.orientation.y*grasping_obs_expected.gripper_pose.pose.orientation.y)+
                                                  (grasping_obs.gripper_pose.pose.orientation.z*grasping_obs_expected.gripper_pose.pose.orientation.z)+
                                                  (grasping_obs.gripper_pose.pose.orientation.w*grasping_obs_expected.gripper_pose.pose.orientation.w)), 2);
    
    total_distance = (finger_distance*finger_weight) + 
                     (sensor_distance*sensor_weight) + 
                     (gripper_distance*gripper_position_weight) + 
                     (gripper_quaternion_distance*gripper_orientation_weight);
    double temperature = 5;
    double prob = pow(2, -1*temperature*total_distance/tau);

    return prob;
}



void RobotInterface::PrintAction(int action, std::ostream& out) const {
    out << "Action is ";
    if (action == A_CLOSE) {
        out << "CLOSE GRIPPER";
    } else if (action == A_OPEN) {
        out << "OPEN GRIPPER";
    }else if (action == A_PICK) {
        out << "PICK";
    }
    else if (action >= A_DECREASE_Y) {
        out << "DECREASE Y by " << get_action_range(action, A_DECREASE_Y);
    } else if (action >= A_INCREASE_Y) {
        out << "INCREASE Y by " << get_action_range(action, A_INCREASE_Y);
    } else if (action >= A_DECREASE_X) {
        out << "DECREASE X by " << get_action_range(action, A_DECREASE_X);
    } else if (action >= A_INCREASE_X) {
        out << "INCREASE X by " << get_action_range(action, A_INCREASE_X);
    }


    out << std::endl;
}

void RobotInterface::PrintObs(const GraspingStateRealArm& state, GraspingObservation& obs, std::ostream& out) const {
    PrintObs(obs,out);
    PrintState(state, out);
}

void RobotInterface::PrintObs(GraspingObservation& grasping_obs, std::ostream& out) const {
 char last_char = '*';
    if(out == std::cout)
    {
        last_char = '\n';
    }
   
    out << grasping_obs.gripper_pose.pose.position.x << " " << 
                        grasping_obs.gripper_pose.pose.position.y << " " <<
                        grasping_obs.gripper_pose.pose.position.z << " " <<
                        grasping_obs.gripper_pose.pose.orientation.x << " " << 
                        grasping_obs.gripper_pose.pose.orientation.y << " " << 
                        grasping_obs.gripper_pose.pose.orientation.z << " " << 
                        grasping_obs.gripper_pose.pose.orientation.w << "|" <<
                        grasping_obs.mico_target_pose.pose.position.x << " " << 
                        grasping_obs.mico_target_pose.pose.position.y << " " <<
                        grasping_obs.mico_target_pose.pose.position.z << " " <<
                        grasping_obs.mico_target_pose.pose.orientation.x << " " << 
                        grasping_obs.mico_target_pose.pose.orientation.y << " " << 
                        grasping_obs.mico_target_pose.pose.orientation.z << " " << 
                        grasping_obs.mico_target_pose.pose.orientation.w << "|" ;
    for(int i = 0; i < 4; i++)
    {
        out << grasping_obs.finger_joint_state[i];
        if(i==3)
        {
            out<<"|";
        }
        else
        {
            out<<" ";
        }
    }
 //out<< std::endl;
  /*  for(int i = 0; i < 48; i++)
    {
        out << grasping_obs.force_values[i].x << " " << 
                grasping_obs.force_values[i].y << " " << 
                grasping_obs.force_values[i].z ;
        if(i==47)
        {
            out<<"|";
        }
        else
        {
            out<<" ";
        }
    }
 //out << std::endl;
 for(int i = 0; i < 48; i++)
    {
        out << grasping_obs.torque_values[i].x << " " << 
                grasping_obs.torque_values[i].y << " " << 
                grasping_obs.torque_values[i].z ;
        if(i==47)
        {
            out<<"*";
        }
        else
        {
            out<<" ";
        }
    }
 */
  for(int i = 0; i < 2; i++)
    {
        out << grasping_obs.touch_sensor_reading[i] ;
        if(i==1)
        {
            out<<last_char;
        }
        else
        {
            out<<" ";
        }
    }
}

void RobotInterface::PrintState(const GraspingStateRealArm& grasping_state, std::ostream& out) const {
    char last_char = '*';
    if(out == std::cout)
    {
        last_char = '\n';
    }
    out << grasping_state.gripper_pose.pose.position.x << " " << 
                        grasping_state.gripper_pose.pose.position.y << " " <<
                        grasping_state.gripper_pose.pose.position.z << " " <<
                        grasping_state.gripper_pose.pose.orientation.x << " " << 
                        grasping_state.gripper_pose.pose.orientation.y << " " << 
                        grasping_state.gripper_pose.pose.orientation.z << " " << 
                        grasping_state.gripper_pose.pose.orientation.w << "|" <<
                        grasping_state.object_pose.pose.position.x << " " << 
                        grasping_state.object_pose.pose.position.y << " " <<
                        grasping_state.object_pose.pose.position.z << " " <<
                        grasping_state.object_pose.pose.orientation.x << " " << 
                        grasping_state.object_pose.pose.orientation.y << " " << 
                        grasping_state.object_pose.pose.orientation.z << " " << 
                        grasping_state.object_pose.pose.orientation.w << "|" ;
    for(int i = 0; i < 4; i++)
    {
        out << grasping_state.finger_joint_state[i];
        if(i==3)
        {
            out<<last_char;
        }
        else
        {
            out<<" ";
        }
    }
    
    
    
   // out << "Object id:"<< grasping_state.object_id ;
    //out << " x y z change (" << grasping_state.x_change << "," << grasping_state.y_change << "," << grasping_state.z_change << ") " ;
    //out << "Object Pose" << grasping_state.object_pose.pose.position.x;
    //out << " Gripper Pose" << grasping_state.gripper_pose.pose.position.x;
    //out << std::endl;
}

void RobotInterface::GenerateGaussianParticleFromState(GraspingStateRealArm& initial_state, std::string type) const {
   
    while(true)
    {
                // the engine for generator samples from a distribution
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            initial_state.object_pose.pose.position.x = Gaussian_Distribution(generator,initial_object_x, 0.03 );
            initial_state.object_pose.pose.position.y = Gaussian_Distribution(generator,initial_object_y, 0.03 );

            
            break;
            //Removing redundant code as this is not called after break
            /*
            if((initial_state.object_pose.pose.position.x < initial_object_x + 0.035) &&
                 (initial_state.object_pose.pose.position.x > initial_object_x - 0.035) &&
                 (initial_state.object_pose.pose.position.y < initial_object_y + 0.035) &&
                 (initial_state.object_pose.pose.position.y > initial_object_y - 0.035))
            {
                break;
            }
            */
      }
      
}

void RobotInterface::GetDefaultStartState(GraspingStateRealArm& initial_state) const {
    int i = initial_gripper_pose_index_x;
    int j = initial_gripper_pose_index_y;
    int object_id = initial_state.object_id;

    initial_state.gripper_pose.pose.position.x = min_x_i + 0.01*i;
    initial_state.gripper_pose.pose.position.y = min_y_i + 0.01*j;
    initial_state.gripper_pose.pose.position.z = initial_gripper_pose_z;
    initial_state.gripper_pose.pose.orientation.x = -0.694327;
    initial_state.gripper_pose.pose.orientation.y = -0.0171483;
    initial_state.gripper_pose.pose.orientation.z = -0.719 ;
    initial_state.gripper_pose.pose.orientation.w = -0.0255881;
    initial_state.object_pose.pose.position.x = initial_object_x;
    initial_state.object_pose.pose.position.y = initial_object_y;
    initial_state.object_pose.pose.position.z = initial_object_pose_z[object_id];
    initial_state.object_pose.pose.orientation.x = -0.0327037 ;
    initial_state.object_pose.pose.orientation.y = 0.0315227;
    initial_state.object_pose.pose.orientation.z = -0.712671 ; 
    initial_state.object_pose.pose.orientation.w = 0.700027;
    initial_state.finger_joint_state[0] = -2.95639e-05 ;
    initial_state.finger_joint_state[1] = 0.00142145;
    initial_state.finger_joint_state[2] = -1.19209e-06 ;
    initial_state.finger_joint_state[3] = -0.00118446 ;
}



double RobotInterface::get_action_range(int action, int action_type) const {
if ((action - action_type) >= (A_DECREASE_X - A_INCREASE_X)) {
        std::cout << "Action " << action << "out of range of action type " << action_type << std::endl;
        assert(false);
    }
    return epsilon * pow(epsilon_multiplier, action - action_type);
}

bool sort_functor(double x1, double y1, double x2, double y2)
{
     if(x1<x2)
    {
        return true;
    }
    else if(x1 == x2)
    {
        return y1<y2;
    }
    else
    {
        return false;
    }
}

bool sort_by_object_relative_pose_initial_state_x (SimulationData data1,SimulationData data2) {
    double x1 = data1.current_object_pose.pose.position.x - data1.current_gripper_pose.pose.position.x;
    //double y1 = data1.current_object_pose.pose.position.y - data1.current_gripper_pose.pose.position.y;
    double x2 = data2.current_object_pose.pose.position.x - data2.current_gripper_pose.pose.position.x;
    //double y2 = data2.current_object_pose.pose.position.y - data2.current_gripper_pose.pose.position.y;
    
   
    return x1 < x2;
}


bool sort_by_object_relative_pose_initial_state_y (SimulationData data1,SimulationData data2) {
    //double x1 = data1.current_object_pose.pose.position.x - data1.current_gripper_pose.pose.position.x;
    double y1 = data1.current_object_pose.pose.position.y - data1.current_gripper_pose.pose.position.y;
    //double x2 = data2.current_object_pose.pose.position.x - data2.current_gripper_pose.pose.position.x;
    double y2 = data2.current_object_pose.pose.position.y - data2.current_gripper_pose.pose.position.y;
    
   
    return y1 < y2;
}

bool sort_by_gripper_pose_initial_state_x (SimulationData data1,SimulationData data2) {
    double x1 = data1.current_gripper_pose.pose.position.x;
    double y1 = data1.current_gripper_pose.pose.position.y;
    double x2 = data2.current_gripper_pose.pose.position.x;
    double y2 = data2.current_gripper_pose.pose.position.y;
    
   
    return x1 < x2;
}

bool sort_by_gripper_pose_initial_state_y (SimulationData data1,SimulationData data2) {
    double x1 = data1.current_gripper_pose.pose.position.x;
    double y1 = data1.current_gripper_pose.pose.position.y;
    double x2 = data2.current_gripper_pose.pose.position.x;
    double y2 = data2.current_gripper_pose.pose.position.y;
    
   
    return y1 < y2;
}

bool sort_by_object_relative_pose_next_state_x (SimulationData data1,SimulationData data2) {
    double x1 = data1.next_object_pose.pose.position.x - data1.next_gripper_pose.pose.position.x;
    //double y1 = data1.current_object_pose.pose.position.y - data1.current_gripper_pose.pose.position.y;
    double x2 = data2.next_object_pose.pose.position.x - data2.next_gripper_pose.pose.position.x;
    //double y2 = data2.current_object_pose.pose.position.y - data2.current_gripper_pose.pose.position.y;
    
   
    return x1 < x2;
}

bool sort_by_object_relative_pose_next_state_y (SimulationData data1,SimulationData data2) {
    //double x1 = data1.current_object_pose.pose.position.x - data1.current_gripper_pose.pose.position.x;
    double y1 = data1.next_object_pose.pose.position.y - data1.next_gripper_pose.pose.position.y;
    //double x2 = data2.current_object_pose.pose.position.x - data2.current_gripper_pose.pose.position.x;
    double y2 = data2.next_object_pose.pose.position.y - data2.next_gripper_pose.pose.position.y;
    
   
    return y1 < y2;
}

bool sort_by_gripper_pose_next_state_x (SimulationData data1,SimulationData data2) {
    double x1 = data1.next_gripper_pose.pose.position.x;
    double y1 = data1.next_gripper_pose.pose.position.y;
    double x2 = data2.next_gripper_pose.pose.position.x;
    double y2 = data2.next_gripper_pose.pose.position.y;
    
   
    return x1 < x2;
}

bool sort_by_gripper_pose_next_state_y (SimulationData data1,SimulationData data2) {
    double x1 = data1.next_gripper_pose.pose.position.x;
    double y1 = data1.next_gripper_pose.pose.position.y;
    double x2 = data2.next_gripper_pose.pose.position.x;
    double y2 = data2.next_gripper_pose.pose.position.y;
    
   
    return y1 < y2;
}

void RobotInterface::GetObsFromData(GraspingStateRealArm current_grasping_state, GraspingObservation& grasping_obs, double random_num, int action, bool debug) const {
    if(action != A_PICK && IsValidState(current_grasping_state)) //State is not terminal
    {
        int gripper_status = GetGripperStatus(current_grasping_state.finger_joint_state);
        if(gripper_status > 0)
        {
            action = A_CLOSE;
        }
       
        bool stateInObjectData = false;
        bool stateInGripperData = false;
    
        int object_id = current_grasping_state.object_id;
        SimulationData tempData;
        std::vector<SimulationData> tempDataVector;
        std::vector<SimulationData> :: iterator x_lower_bound, x_upper_bound, xy_lower_bound, xy_upper_bound;
        //tempData.next_gripper_pose = current_grasping_state.gripper_pose;
        //tempData.next_object_pose = current_grasping_state.object_pose;
    
        //sort(simulationDataCollectionWithObject[action].begin(), simulationDataCollectionWithObject[action].end(), sort_by_object_relative_pose_next_state_x);
        //tempData.next_object_pose.pose.position.x = tempData.next_object_pose.pose.position.x - 0.005;
        //x_lower_bound = lower_bound(simulationDataCollectionWithObject[action].begin(), simulationDataCollectionWithObject[action].end(), tempData, sort_by_object_relative_pose_next_state_x);
        //tempData.next_object_pose.pose.position.x = tempData.next_object_pose.pose.position.x + 0.01;    
        //x_upper_bound = upper_bound(simulationDataCollectionWithObject[action].begin(), simulationDataCollectionWithObject[action].end(), tempData, sort_by_object_relative_pose_next_state_x);
    
        //sort(x_lower_bound, x_upper_bound, sort_by_object_relative_pose_next_state_y);
        //tempData.next_object_pose.pose.position.y = tempData.next_object_pose.pose.position.y - 0.005;
        //xy_lower_bound = lower_bound(x_lower_bound, x_upper_bound, tempData, sort_by_object_relative_pose_next_state_y);
        //tempData.next_object_pose.pose.position.y = tempData.next_object_pose.pose.position.y + 0.01;
        //xy_upper_bound = upper_bound(x_lower_bound, x_upper_bound, tempData, sort_by_object_relative_pose_next_state_y);

        int len_simulation_data = ((std::vector<SimulationData>)(simulationDataCollectionWithObject[object_id][action])).size();

        for(int i = 0; i < len_simulation_data; i++)
        {
            double x1 = simulationDataCollectionWithObject[object_id][action][i].next_object_pose.pose.position.x - simulationDataCollectionWithObject[object_id][action][i].next_gripper_pose.pose.position.x;
            double x2 = current_grasping_state.object_pose.pose.position.x - current_grasping_state.gripper_pose.pose.position.x;
            if(abs(x1-x2) <= 0.005)
            {
                double y1 = simulationDataCollectionWithObject[object_id][action][i].next_object_pose.pose.position.y - simulationDataCollectionWithObject[object_id][action][i].next_gripper_pose.pose.position.y;
                double y2 = current_grasping_state.object_pose.pose.position.y - current_grasping_state.gripper_pose.pose.position.y;
                if(abs(y1-y2) <= 0.005)
                {
                    tempDataVector.push_back(simulationDataCollectionWithObject[object_id][action][i]);
                }
            }

        }
        xy_lower_bound = tempDataVector.begin();
        xy_upper_bound = tempDataVector.end();
        // check if x and y exists in simulation data with object
        if(std::distance(xy_lower_bound,xy_upper_bound) > 0)
        {   stateInObjectData = true;

            //Get the closest gripper state
            double min_distance = 100000;

            for(std::vector<SimulationData> :: iterator it = xy_lower_bound; it < xy_upper_bound; it++)
            {
                double temp_distance = 0;
                /*if(action < 8 || action > 15)
                { // if move in x check distance between only x
                temp_distance = pow(((*it).next_gripper_pose.pose.position.x - current_grasping_state.gripper_pose.pose.position.x), 2);
                }
                if (action >= 8)
                {
                    // if move in y check distance between only y 
                    temp_distance = temp_distance + pow(((*it).next_gripper_pose.pose.position.y - current_grasping_state.gripper_pose.pose.position.y), 2);
                }*/
                double x1 = (*it).next_gripper_pose.pose.position.x - (*it).next_object_pose.pose.position.x;
                double x2 = current_grasping_state.gripper_pose.pose.position.x - current_grasping_state.object_pose.pose.position.x;
                temp_distance = temp_distance + pow(x1 - x2, 2);
                double y1 = (*it).next_gripper_pose.pose.position.y - (*it).next_object_pose.pose.position.y;
                double y2 = current_grasping_state.gripper_pose.pose.position.y - current_grasping_state.object_pose.pose.position.y;
                temp_distance = temp_distance + pow(y1 - y2, 2);
                temp_distance = pow(temp_distance, 0.5);
                if(temp_distance < min_distance)
                {
                    min_distance = temp_distance;
                    tempData = (*it);
                }
            }


        }
        else
        {
            sort(simulationDataCollectionWithoutObject[action].begin(), simulationDataCollectionWithoutObject[action].end(), sort_by_gripper_pose_next_state_x);
            tempData.next_gripper_pose.pose.position.x = tempData.next_gripper_pose.pose.position.x - 0.005;
            x_lower_bound = lower_bound(simulationDataCollectionWithoutObject[action].begin(), simulationDataCollectionWithoutObject[action].end(), tempData, sort_by_gripper_pose_next_state_x);
            tempData.next_gripper_pose.pose.position.x = tempData.next_gripper_pose.pose.position.x + 0.01;    
            x_upper_bound = upper_bound(simulationDataCollectionWithoutObject[action].begin(), simulationDataCollectionWithoutObject[action].end(), tempData, sort_by_gripper_pose_next_state_x);
    
            sort(x_lower_bound, x_upper_bound, sort_by_gripper_pose_initial_state_x);
            tempData.next_gripper_pose.pose.position.y = tempData.next_gripper_pose.pose.position.y - 0.005;
            xy_lower_bound = lower_bound(x_lower_bound, x_upper_bound, tempData, sort_by_gripper_pose_next_state_y);
            tempData.next_gripper_pose.pose.position.y = tempData.next_gripper_pose.pose.position.y + 0.01;
            xy_upper_bound = upper_bound(x_lower_bound, x_upper_bound, tempData, sort_by_gripper_pose_next_state_y);  
            
            // check if x and y exists in simulation data without object
            if(std::distance(xy_lower_bound,xy_upper_bound) > 0)
            {   stateInGripperData = true;
                //Get the closest gripper state
                double min_distance = 100000;

                for(std::vector<SimulationData> :: iterator it = xy_lower_bound; it < xy_upper_bound; it++)
                {
                    double temp_distance = 0;
                    if(action < 8 || action > 15)
                    { // if move in x check distance between only x
                        temp_distance = pow(((*it).next_gripper_pose.pose.position.x - current_grasping_state.gripper_pose.pose.position.x), 2);
                    }
                    if (action >= 8)
                    {
                        // if move in y check distance between only y 
                        temp_distance = temp_distance + pow(((*it).next_gripper_pose.pose.position.y - current_grasping_state.gripper_pose.pose.position.y), 2);
                    }

                    temp_distance = pow(temp_distance, 0.5);
                    if(temp_distance < min_distance)
                    {
                        min_distance = temp_distance;
                        tempData = (*it);
                    }
                }

            }
        }
        if(stateInObjectData || stateInGripperData)
        {
            grasping_obs.gripper_pose = current_grasping_state.gripper_pose;
            grasping_obs.mico_target_pose = tempData.mico_target_pose; //Does not matter for now
            for(int i = 0; i < 4; i++)
            {
                grasping_obs.finger_joint_state[i] = tempData.next_finger_joint_state[i];
            }
            for(int i = 0; i < 2; i++)
            {
                grasping_obs.touch_sensor_reading[i] = tempData.touch_sensor_reading[i];
            }
            //ConvertObs48ToObs2(tempData.touch_sensor_reading, grasping_obs.touch_sensor_reading);

        }
        else
        {
            GetObsUsingDefaultFunction(current_grasping_state, grasping_obs);
        }
        
    }
    else
    {
        GetObsUsingDefaultFunction(current_grasping_state, grasping_obs);
    }
}

//Return value 0 if open
//Return value 1 if closed without object inside it
//Return value 2 if closed with object inside it
int RobotInterface::GetGripperStatus(double finger_joint_state[]) const {
double degree_readings[4];
    for(int i = 0; i < 4; i++)
    {
        degree_readings[i] = finger_joint_state[i]*180/3.14;
    }
    
    if(degree_readings[0] > 22 && //Changed from 20 to 22 looking at the data from 7cm cylinder object
       degree_readings[1] > 85 &&
       degree_readings[2] > 22 && //Changed from 20 to 22 looking at the data from 7cm cylinder object
       degree_readings[3] > 85)
    {//joint1 > 20 joint2 > 85 
        return 1;
    }
   
    if(//degree_readings[0] > 2 &&
       degree_readings[1] > 25 && //Changed from 45 to 25 looking at data
       //degree_readings[2] > 2 &&
       degree_readings[3] > 25)  //Changed from 45 to 25 looking at data
    {//joint1 > 2 joint2 > 45 return 2
        return 2;
    }
    
    
    return 0;
}

void RobotInterface::UpdateNextStateValuesBasedAfterStep(GraspingStateRealArm& grasping_state, GraspingObservation grasping_obs, double reward, int action) const {
    grasping_state.pre_action = action;
    CheckTouch(grasping_obs.touch_sensor_reading, grasping_state.touch);
    int gripper_status = GetGripperStatus(grasping_state.finger_joint_state);
    if (gripper_status == 2){
        if(reward < -1)
        {
            gripper_status = 1;
        }
    }
    grasping_state.gripper_status = gripper_status;
}


void RobotInterface::GetNextStateAndObsFromData(GraspingStateRealArm current_grasping_state, GraspingStateRealArm& grasping_state, GraspingObservation& grasping_obs, int action, bool debug) const {
    bool stateInObjectData = false;
    bool stateInGripperData = false;
    int object_id = current_grasping_state.object_id;
    SimulationData tempData;
    std::vector<SimulationData> tempDataVector;
    std::vector<SimulationData> :: iterator x_lower_bound, x_upper_bound, xy_lower_bound, xy_upper_bound;
    //tempData.current_gripper_pose = current_grasping_state.gripper_pose;
    //tempData.current_object_pose = current_grasping_state.object_pose;
    double step_start_t1 = despot::get_time_second();
    for(int i = 0; i < simulationDataCollectionWithObject[object_id][action].size(); i++)
    {
        double x1 = simulationDataCollectionWithObject[object_id][action][i].current_object_pose.pose.position.x - simulationDataCollectionWithObject[object_id][action][i].current_gripper_pose.pose.position.x;
        double x2 = current_grasping_state.object_pose.pose.position.x - current_grasping_state.gripper_pose.pose.position.x;
        if(abs(x1-x2) <= 0.005)
        {
            double y1 = simulationDataCollectionWithObject[object_id][action][i].current_object_pose.pose.position.y - simulationDataCollectionWithObject[object_id][action][i].current_gripper_pose.pose.position.y;
            double y2 = current_grasping_state.object_pose.pose.position.y - current_grasping_state.gripper_pose.pose.position.y;
            if(abs(y1-y2) <= 0.005)
            {
               /* if(debug)
                {
                    std::cout << "Pushing particle with difference(" << abs(x1-x2) << ", " << abs(y1-y2) << ")" << std::endl;
                    std::cout << "x1 = " << x1 << " x2 = " << x2 << " y1 = " << y1 << " y2 = " << y2 << std::endl;
                    simulationDataCollectionWithObject[action][i].PrintSimulationData();
                    PrintState(current_grasping_state);
                }*/
                tempDataVector.push_back(simulationDataCollectionWithObject[object_id][action][i]);
            }
        }
       
    }
    //sort(simulationDataCollectionWithObject[action].begin(), simulationDataCollectionWithObject[action].end(), sort_by_object_relative_pose_initial_state_x);
     double step_start_t1_1 = despot::get_time_second();
     //tempData.current_object_pose.pose.position.x = tempData.current_object_pose.pose.position.x - 0.005;
    //x_lower_bound = lower_bound(simulationDataCollectionWithObject[action].begin(), simulationDataCollectionWithObject[action].end(), tempData, sort_by_object_relative_pose_initial_state_x);
    //tempData.current_object_pose.pose.position.x = tempData.current_object_pose.pose.position.x + 0.01;    
    //x_upper_bound = upper_bound(simulationDataCollectionWithObject[action].begin(), simulationDataCollectionWithObject[action].end(), tempData, sort_by_object_relative_pose_initial_state_x);

    double step_start_t2 = despot::get_time_second();
    //sort(x_lower_bound, x_upper_bound, sort_by_object_relative_pose_initial_state_y);
    
    //tempData.current_object_pose.pose.position.y = tempData.current_object_pose.pose.position.y - 0.005;
    //xy_lower_bound = lower_bound(x_lower_bound, x_upper_bound, tempData, sort_by_object_relative_pose_initial_state_y);
    //tempData.current_object_pose.pose.position.y = tempData.current_object_pose.pose.position.y + 0.01;
    //xy_upper_bound = upper_bound(x_lower_bound, x_upper_bound, tempData, sort_by_object_relative_pose_initial_state_y);
    
    double step_start_t3 = despot::get_time_second();
    xy_lower_bound = tempDataVector.begin();
    xy_upper_bound = tempDataVector.end();
    // check if x and y exists in simulation data with object
    if(std::distance(xy_lower_bound,xy_upper_bound) > 0)
    {   stateInObjectData = true;
    //std::cout << "No of objects : " << xy_upper_bound - xy_lower_bound << std::endl;
        //Get the closest gripper state
        double min_distance = 100000;
        
        for(std::vector<SimulationData> :: iterator it = xy_lower_bound; it < xy_upper_bound; it++)
        {
            double temp_distance = 0;
            if(action == A_PICK)
            {
                for(int i = 0; i < 4; i++)
                {
                temp_distance = pow(((*it).current_finger_joint_state[i] - current_grasping_state.finger_joint_state[i]), 2);
                }
            }
            else{
                /*if(action < 8 || action > 15)
                { // if move in x check distance between only x
                temp_distance = pow(((*it).current_gripper_pose.pose.position.x - current_grasping_state.gripper_pose.pose.position.x), 2);
                }
                if (action >= 8)
                {
                    // if move in y check distance between only y 
                    temp_distance = temp_distance + pow(((*it).current_gripper_pose.pose.position.y - current_grasping_state.gripper_pose.pose.position.y), 2);
                }*/
                double x1 = (*it).current_gripper_pose.pose.position.x - (*it).current_object_pose.pose.position.x;
                double x2 = current_grasping_state.gripper_pose.pose.position.x - current_grasping_state.object_pose.pose.position.x;
                temp_distance = temp_distance + pow(x1 - x2, 2);
                double y1 = (*it).current_gripper_pose.pose.position.y - (*it).current_object_pose.pose.position.y;
                double y2 = current_grasping_state.gripper_pose.pose.position.y - current_grasping_state.object_pose.pose.position.y;
                temp_distance = temp_distance + pow(y1 - y2, 2);
            }
            temp_distance = pow(temp_distance, 0.5);
            if(temp_distance < min_distance)
            {
                min_distance = temp_distance;
                tempData = (*it);
            }
        }
        
            
    }
    else
    {
        
        std::sort(simulationDataCollectionWithoutObject[action].begin(), simulationDataCollectionWithoutObject[action].end(), sort_by_gripper_pose_initial_state_x);
        tempData.current_gripper_pose.pose.position.x = tempData.current_gripper_pose.pose.position.x - 0.005;
        x_lower_bound = std::lower_bound(simulationDataCollectionWithoutObject[action].begin(), simulationDataCollectionWithoutObject[action].end(), tempData, sort_by_gripper_pose_initial_state_x);
        tempData.current_gripper_pose.pose.position.x = tempData.current_gripper_pose.pose.position.x + 0.01;    
        x_upper_bound = std::upper_bound(simulationDataCollectionWithoutObject[action].begin(), simulationDataCollectionWithoutObject[action].end(), tempData, sort_by_gripper_pose_initial_state_x);
    
        std::sort(x_lower_bound, x_upper_bound, sort_by_gripper_pose_initial_state_x);
        tempData.current_gripper_pose.pose.position.y = tempData.current_gripper_pose.pose.position.y - 0.005;
        xy_lower_bound = std::lower_bound(x_lower_bound, x_upper_bound, tempData, sort_by_gripper_pose_initial_state_y);
        tempData.current_gripper_pose.pose.position.y = tempData.current_gripper_pose.pose.position.y + 0.01;
        xy_upper_bound = std::upper_bound(x_lower_bound, x_upper_bound, tempData, sort_by_gripper_pose_initial_state_y);
    

        // check if x and y exists in simulation data without object
        if(std::distance(xy_lower_bound,xy_upper_bound) > 0)
        {   stateInGripperData = true;
            //Get the closest gripper state
            double min_distance = 100000;
        
            for(std::vector<SimulationData> :: iterator it = xy_lower_bound; it < xy_upper_bound; it++)
            {
                double temp_distance = 0;
                if(action < A_INCREASE_Y || action > A_CLOSE-1)
                { // if move in x check distance between only x
                    temp_distance = pow(((*it).current_gripper_pose.pose.position.x - current_grasping_state.gripper_pose.pose.position.x), 2);
                }
                if (action >= A_INCREASE_Y)
                {
                    // if move in y check distance between only y 
                    temp_distance = temp_distance + pow(((*it).current_gripper_pose.pose.position.y - current_grasping_state.gripper_pose.pose.position.y), 2);
                }
            
                temp_distance = pow(temp_distance, 0.5);
                if(temp_distance < min_distance)
                {
                    min_distance = temp_distance;
                    tempData = (*it);
                }
            }
            
        }
    
    }
    double step_start_t4 = despot::get_time_second();
    
    if(stateInObjectData || stateInGripperData)
    {
        if(debug)
        {
            std::cout << "State being updated from ";
            if (stateInObjectData){
                std::cout << "object ";
            }
            if (stateInGripperData) {
                std::cout << "gripper ";
            }
            std::cout << "data\n";
            tempData.PrintSimulationData();
        }
        //Update next state
        //Need to update z for all actions to determine invalid state
        //if(action == A_PICK)
        //{
           grasping_state.gripper_pose.pose.position.z = grasping_state.gripper_pose.pose.position.z + tempData.next_gripper_pose.pose.position.z - tempData.current_gripper_pose.pose.position.z;
           grasping_state.object_pose.pose.position.z = grasping_state.gripper_pose.pose.position.z + tempData.next_object_pose.pose.position.z - tempData.next_gripper_pose.pose.position.z;
        
        //}

        double next_gripper_pose_boundary_margin_x = 0.0;
        double next_gripper_pose_boundary_margin_y = 0.0;
        
        if(action < A_CLOSE)
        {
            int action_offset = (action/(A_DECREASE_X - A_INCREASE_X)) * (A_DECREASE_X - A_INCREASE_X);
            double action_range = get_action_range(action, action_offset);
            int on_bits[2];
            if(action < A_DECREASE_X) //action is increase x
            {
                if((tempData.next_gripper_pose.pose.position.x - tempData.current_gripper_pose.pose.position.x ) < action_range)
                {
                    if(tempData.next_gripper_pose.pose.position.x > max_x_i)
                    {
                        if(!CheckTouch(tempData.touch_sensor_reading, on_bits))
                        {
                            next_gripper_pose_boundary_margin_x = action_range - (tempData.next_gripper_pose.pose.position.x - tempData.current_gripper_pose.pose.position.x );
                        }
                    }
                }
            }
            else if (action < A_INCREASE_Y) //action is decrease x
            {
                if((-tempData.next_gripper_pose.pose.position.x + tempData.current_gripper_pose.pose.position.x ) < action_range)
                {
                    if(tempData.next_gripper_pose.pose.position.x < min_x_i)
                    {
                        if(!CheckTouch(tempData.touch_sensor_reading, on_bits))
                        {
                            next_gripper_pose_boundary_margin_x = - action_range + (-tempData.next_gripper_pose.pose.position.x + tempData.current_gripper_pose.pose.position.x );
                        }
                    }
                }
            }
            else if (action < A_DECREASE_Y) //action is increase y
            {
                if((tempData.next_gripper_pose.pose.position.y - tempData.current_gripper_pose.pose.position.y ) < action_range)
                {
                    if(tempData.next_gripper_pose.pose.position.y > max_y_i)
                    {
                        if(!CheckTouch(tempData.touch_sensor_reading, on_bits))
                        {
                            next_gripper_pose_boundary_margin_y = action_range - (tempData.next_gripper_pose.pose.position.y - tempData.current_gripper_pose.pose.position.y );
                        }
                    }
                }
            }
            else //action is decrease y
            {
                if((-tempData.next_gripper_pose.pose.position.y + tempData.current_gripper_pose.pose.position.y ) < action_range)
                {
                    if(tempData.next_gripper_pose.pose.position.y < min_y_i)
                    {
                        if(!CheckTouch(tempData.touch_sensor_reading, on_bits))
                        {
                            next_gripper_pose_boundary_margin_y = - action_range + (-tempData.next_gripper_pose.pose.position.y + tempData.current_gripper_pose.pose.position.y );
                        }
                    }
                }
            }
        }
            
            
            
            
        grasping_state.gripper_pose.pose.position.x = grasping_state.gripper_pose.pose.position.x + next_gripper_pose_boundary_margin_x + tempData.next_gripper_pose.pose.position.x - tempData.current_gripper_pose.pose.position.x;
        grasping_state.gripper_pose.pose.position.y = grasping_state.gripper_pose.pose.position.y + next_gripper_pose_boundary_margin_y + tempData.next_gripper_pose.pose.position.y - tempData.current_gripper_pose.pose.position.y;
        
        
        grasping_state.object_pose.pose.position.x = grasping_state.gripper_pose.pose.position.x + tempData.next_object_pose.pose.position.x - (tempData.next_gripper_pose.pose.position.x + next_gripper_pose_boundary_margin_x );
        grasping_state.object_pose.pose.position.y = grasping_state.gripper_pose.pose.position.y + tempData.next_object_pose.pose.position.y - (tempData.next_gripper_pose.pose.position.y + next_gripper_pose_boundary_margin_y );
        
        CheckAndUpdateGripperBounds(grasping_state, action);
           
           
        //Update next observation
        grasping_obs.gripper_pose = grasping_state.gripper_pose;
        grasping_obs.mico_target_pose = tempData.mico_target_pose; //No need
        for(int i = 0; i < 4; i++)
        {
            grasping_state.finger_joint_state[i] = tempData.next_finger_joint_state[i];
            grasping_obs.finger_joint_state[i] = tempData.next_finger_joint_state[i];
        }
        for(int i = 0; i < 2; i++)
        {
            grasping_obs.touch_sensor_reading[i] = tempData.touch_sensor_reading[i];
        }
        //ConvertObs48ToObs2(tempData.touch_sensor_reading, grasping_obs.touch_sensor_reading);
        
    }
    else
    {
         if(debug)
        {
            std::cout << "State being updated from default function" << std::endl;
        }
        GetNextStateAndObsUsingDefaulFunction(grasping_state, grasping_obs, action);
    }
    double step_start_t5 = despot::get_time_second();
    //std::cout << "T112 " << step_start_t1_1 - step_start_t1 << " T212 " << step_start_t2 - step_start_t1_1 <<
    //        " T23 " << step_start_t3 - step_start_t2 << 
    //       " T34 " << step_start_t4 - step_start_t3 << " T45 " << step_start_t5 - step_start_t4 << std::endl;  

}

void RobotInterface::GetNextStateAndObsUsingDefaulFunction(GraspingStateRealArm& grasping_state, GraspingObservation& grasping_obs, int action, bool debug) const {
 
    if(action < A_CLOSE)
    {
        int action_offset = (action/(A_DECREASE_X - A_INCREASE_X)) * (A_DECREASE_X - A_INCREASE_X);
        if(action < A_DECREASE_X)
        {
            grasping_state.gripper_pose.pose.position.x = grasping_state.gripper_pose.pose.position.x + get_action_range(action, action_offset);
        }
        else if (action < A_INCREASE_Y)
        {
            grasping_state.gripper_pose.pose.position.x = grasping_state.gripper_pose.pose.position.x - get_action_range(action, action_offset);
            
        }
        else if (action < A_DECREASE_Y)
        {
            grasping_state.gripper_pose.pose.position.y = grasping_state.gripper_pose.pose.position.y + get_action_range(action,action_offset);
            
        }
        else if (action < A_CLOSE)
        {
            grasping_state.gripper_pose.pose.position.y = grasping_state.gripper_pose.pose.position.y - get_action_range(action, action_offset);
            
        }
    }
    else if (action == A_CLOSE)
    {
        grasping_state.finger_joint_state[0] = 22.5*3.14/180;
        grasping_state.finger_joint_state[1] = 90*3.14/180;
        grasping_state.finger_joint_state[2] = 22.5*3.14/180;
        grasping_state.finger_joint_state[3] = 90*3.14/180;
    }
    else if (action == A_OPEN)
    {
        grasping_state.finger_joint_state[0] = 0*3.14/180;
        grasping_state.finger_joint_state[1] = 0*3.14/180;
        grasping_state.finger_joint_state[2] = 0*3.14/180;
        grasping_state.finger_joint_state[3] = 0*3.14/180;
    }
    else if(action == A_PICK)
    {
        GetDefaultPickState(grasping_state);
        
    }
    else
    {
        std::cout << "Underfined for this action" << std::endl;
        assert(false);
    }
    
    CheckAndUpdateGripperBounds(grasping_state, action);
   
    
    GetObsUsingDefaultFunction(grasping_state, grasping_obs);
}

void RobotInterface::GetObsUsingDefaultFunction(GraspingStateRealArm grasping_state, GraspingObservation& grasping_obs, bool debug) const {
    //Gripper pose
    grasping_obs.gripper_pose = grasping_state.gripper_pose;
                    
    //Mico target pose
    grasping_obs.mico_target_pose = grasping_state.gripper_pose;
    
    //Finger Joints
    for(int i = 0; i < 4; i++)
    {
        grasping_obs.finger_joint_state[i] = grasping_state.finger_joint_state[i];
    }
    
    int gripper_status = GetGripperStatus(grasping_state.finger_joint_state);
    
    double touch_sensor_reading[48];
    for(int i = 0; i < 48; i++)
        {
        if(gripper_status == 0)
            {
                touch_sensor_reading[i] = touch_sensor_mean[i];
            }
        if(gripper_status == 1)
            {
                touch_sensor_reading[i] = touch_sensor_mean_closed_without_object[i];
            }
        if(gripper_status == 2)
            {
                touch_sensor_reading[i] = touch_sensor_mean_closed_with_object[i];
            }
        }
    ConvertObs48ToObs2(touch_sensor_reading, grasping_obs.touch_sensor_reading);
}

void RobotInterface::GetReward(GraspingStateRealArm initial_grasping_state, GraspingStateRealArm grasping_state, GraspingObservation grasping_obs, int action, double& reward) const {
bool validState = IsValidState(grasping_state);
    if(action == A_PICK)
    {
        if(IsValidPick(grasping_state, grasping_obs))
        {
            reward = pick_reward;
        }
        else
        {
            reward = pick_penalty;
        }
    }
    else
    {
        if(validState)
        {
            int initial_gripper_status = GetGripperStatus(initial_grasping_state.finger_joint_state);
            if((initial_gripper_status == 0 && action == A_OPEN) || 
              (initial_gripper_status !=0 && action <= A_CLOSE)  ||
              (initial_grasping_state.gripper_pose.pose.position.x <=min_x_i && (action >= A_DECREASE_X && action < A_INCREASE_Y)) ||
              (initial_grasping_state.gripper_pose.pose.position.x >=max_x_i && (action >= A_INCREASE_X && action < A_DECREASE_X)) ||
              (initial_grasping_state.gripper_pose.pose.position.y <=min_y_i && (action >= A_DECREASE_Y && action < A_CLOSE) )||
              (initial_grasping_state.gripper_pose.pose.position.y >=max_y_i && (action >= A_INCREASE_Y && action < A_DECREASE_Y) )
                    
              )
            {//Disallow open action when gripper is open and move actions when gripper is close
             //Other forbidden actions
                reward = -1*pick_reward;
            }
            else
            {
            
            
            
                int on_bits[2];
                bool touchDetected = CheckTouch(grasping_obs.touch_sensor_reading, on_bits);
                if(touchDetected)
                {
                
                
                    if(separate_close_reward)
                    {
                        int gripper_status = GetGripperStatus(grasping_state.finger_joint_state);
                        if(gripper_status == 0) //gripper is open
                        {
                            reward = -0.5;
                            //TODO no reward if touch due to touching the wall
                        }
                        else
                        {//check grasp stability if action was close gripper
                            if (action==A_CLOSE)
                            {
                                GetRewardBasedOnGraspStability(grasping_state, grasping_obs, reward);
                            }
                            else
                            {
                                reward = -1;
                            }
                        }
                 
                    }
                    else
                    {
                        reward = -0.5;
                    }
                    /*if(gripper_status == 1) // Touching without object
                    {
                    reward = -1; 
                    }
                    if(gripper_status == 2)
                    {
                    reward = 1;
                    }*/
                
                }
            
                else
                {
                    reward = -1;
                }
            }
        }
        else
        {
            reward = invalid_state_penalty;
        }
       
    }
}

//Conert 48 sensor observation to 2 sensor observation on front fingers by taking max
void RobotInterface::ConvertObs48ToObs2(double current_sensor_values[], double on_bits[]) const {
    
    
    for(int j = 0; j < 2; j ++)
    {
        on_bits[j] = 0.0;
        for(int i = 12; i < 24; i++)
        {
        
            if(on_bits[j] < current_sensor_values[i + j*24])
            {
                on_bits[j] = current_sensor_values[i + j*24];
            }
        }
    }
}






