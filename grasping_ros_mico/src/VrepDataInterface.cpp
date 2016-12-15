/* 
 * File:   VrepDataInterface.cpp
 * Author: neha
 * 
 * Created on November 11, 2016, 6:30 PM
 */

#include "VrepDataInterface.h"
#include "RobotInterface.h"
#include <chrono>

VrepDataInterface::VrepDataInterface(int start_state_index_) : start_state_index(start_state_index_){
}

VrepDataInterface::VrepDataInterface(const VrepDataInterface& orig) {
}

VrepDataInterface::~VrepDataInterface() {
}


void VrepDataInterface::CheckAndUpdateGripperBounds(GraspingStateRealArm& grasping_state, int action) const {
    if(action != A_PICK)
    {
    if(grasping_state.gripper_pose.pose.position.x < min_x_i)
    {
        grasping_state.gripper_pose.pose.position.x= min_x_i;
    }
    
    if(grasping_state.gripper_pose.pose.position.x > max_x_i)
    {
        grasping_state.gripper_pose.pose.position.x= max_x_i;
    }
    
    //if(grasping_state.gripper_pose.pose.position.x < gripper_in_x_i)
    if(grasping_state.gripper_pose.pose.position.x <= max_x_i)
    {
        if(grasping_state.gripper_pose.pose.position.y > max_y_i - gripper_out_y_diff)
        {
            grasping_state.gripper_pose.pose.position.y= max_y_i - gripper_out_y_diff;
        }
        if(grasping_state.gripper_pose.pose.position.y < min_y_i + gripper_out_y_diff)
        {
            grasping_state.gripper_pose.pose.position.y= min_y_i + gripper_out_y_diff;
        }
        
    }
    else
    {
        if(grasping_state.gripper_pose.pose.position.y > max_y_i)
        {
             grasping_state.gripper_pose.pose.position.y= max_y_i;
        }
        if(grasping_state.gripper_pose.pose.position.y < min_y_i)
        {
             grasping_state.gripper_pose.pose.position.y= min_y_i;
        }
    }
    }
}

bool VrepDataInterface::CheckTouch(double current_sensor_values[], int on_bits[], int size) const {
//return false;
    bool touchDetected = false;
    for(int i = 0; i < size; i++)
    {
        on_bits[i] = 0;
        //if(current_sensor_values[i] > (touch_sensor_mean[i] + (3*touch_sensor_std[i])))
        if(current_sensor_values[i] > 0.35)
        {
            touchDetected = true;
            on_bits[i] = 1;
        }
    }
    
    return touchDetected;
}

void VrepDataInterface::CreateStartState(GraspingStateRealArm& initial_state, std::string type) const {
    
    int i = 0;
    int j = 7;

    initial_state.gripper_pose.pose.position.x = min_x_i + 0.01*i;
    initial_state.gripper_pose.pose.position.y = min_y_i + 0.01*j;
    initial_state.gripper_pose.pose.position.z = 1.73337;
    initial_state.gripper_pose.pose.orientation.x = -0.694327;
    initial_state.gripper_pose.pose.orientation.y = -0.0171483;
    initial_state.gripper_pose.pose.orientation.z = -0.719 ;
    initial_state.gripper_pose.pose.orientation.w = -0.0255881;
    initial_state.object_pose.pose.position.x = 0.498689;
    initial_state.object_pose.pose.position.y = 0.148582;
    initial_state.object_pose.pose.position.z = 1.7066;
    initial_state.object_pose.pose.orientation.x = -0.0327037 ;
    initial_state.object_pose.pose.orientation.y = 0.0315227;
    initial_state.object_pose.pose.orientation.z = -0.712671 ; 
    initial_state.object_pose.pose.orientation.w = 0.700027;
    initial_state.finger_joint_state[0] = -2.95639e-05 ;
    initial_state.finger_joint_state[1] = 0.00142145;
    initial_state.finger_joint_state[2] = -1.19209e-06 ;
    initial_state.finger_joint_state[3] = -0.00118446 ;
    
    
   /* if (start_state_index >= 0 ) 
    {
        std::cout << "Start_state index is " << start_state_index << std::endl;
        
        std::vector<GraspingStateRealArm> initial_states =  InitialStartStateParticles(initial_state);
        std::cout << "Particle size is " <<  initial_states.size()<< std::endl;
        int ii = start_state_index % initial_states.size();
        initial_state.object_pose.pose.position.x = initial_states[ii].object_pose.pose.position.x ;
        initial_state.object_pose.pose.position.y = initial_states[ii].object_pose.pose.position.y ;
        //initial_state = initial_states[i];
    //return  initial_states[ii];
    }
    else
    {
*/        while(true){
            
                // the engine for generator samples from a distribution
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            initial_state.object_pose.pose.position.x = Gaussian_Distribution(generator,0.498689, 0.03 );
            initial_state.object_pose.pose.position.y = Gaussian_Distribution(generator,0.148582, 0.03 );
            if(IsValidState(initial_state))
            {
                break;
            }
            
  //      }
    }
    
    
        
    

}

std::vector<GraspingStateRealArm> VrepDataInterface::InitialStartStateParticles(const GraspingStateRealArm start) const
 {
   
    //cout << "In initial belief" << endl;
    std::vector<GraspingStateRealArm> particles;
    int num_particles = 0;
    
    
    for(int i = 0; i < 10; i++)
    {
        for(int j = 0; j < 10; j++)
        {
           
            GraspingStateRealArm grasping_state(start);
            
            grasping_state.object_pose.pose.position.y = min_y_o + (j*(max_y_o - min_y_o)/9.0);
            grasping_state.object_pose.pose.position.x = min_x_o + (i*(max_x_o - min_x_o)/9.0);
            if(IsValidState(grasping_state))
            {
               particles.push_back(grasping_state);
               num_particles = num_particles + 1; 
            }
            
        }
    }
    std::cout << "Num particles : " << num_particles << std::endl;
    return particles;
}



void VrepDataInterface::GetDefaultPickState(GraspingStateRealArm& grasping_state) const {
grasping_state.gripper_pose.pose.position.z = grasping_state.gripper_pose.pose.position.z + pick_z_diff;
        grasping_state.gripper_pose.pose.position.x =  pick_x_val; 
        grasping_state.gripper_pose.pose.position.y =  pick_y_val;
        int gripper_status = GetGripperStatus(grasping_state.finger_joint_state);
        if(gripper_status == 2) //Object is inside gripper and gripper is closed
        {

            grasping_state.object_pose.pose.position.x = grasping_state.gripper_pose.pose.position.x + 0.03;
            grasping_state.object_pose.pose.position.y = grasping_state.gripper_pose.pose.position.y;
            grasping_state.object_pose.pose.position.z = grasping_state.gripper_pose.pose.position.z;
        }
}

void VrepDataInterface::GetRewardBasedOnGraspStability(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs, double& reward) const {
    bool grasp_stable = true;
    
    //grasp stability criteria 1
    /*double pick_reward;
    Step(grasping_state,Random::RANDOM.NextDouble(), A_PICK, pick_reward, grasping_obs);
    if(pick_reward == 20)
    {
        grasp_stable = true;
    }
    else
    {
        grasp_stable = false;
    }
    */
    
    //grasp stability criteria 2
    int gripper_status = GetGripperStatus(grasping_state.finger_joint_state);
    if (gripper_status ==2)
    {
        grasp_stable = true;
    }
    else
    {
        grasp_stable = false;
    }
    
    //TODO have stability criteria based on regression model learned from grasp success and joint angle and touch readings
    if(grasp_stable)
    {
        reward = -0.5;
    }
    else
    {
        reward = -1.5;
    }
}

bool VrepDataInterface::IsValidPick(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs) const {
bool isValidPick = true;
    
    //if object and tip are far from each other set false
    double distance = 0;
    distance = distance + pow(grasping_state.gripper_pose.pose.position.x - grasping_state.object_pose.pose.position.x, 2);
    distance = distance + pow(grasping_state.gripper_pose.pose.position.y - grasping_state.object_pose.pose.position.y, 2);
    distance = distance + pow(grasping_state.gripper_pose.pose.position.z - grasping_state.object_pose.pose.position.z, 2);
    distance = pow(distance, 0.5);
    if(distance > 0.08)
    {
        isValidPick= false;
    }
            
    
    // if target and tip are far from each other set false
    distance = 0;
    distance = distance + pow(grasping_state.gripper_pose.pose.position.x - grasping_obs.mico_target_pose.pose.position.x, 2);
    distance = distance + pow(grasping_state.gripper_pose.pose.position.y - grasping_obs.mico_target_pose.pose.position.y, 2);
    distance = distance + pow(grasping_state.gripper_pose.pose.position.z - grasping_obs.mico_target_pose.pose.position.z, 2);
    distance = pow(distance, 0.5);
    if(distance > 0.03)
    {
        isValidPick = false;
    }
      
    
    return isValidPick;
}

bool VrepDataInterface::IsValidState(GraspingStateRealArm grasping_state) const {
    bool isValid = true;
    //Check gripper is in its range
    if(grasping_state.gripper_pose.pose.position.x < min_x_i - 0.005 ||
       grasping_state.gripper_pose.pose.position.x > max_x_i + 0.005||
       grasping_state.gripper_pose.pose.position.y < min_y_i - 0.005 ||
       grasping_state.gripper_pose.pose.position.y > max_y_i + 0.005)
    {
        return false;
    }
    
//    if(grasping_state.gripper_pose.pose.position.x < gripper_in_x_i)
    if(grasping_state.gripper_pose.pose.position.x < max_x_i + 0.005)
    {
        if(grasping_state.gripper_pose.pose.position.y < min_y_i + gripper_out_y_diff - 0.005 ||
         grasping_state.gripper_pose.pose.position.y > max_y_i - gripper_out_y_diff + 0.005)
        {
            return false;
        }  
    }
    
    
        
    //Check object is in its range
    if(grasping_state.object_pose.pose.position.x < min_x_o ||
       grasping_state.object_pose.pose.position.x > max_x_o ||
       grasping_state.object_pose.pose.position.y < min_y_o ||
       grasping_state.object_pose.pose.position.y > max_y_o ||
       grasping_state.object_pose.pose.position.z < min_z_o) // Object has fallen
    {
        return false;
    }
    return isValid;
}

bool VrepDataInterface::StepActual(GraspingStateRealArm& state, double random_num, int action, double& reward, GraspingObservation& obs) const {
    return Step(state, random_num, action, reward, obs, false);
}





