/* 
 * File:   VrepDataInterface.cpp
 * Author: neha
 * 
 * Created on November 11, 2016, 6:30 PM
 */

#include "VrepDataInterface.h"
#include "RobotInterface.h"

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
        if(current_sensor_values[i] > vrep_touch_threshold)
        {
            touchDetected = true;
            on_bits[i] = 1;
        }
    }
    
    return touchDetected;
}

void VrepDataInterface::CreateStartState(GraspingStateRealArm& initial_state, std::string type) const {
    
    GetDefaultStartState(initial_state);
    if(start_state_index == -2)
    {//Position that mimics real arm object slipping
      initial_state.object_pose.pose.position.y = initial_object_y + 0.07 ;
      initial_state.object_pose.pose.position.x = initial_object_x + 0.03;
      return;
    }     
    if (start_state_index >= 0 ) 
    {
        std::cout << "Start_state index is " << start_state_index << std::endl;
        
        //std::vector<GraspingStateRealArm> initial_states =  InitialStartStateParticles(initial_state);
        //std::cout << "Particle size is " <<  initial_states.size()<< std::endl;
        //int ii = start_state_index % initial_states.size();
        //initial_state.object_pose.pose.position.x = initial_states[ii].object_pose.pose.position.x ;
        //initial_state.object_pose.pose.position.y = initial_states[ii].object_pose.pose.position.y ;
        //initial_state = initial_states[i];
    //return  initial_states[ii];
        int start_state_index_mod = start_state_index % 81;
        
        if(start_state_index_mod < 49)
        {
        int i = (start_state_index_mod % 49) / 7;
        int j = (start_state_index_mod % 49) % 7;
        initial_state.object_pose.pose.position.y = initial_object_y -0.03 + (j*0.01);
        initial_state.object_pose.pose.position.x = initial_object_x -0.03 + (i*0.01);
        }
        else
        {
            int actual_index = (start_state_index_mod - 49) % 32;
            int i ;
            int j;
            if(actual_index < 8)
            {
                i = actual_index % 8;
                j = 0;
            }
            else if(actual_index < 16)
            {
                i = 8;
                j = (actual_index - 8) % 8;
            }
            else if(actual_index < 24)
            {
                i = 8 - (((actual_index - 16))%8);
                j=8;
            }
            else
            {
                i = 0;
                j = 8 - (((actual_index - 24))%8);
            }
            initial_state.object_pose.pose.position.y = initial_object_y -0.04 + (j*0.01);
            initial_state.object_pose.pose.position.x = initial_object_x -0.04 + (i*0.01);
        }
    }
    else
    {
       while(true){
            
          GenerateGaussianParticleFromState(initial_state, type);
          // initial_state.object_pose.pose.position.y = initial_object_y ;
          // initial_state.object_pose.pose.position.x = initial_object_x ;
           //Setting for which learned policy shows wierd transition in data model
           //This is because the gripper comes below hand
           //initial_state.object_pose.pose.position.x = 0.521875;
           //initial_state.object_pose.pose.position.y = 0.138427;
           //initial_state.gripper_pose.pose.position.x =0.447917;
           //initial_state.gripper_pose.pose.position.y = 0.151595;
           //break;
           //initial_state.object_pose.pose.position.x = 0.518375;
           //initial_state.object_pose.pose.position.y = 0.155915;
           //initial_state.object_pose.pose.position.x = initial_object_x - 0.03;
           //initial_state.object_pose.pose.position.y = initial_object_y;
           // the engine for generator samples from a distribution
           /* unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator(seed);
            initial_state.object_pose.pose.position.x = Gaussian_Distribution(generator,0.498689, 0.03 );
            initial_state.object_pose.pose.position.y = Gaussian_Distribution(generator,0.148582, 0.03 );
            
            */
           
           
           if(IsValidState(initial_state))
            {
               if((initial_state.object_pose.pose.position.x <= initial_object_x + 0.03) &&
                 (initial_state.object_pose.pose.position.x >= initial_object_x - 0.03) &&
                 (initial_state.object_pose.pose.position.y <= initial_object_y + 0.03) &&
                 (initial_state.object_pose.pose.position.y >= initial_object_y - 0.03))
            {
                break;
            }
               
            }
        
      }
    }
    
    
        
    

}

std::vector<GraspingStateRealArm> VrepDataInterface::InitialStartStateParticles(const GraspingStateRealArm start) const
 {
   
    //cout << "In initial belief" << endl;
    std::vector<GraspingStateRealArm> particles;
    int num_particles = 0;
    
    
    for(int i = 0; i < 7; i++)
    {
        for(int j = 0; j < 7; j++)
        {
           
            GraspingStateRealArm grasping_state(start);
            
            grasping_state.object_pose.pose.position.y = initial_object_y -0.03 + (j*0.01);
            grasping_state.object_pose.pose.position.x = initial_object_x -0.03 + (i*0.01);
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
    double z_diff_from_cylinder = initial_object_pose_z[grasping_state.object_id] - initial_object_pose_z[0];
    distance = distance + pow(grasping_state.gripper_pose.pose.position.x - grasping_state.object_pose.pose.position.x, 2);
    distance = distance + pow(grasping_state.gripper_pose.pose.position.y - grasping_state.object_pose.pose.position.y, 2);
    distance = distance + pow(grasping_state.gripper_pose.pose.position.z - grasping_state.object_pose.pose.position.z + z_diff_from_cylinder, 2);
    distance = pow(distance, 0.5);
    if(distance > 0.08)
    {
        isValidPick= false;
    }
            
    
    // if target and tip are far from each other set false
    //Removing this as mico target pose is not updated properly
    /*distance = 0;
    distance = distance + pow(grasping_state.gripper_pose.pose.position.x - grasping_obs.mico_target_pose.pose.position.x, 2);
    distance = distance + pow(grasping_state.gripper_pose.pose.position.y - grasping_obs.mico_target_pose.pose.position.y, 2);
    distance = distance + pow(grasping_state.gripper_pose.pose.position.z - grasping_obs.mico_target_pose.pose.position.z, 2);
    distance = pow(distance, 0.5);
    if(distance > 0.03)
    {
        //std::cout << "Pick false because of tip" << std::endl;
        isValidPick = false;
    }
      */
    
    return isValidPick;
}

bool VrepDataInterface::IsValidState(GraspingStateRealArm grasping_state) const {
    bool isValid = true;
    int object_id = grasping_state.object_id;
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
       grasping_state.object_pose.pose.position.z < min_z_o[object_id]) // Object has fallen
    {
        return false;
    }
    return isValid;
}

bool VrepDataInterface::StepActual(GraspingStateRealArm& state, double random_num, int action, double& reward, GraspingObservation& obs) const {
    std::cout << action << std::endl;
    return Step(state, random_num, action, reward, obs, false);
}





