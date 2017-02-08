/* 
 * File:   RealArmInterface.cpp
 * Author: neha
 * 
 * Created on May 6, 2015, 2:06 PM
 */

#include "RealArmInterface.h"

RealArmInterface::RealArmInterface() {
    micoActionFeedbackClient = grasping_n.serviceClient<grasping_ros_mico::MicoActionFeedback>("mico_action_feedback_server");
}

RealArmInterface::RealArmInterface(const RealArmInterface& orig) {
}

RealArmInterface::~RealArmInterface() {
}

bool RealArmInterface::StepActual(GraspingStateRealArm& state, double random_num, int action, double& reward, GraspingObservation& obs) const {

    grasping_ros_mico::MicoActionFeedback micoActionFeedback_srv;
    if(action < A_CLOSE)
    {
        micoActionFeedback_srv.request.action = micoActionFeedback_srv.request.ACTION_MOVE;
        micoActionFeedback_srv.request.move_x = 0;
        micoActionFeedback_srv.request.move_y = 0;
        int action_offset = (action/(A_DECREASE_X - A_INCREASE_X)) * (A_DECREASE_X - A_INCREASE_X);
        double movement_value = get_action_range(action, action_offset);
        if(action_offset == A_INCREASE_X)
        {
            micoActionFeedback_srv.request.move_x = movement_value;
        }
        else if(action_offset == A_DECREASE_X)
        {
            micoActionFeedback_srv.request.move_x = -1*movement_value;
        }
        else if(action_offset == A_INCREASE_Y)
        {
            micoActionFeedback_srv.request.move_y = movement_value;
        }
        else if(action_offset == A_DECREASE_Y)
        {
            micoActionFeedback_srv.request.move_y = -1*movement_value;
        }
    }
    else if(action == A_CLOSE)
    {
        micoActionFeedback_srv.request.action = micoActionFeedback_srv.request.ACTION_CLOSE;
    }
    else if(action == A_OPEN)
    {
        micoActionFeedback_srv.request.action = micoActionFeedback_srv.request.ACTION_OPEN;
    }
    else if (action == A_PICK)
    {
        micoActionFeedback_srv.request.action = micoActionFeedback_srv.request.ACTION_PICK;
    }
    
    if(micoActionFeedbackClient.call(micoActionFeedback_srv))
    {
        state.gripper_pose = micoActionFeedback_srv.response.gripper_pose;
        AdjustRealGripperPoseToSimulatedPose(state.gripper_pose);
        //Finger Joints
        for(int i = 0; i < 4; i=i+2)
        {
            state.finger_joint_state[i] = micoActionFeedback_srv.response.finger_joint_state[i];
            obs.finger_joint_state[i] = micoActionFeedback_srv.response.finger_joint_state[i];
        }
        AdjustRealFingerJointsToSimulatedJoints(state.finger_joint_state);
        AdjustRealFingerJointsToSimulatedJoints(obs.finger_joint_state);
        for(int i = 0; i < 2; i++)
        {
            obs.touch_sensor_reading[i] = micoActionFeedback_srv.response.touch_sensor_reading[i];
        }
        AdjustTouchSensorToSimulatedTouchSensor(obs.touch_sensor_reading);
        
        obs.gripper_pose = state.gripper_pose;
        obs.mico_target_pose = obs.gripper_pose; //Not being used
        
    }
    else
    {
        std::cout << "Call for action failed " << std::endl;
            assert(false);
    }
    
    if(action == A_CLOSE)
    {
        return true;
    }
    return false;
    
}

void RealArmInterface::CreateStartState(GraspingStateRealArm& initial_state, std::string type) const {
    
    //Get robot pose and finger joints
    //Calling open gripper functon of step actual for that
    double reward;
    GraspingObservation grasping_obs;
    StepActual(initial_state, 0.0, A_OPEN,reward, grasping_obs );
    
    //Get object pose
    //Ideally should call object detector but currently it is not ready
    //So adding a heurostic object pose based on the gripper pose
    
    initial_state.object_pose = initial_state.gripper_pose;
    initial_state.object_pose.pose.position.x = min_x_o + 0.03;
    
    

}

//To make sure that gripper does not step out of its boundary for a particular bin
//This is used only for particles hence copied from vrep interface
void RealArmInterface::CheckAndUpdateGripperBounds(GraspingStateRealArm& grasping_state, int action) const {

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

//Only used by particles therefore copying from vrep interface
bool RealArmInterface::CheckTouch(double current_sensor_values[], int on_bits[], int size) const {
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

//Only used by particles therefore copying from vrep interface
void RealArmInterface::GetDefaultPickState(GraspingStateRealArm& grasping_state) const {
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

//Only used by particles therefore copying from vrep interface
void RealArmInterface::GetRewardBasedOnGraspStability(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs, double& reward) const {
    double pick_reward;
    Step(grasping_state, despot::Random::RANDOM.NextDouble(), A_PICK, pick_reward, grasping_obs);
    if(pick_reward == 20)
    {
        reward = -0.5;
    }
    else
    {
        reward = -1.5;
    }
}

//Only used by particles therefore copying from vrep interface
bool RealArmInterface::IsValidPick(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs) const {
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
    //distance = 0;
    //distance = distance + pow(grasping_state.gripper_pose.pose.position.x - grasping_obs.mico_target_pose.pose.position.x, 2);
    //distance = distance + pow(grasping_state.gripper_pose.pose.position.y - grasping_obs.mico_target_pose.pose.position.y, 2);
    //distance = distance + pow(grasping_state.gripper_pose.pose.position.z - grasping_obs.mico_target_pose.pose.position.z, 2);
    //distance = pow(distance, 0.5);
    //if(distance > 0.03)
   // {
   //     isValidPick = false;
   // }
      
    
    return isValidPick;
}

//only used by particles to check if the state is valid. 
//We do not know the real state for actual simulation
//Copying from vrep interface
bool RealArmInterface::IsValidState(GraspingStateRealArm grasping_state) const {
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

void RealArmInterface::AdjustRealFingerJointsToSimulatedJoints(double gripper_joint_values[]) const {
    for(int i = 0; i < 4; i++)
    {
        if(i % 2 == 0)
        {   gripper_joint_values[i] = gripper_joint_values[i] - real_finger_joint_min;
            gripper_joint_values[i] = gripper_joint_values[i] * (vrep_finger_joint_max - vrep_finger_joint_min);
            gripper_joint_values[i] = gripper_joint_values[i] /(real_finger_joint_max - real_finger_joint_min);
            gripper_joint_values[i] = gripper_joint_values[i] + vrep_finger_joint_min;
        }
        else 
        {
            gripper_joint_values[i] = 0;
        }
    }
}

void RealArmInterface::AdjustTouchSensorToSimulatedTouchSensor(double gripper_obs_values[]) const {
    double gripper_obs_values_copy[2];
    for(int i = 0; i < 2; i++)
    {
        gripper_obs_values_copy[i] = gripper_obs_values[i];
    }
    /*for(int i = 0; i < 12 ; i++)
    {
        gripper_obs_values[i] = gripper_obs_values_copy[i+12];
    }
    for(int i = 12; i < 24 ; i++)
    {
        gripper_obs_values[i] = gripper_obs_values_copy[i-12];
    }
    for(int i = 24; i < 36 ; i++)
    {
        gripper_obs_values[i] = gripper_obs_values_copy[i+12];
    }
    for(int i = 36; i < 48 ; i++)
    {
        gripper_obs_values[i] = gripper_obs_values_copy[i-12];
    }*/
    //TODO : correct this function
}


void RealArmInterface::AdjustRealGripperPoseToSimulatedPose(geometry_msgs::PoseStamped& gripper_pose) const {
    //Get tip pose
    gripper_pose.pose.position.x = gripper_pose.pose.position.x + tip_wrt_hand_link_x;
    
    //Adjust with vrep
    gripper_pose.pose.position.x = gripper_pose.pose.position.x - real_gripper_in_x_i + gripper_in_x_i;
    gripper_pose.pose.position.y = gripper_pose.pose.position.y - real_min_y_i + min_y_i + gripper_out_y_diff;
}

void RealArmInterface::AdjustRealObjectPoseToSimulatedPose(geometry_msgs::PoseStamped& object_pose) const {
    object_pose.pose.position.x = object_pose.pose.position.x - real_min_x_o + min_x_o;
    object_pose.pose.position.y = object_pose.pose.position.y - real_min_y_o + min_y_o;
    
}

double RealArmInterface::ObsProb(GraspingObservation grasping_obs, const GraspingStateRealArm& grasping_state, int action) const {
    GraspingObservation grasping_obs_expected;
    
    GetObsFromData(grasping_state, grasping_obs_expected, despot::Random::RANDOM.NextDouble(), action);
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
    for(int i = 0; i < 4; i=i+2)
    {
        finger_distance = finger_distance + abs(grasping_obs.finger_joint_state[i] - grasping_obs_expected.finger_joint_state[i]);
    }
    finger_distance = finger_distance/4.0;
    
    double sensor_distance = 0;
    int real_sensor_bits[4];
    int particle_sensor_bits[4];
    int particle_sensor_bits_all[48];
    CheckTouch(grasping_obs_expected.touch_sensor_reading, particle_sensor_bits_all);
    for(int i = 0; i < 4; i++)
    {
        real_sensor_bits[i] = 0;
        particle_sensor_bits[i] = 0;
    }
    for(int i = 0; i < 48; i++)
    {
        if(grasping_obs.touch_sensor_reading[i] > 0.01)
        {
            real_sensor_bits[i/12] = 1;
        }
        if(particle_sensor_bits_all[i] > 0.01)
        {
            particle_sensor_bits[i/12] = 1;
        }
    }
    //TODO check if better results by comparing only 4 values 
    for(int i = 0; i < 48; i++)
    {
        sensor_distance = sensor_distance + abs(grasping_obs.touch_sensor_reading[i] - particle_sensor_bits_all[i]);
    }
    sensor_distance = sensor_distance/4.0;
    
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
    
    double prob = pow(2, -1*total_distance/tau);

    return prob;
}










