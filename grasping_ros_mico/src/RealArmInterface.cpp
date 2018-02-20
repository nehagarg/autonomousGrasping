/* 
 * File:   RealArmInterface.cpp
 * Author: neha
 * 
 * Created on May 6, 2015, 2:06 PM
 */

#include "RealArmInterface.h"

RealArmInterface::RealArmInterface(int start_state_index_) : VrepDataInterface(start_state_index_) {
    micoActionFeedbackClient = grasping_n.serviceClient<grasping_ros_mico::MicoActionFeedback>("mico_action_feedback_server");
    //realArmObs = true;
    max_x_i = 0.5279 + 0.08;  // range for gripper movement
    //min_y_i = 0.0816 - 0.08; // range for gripper movement
    //max_y_i = 0.2316 + 0.08;
}

RealArmInterface::RealArmInterface(const RealArmInterface& orig) {
}

RealArmInterface::~RealArmInterface() {
}

bool RealArmInterface::StepActual(GraspingStateRealArm& state, double random_num, int action, double& reward, GraspingObservation& obs) const {
    /*if (action == A_OPEN)
    {
        action = 0;
    }*/
    GraspingStateRealArm initial_grasping_state = state;
    grasping_ros_mico::MicoActionFeedback micoActionFeedback_srv;
    micoActionFeedback_srv.request.check_touch = RobotInterface::check_touch;
    micoActionFeedback_srv.request.check_vision_movement = RobotInterface::version7;
    if(action < A_CLOSE)
    {
        //std::cout << "gripper statsu is  " << state.gripper_status << std::endl ;
        micoActionFeedback_srv.request.action = micoActionFeedback_srv.request.ACTION_MOVE;
        micoActionFeedback_srv.request.move_x = 0;
        micoActionFeedback_srv.request.move_y = 0;
        if (state.gripper_status == 0)
            {
            int action_offset = (action/(A_DECREASE_X - A_INCREASE_X)) * (A_DECREASE_X - A_INCREASE_X);
            double movement_value = get_action_range(action, action_offset);
            //std::cout << "Movement value is " << movement_value << std::endl;
            if(action_offset == A_INCREASE_X)
            {
                if ((state.gripper_pose.pose.position.x + movement_value) > max_x_i)
                {
                    movement_value = max_x_i - state.gripper_pose.pose.position.x;
                }
                micoActionFeedback_srv.request.move_x = movement_value;
            }
            else if(action_offset == A_DECREASE_X)
            {
                if ((state.gripper_pose.pose.position.x - movement_value) < min_x_i)
                {
                    movement_value = -min_x_i + state.gripper_pose.pose.position.x;
                }
                micoActionFeedback_srv.request.move_x = -1*movement_value;
            }
            else if(action_offset == A_INCREASE_Y)
            {
                if ((state.gripper_pose.pose.position.y + movement_value) > max_y_i)
                {
                    movement_value = max_y_i - state.gripper_pose.pose.position.y;
                }
                micoActionFeedback_srv.request.move_y = movement_value;
            }
            else if(action_offset == A_DECREASE_Y)
            {
                if ((state.gripper_pose.pose.position.y - movement_value) < min_y_i)
                {
                    movement_value = -min_y_i + state.gripper_pose.pose.position.y;
                }
                micoActionFeedback_srv.request.move_y = -1*movement_value;
            }
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
            state.finger_joint_state[i+1] = micoActionFeedback_srv.response.finger_joint_state[i/2];
            obs.finger_joint_state[i+1] = micoActionFeedback_srv.response.finger_joint_state[i/2];
        }
        AdjustRealFingerJointsToSimulatedJoints(state.finger_joint_state);
        AdjustRealFingerJointsToSimulatedJoints(obs.finger_joint_state);
        for(int i = 0; i < 2; i++)
        {
            int finger_index = i;
            if(RobotInterface::version6 || RobotInterface::version7)
            {
                finger_index = 1-i;
            }
            obs.touch_sensor_reading[i] = micoActionFeedback_srv.response.touch_sensor_reading[finger_index];
        }
        AdjustTouchSensorToSimulatedTouchSensor(obs.touch_sensor_reading);
        if(RobotInterface::version7)
        {
            obs.vision_movement = micoActionFeedback_srv.response.vision_movement;
        }
        obs.gripper_pose = state.gripper_pose;
        obs.mico_target_pose = obs.gripper_pose; //Not being used
        
    }
    else
    {
        std::cout << "Call for action failed " << std::endl;
            assert(false);
    }
    
    GetReward(initial_grasping_state, state, obs, action, reward);
    UpdateNextStateValuesBasedAfterStep(state,obs,reward,action);
    bool validState = IsValidState(state);
    //Decide if terminal state is reached
    if(action == A_PICK || !validState) //Wither pick called or invalid state reached
    {
        return true;
    }
    return false;
    
    /*if(action == A_CLOSE)
    {
        return true;
    }
    return false;
    */
}

void RealArmInterface::CreateStartState(GraspingStateRealArm& initial_state, std::string type) const {
    
    //TODO: Set min and max touch values by closing and opening the gripper
    //Fetch touch threshold value from mico action feedback node
    //Get robot pose and finger joints
    //Calling open gripper functon for that
    if(graspObjects.find(initial_state.object_id) == graspObjects.end())
    {
        //This will load object properties
        graspObjects[initial_state.object_id] = getGraspObject(object_id_to_filename[initial_state.object_id]);
    }
    
    VrepDataInterface::CreateStartState(initial_state, type);
    grasping_ros_mico::MicoActionFeedback micoActionFeedback_srv;
    //Move to pre grasp pos
    micoActionFeedback_srv.request.action = micoActionFeedback_srv.request.INIT_POS;
    micoActionFeedbackClient.call(micoActionFeedback_srv);
    
    micoActionFeedback_srv.request.action = micoActionFeedback_srv.request.GET_TOUCH_THRESHOLD;
    if(micoActionFeedbackClient.call(micoActionFeedback_srv))
    {
        real_touch_threshold = micoActionFeedback_srv.response.touch_sensor_reading[0];
        real_touch_value_min = 0;
        double real_touch_value_max_calc = (vrep_touch_value_max/vrep_touch_threshold)*real_touch_threshold;
        if (real_touch_value_max_calc > real_touch_value_max)
        {
            real_touch_value_max = real_touch_value_max_calc;
        }
        std::cout << "Touch params: min="<< real_touch_value_min 
                  << "thresh="<< real_touch_threshold
                  << "max=" << real_touch_value_max << std::endl;
    }
    micoActionFeedback_srv.request.action = micoActionFeedback_srv.request.ACTION_OPEN;
    if(micoActionFeedbackClient.call(micoActionFeedback_srv))
    {
       initial_state.gripper_pose = micoActionFeedback_srv.response.gripper_pose;
       //min_x offset for point clod movement is 0.39 - 0.355
       real_gripper_offset_x = min_x_i + initial_gripper_pose_index_x*0.01 - initial_state.gripper_pose.pose.position.x;
       real_gripper_offset_y = min_y_i + initial_gripper_pose_index_y*0.01 - initial_state.gripper_pose.pose.position.y;
       real_gripper_offset_z = initial_gripper_pose_z - initial_state.gripper_pose.pose.position.z;
       AdjustRealGripperPoseToSimulatedPose(initial_state.gripper_pose);
       //Finger Joints
        for(int i = 0; i < 4; i=i+2)
        {
            initial_state.finger_joint_state[i] = micoActionFeedback_srv.response.finger_joint_state[i/2];
           
        }
        AdjustRealFingerJointsToSimulatedJoints(initial_state.finger_joint_state);

    }
    
    //Get object pose
    //Ideally should call object detector but currently it is not ready
    //So adding a default object pose . When object pose from kinect is available should compute offeset from defalut pose
    
    //initial_state.object_pose = initial_state.gripper_pose;
    initial_state.object_pose.pose.position.x = graspObjects[initial_state.object_id]->initial_object_x;
    initial_state.object_pose.pose.position.y = graspObjects[initial_state.object_id]->initial_object_y;
    initial_state.object_pose.pose.position.z = graspObjects[initial_state.object_id]->initial_object_pose_z;
    AdjustRealObjectPoseToSimulatedPose(initial_state.object_pose);
    int proceed;
    std::cout << "Shall I prroceed?[1=yes]\n";
    std::cin >> proceed;
    if(proceed !=1)
    {
        assert(0==1);
    }

}




void RealArmInterface::AdjustRealFingerJointsToSimulatedJoints(double gripper_joint_values[]) const {
    //Adjust the gathered real joint values according to vrep
    for(int i = 0; i < 4; i=i+2)
    {
        if (gripper_joint_values[i+1]  < 0)
        {
            gripper_joint_values[i+1] = 0;
        }
        
        gripper_joint_values[i+1] = gripper_joint_values[i+1] - real_finger_joint_min;
            gripper_joint_values[i+1] = gripper_joint_values[i+1] * (vrep_finger_joint_max - vrep_finger_joint_min);
            gripper_joint_values[i+1] = gripper_joint_values[i+1] /(real_finger_joint_max - real_finger_joint_min);
            gripper_joint_values[i+1] = gripper_joint_values[i+1] + vrep_finger_joint_min;
        
    }
    for(int i = 0; i < 4; i=i+2)
    {
        gripper_joint_values[i] = vrep_dummy_finger_joint_min;
        if(gripper_joint_values[i+1] >= vrep_finger_joint_for_dummy_joint_value_change)
        {
            double add_value = (vrep_dummy_finger_joint_max - vrep_dummy_finger_joint_min);
            add_value = add_value*(gripper_joint_values[i+1] - vrep_finger_joint_for_dummy_joint_value_change);
            add_value = add_value/(vrep_finger_joint_max-vrep_finger_joint_for_dummy_joint_value_change);
            gripper_joint_values[i] = gripper_joint_values[i] + add_value;
        }
        
    }
}

void RealArmInterface::AdjustTouchSensorToSimulatedTouchSensor(double gripper_obs_values[]) const {

    //TODO confirm finger order is same
    for(int i = 0; i < 2; i++)
    {
        if(gripper_obs_values[i] < real_touch_value_min)
        {
            gripper_obs_values[i] = real_touch_value_min;
        }
        if(gripper_obs_values[i] > real_touch_value_max)
        {
            gripper_obs_values[i] = real_touch_value_max;
        }
        if(gripper_obs_values[i] <= real_touch_threshold)
        {
            //Interpolate between vrep min and vrep touch threshold
            gripper_obs_values[i] = gripper_obs_values[i] - real_touch_value_min;
            gripper_obs_values[i] = gripper_obs_values[i]*(vrep_touch_threshold - vrep_touch_value_min);
            gripper_obs_values[i] = gripper_obs_values[i]/(real_touch_threshold - real_touch_value_min);
            gripper_obs_values[i] = gripper_obs_values[i] + vrep_touch_value_min;
            
        }
        else
        {
            //Interpolate between vrep touch threshold and vrep max
            gripper_obs_values[i] = gripper_obs_values[i] - real_touch_threshold;
            gripper_obs_values[i] = gripper_obs_values[i]*(vrep_touch_value_max - vrep_touch_threshold);
            gripper_obs_values[i] = gripper_obs_values[i]/(real_touch_value_max - real_touch_threshold);
            gripper_obs_values[i] = gripper_obs_values[i] + vrep_touch_threshold;

        }
    }
    
}


void RealArmInterface::AdjustRealGripperPoseToSimulatedPose(geometry_msgs::PoseStamped& gripper_pose) const {
     //Adjust with vrep
    gripper_pose.pose.position.x = gripper_pose.pose.position.x + real_gripper_offset_x;
    gripper_pose.pose.position.y = gripper_pose.pose.position.y + real_gripper_offset_y;
    gripper_pose.pose.position.z = gripper_pose.pose.position.z + real_gripper_offset_z;

    
    
    //Get tip pose
    //gripper_pose.pose.position.x = gripper_pose.pose.position.x + tip_wrt_hand_link_x;
    
   }

void RealArmInterface::AdjustRealObjectPoseToSimulatedPose(geometry_msgs::PoseStamped& object_pose) const {
    //This function will be needed when using kinect to determine object pose
    //object_pose.pose.position.x = object_pose.pose.position.x - real_min_x_o + min_x_o;
    //object_pose.pose.position.y = object_pose.pose.position.y - real_min_y_o + min_y_o;
    
}












