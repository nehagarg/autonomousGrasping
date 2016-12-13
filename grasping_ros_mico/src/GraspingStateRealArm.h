/* 
 * File:   GraspingStateRealArm.h
 * Author: neha
 *
 * Created on May 5, 2015, 11:18 AM
 */

#ifndef GRASPINGSTATEREALARM_H
#define	GRASPINGSTATEREALARM_H

#include <despot/core/pomdp.h> 
#include "geometry_msgs/PoseStamped.h"

class GraspingStateRealArm : public despot::State {
    public:
        geometry_msgs::PoseStamped gripper_pose;
        geometry_msgs::PoseStamped object_pose;
        double finger_joint_state[4]; //For finger configuration
        int object_id ;
        //double x_change = 0.0; //difference in x coordinate after taking an action
        //double y_change = 0.0; // change in y coordinate after taking an action
        //double z_change = 0.0; //change in z coordinate after taking an action
        int touch[2]; //Store if the touch is being observed at 2 fingers , useful for stop until touch actions
        int pre_action; //Store previous action, useful for putting restrictions on next action. eg. only open/pick after close
        int gripper_status; //1 for not stable or object not inside it 2 for stable grasp with object inside gripper //Based on reward achieved in last state and finger configuration. Useful to determine if the gripper did a stable grasp or unstable grasp and previous action was close
        
    GraspingStateRealArm() {
        object_id = -1;
        touch[0] = 0;
        touch[1] = 0;
        pre_action = -1;
        gripper_status = 0;
    }
    
    GraspingStateRealArm(const GraspingStateRealArm& initial_state) : State()
    {
        gripper_pose.pose.position.x  = initial_state.gripper_pose.pose.position.x ;
        gripper_pose.pose.position.y  = initial_state.gripper_pose.pose.position.y ;
        gripper_pose.pose.position.z  = initial_state.gripper_pose.pose.position.z ;
        gripper_pose.pose.orientation.x = initial_state.gripper_pose.pose.orientation.x ;
        gripper_pose.pose.orientation.y = initial_state.gripper_pose.pose.orientation.y ;
        gripper_pose.pose.orientation.z = initial_state.gripper_pose.pose.orientation.z  ;
        gripper_pose.pose.orientation.w = initial_state.gripper_pose.pose.orientation.w ;
        object_pose.pose.position.x = initial_state.object_pose.pose.position.x ;
        object_pose.pose.position.y = initial_state.object_pose.pose.position.y ;
        object_pose.pose.position.z = initial_state.object_pose.pose.position.z ;
        object_pose.pose.orientation.x = initial_state.object_pose.pose.orientation.x  ;
        object_pose.pose.orientation.y = initial_state.object_pose.pose.orientation.y ;
        object_pose.pose.orientation.z = initial_state.object_pose.pose.orientation.z  ; 
        object_pose.pose.orientation.w = initial_state.object_pose.pose.orientation.w ;
        finger_joint_state[0] = initial_state.finger_joint_state[0]  ;
        finger_joint_state[1] = initial_state.finger_joint_state[1] ;
        finger_joint_state[2]= initial_state.finger_joint_state[2] ;
        finger_joint_state[3]= initial_state.finger_joint_state[3] ;
        object_id = initial_state.object_id;
    }

    ~GraspingStateRealArm() {
    }
    
    void getStateFromStream(std::istream& inputString) {
            char c;
    inputString >> gripper_pose.pose.position.x;
    inputString >> gripper_pose.pose.position.y;
    inputString >> gripper_pose.pose.position.z;
    inputString >> gripper_pose.pose.orientation.x;
    inputString >> gripper_pose.pose.orientation.y;
    inputString >> gripper_pose.pose.orientation.z;
    inputString >> gripper_pose.pose.orientation.w;
    inputString >> c;

    inputString >> object_pose.pose.position.x;
    inputString >> object_pose.pose.position.y;
    inputString >> object_pose.pose.position.z;
    inputString >> object_pose.pose.orientation.x;
    inputString >> object_pose.pose.orientation.y;
    inputString >> object_pose.pose.orientation.z;
    inputString >> object_pose.pose.orientation.w;
    inputString >> c;

    for (int i = 0; i < 4; i++) {
       inputString >> finger_joint_state[i];
    }

    }
    
    void getStateFromString(std::string state_string) {
            //  0.3379 0.1516 1.73337 -0.694327 -0.0171483 -0.719 -0.0255881|0.4586 0.0829 1.7066 -0.0327037 0.0315227 -0.712671 0.700027|-2.95639e-05 0.00142145 -1.19209e-
    //06 -0.00118446
    std::istringstream inputString(state_string);
    getStateFromStream(inputString);
    }
    
    
    
};

#endif	/* GRASPINGSTATEREALARM_H */

