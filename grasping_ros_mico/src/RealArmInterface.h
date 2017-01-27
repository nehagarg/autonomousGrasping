/* 
 * File:   RealArmInterface.h
 * Author: neha
 *
 * Created on May 6, 2015, 2:06 PM
 */

#ifndef REALARMINTERFACE_H
#define	REALARMINTERFACE_H

#include "RobotInterface.h"
#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/Float32MultiArray.h"
#include "grasping_ros_mico/MicoActionFeedback.h"

class RealArmInterface : public RobotInterface{
public:
    RealArmInterface();
    RealArmInterface(const RealArmInterface& orig);
    virtual ~RealArmInterface();
    

    bool StepActual(GraspingStateRealArm& state, double random_num, int action, double& reward, GraspingObservation& obs) const;
    
    void CreateStartState(GraspingStateRealArm& initial_state, std::string type) const;
    
    bool IsValidState(GraspingStateRealArm grasping_state) const;
    
    double ObsProb(GraspingObservation grasping_obs, const GraspingStateRealArm& grasping_state, int action) const;
    
        //For real scenario
    double real_min_x_i = 0.3379; //range for gripper movement
    double real_max_x_i = 0.5279;  // range for gripper movement
    double real_min_y_i = 0.0816; // range for gripper movement
    double real_max_y_i = 0.2316; // range for gripper movement 
    double real_gripper_in_x_i = 0.3779;
 
    
    double real_min_x_o = 0.4586; //range for object location
    double real_max_x_o = 0.5517;  // range for object location
    double real_min_y_o = 0.0829; // range for object location
    double real_max_y_o = 0.2295; // range for object location
    
    double vrep_finger_joint_min = 0;
    double vrep_finger_joint_max = 45;
    double real_finger_joint_min = 0; //To confirm
    double real_finger_joint_max = 115; //To confirm
    
    double tip_wrt_hand_link_x = 0.127;

private:
    
    mutable ros::NodeHandle grasping_n;
    mutable ros::ServiceClient micoActionFeedbackClient;
    
    
    void CheckAndUpdateGripperBounds(GraspingStateRealArm& grasping_state, int action) const;
    bool CheckTouch(double current_sensor_values[], int on_bits[], int size = 2) const;
    void GetDefaultPickState(GraspingStateRealArm& grasping_state) const;
    void GetRewardBasedOnGraspStability(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs, double& reward) const;
    bool IsValidPick(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs) const;

    void AdjustRealGripperPoseToSimulatedPose(geometry_msgs::PoseStamped& gripper_pose) const;
    void AdjustRealObjectPoseToSimulatedPose(geometry_msgs::PoseStamped& object_pose) const;
    void AdjustRealFingerJointsToSimulatedJoints(double gripper_joint_values[]) const;
    void AdjustTouchSensorToSimulatedTouchSensor(double gripper_joint_values[]) const;


};

#endif	/* REALARMINTERFACE_H */

