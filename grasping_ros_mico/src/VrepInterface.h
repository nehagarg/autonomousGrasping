/* 
 * File:   VrepInterface.h
 * Author: neha
 *
 * Created on May 4, 2015, 6:30 PM
 */

#ifndef VREPINTERFACE_H
#define	VREPINTERFACE_H

#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Vector3.h"
#include "sensor_msgs/JointState.h"

#include "vrep_common/simRosStartSimulation.h"
#include "vrep_common/simRosStopSimulation.h"
#include "vrep_common/simRosGetJointState.h"
#include "vrep_common/simRosGetObjectHandle.h"
#include "vrep_common/simRosGetObjectPose.h"
#include "vrep_common/simRosSetObjectPose.h"
#include "vrep_common/simRosReadForceSensor.h"
#include "vrep_common/simRosGetJointState.h"
#include "vrep_common/simRosSetJointPosition.h"
#include "vrep_common/simRosSetIntegerSignal.h"

#include "VrepDataInterface.h"

class VrepInterface : public VrepDataInterface {
public:
    VrepInterface(int start_state_index_ = -1);
    VrepInterface(const VrepInterface& orig);
    virtual ~VrepInterface();
    
    void GatherData(int object_id = 0) const;
    void GatherJointData(int object_id = 0) const;
    void GatherGripperStateData(int object_id = 0) const;
   

    bool StepActual(GraspingStateRealArm& state, double random_num, int action,
        double& reward, GraspingObservation& obs) const;
    void CreateStartState(GraspingStateRealArm& initial_state, std::string type = "DEFAULT") const;
    //bool IsValidState(GraspingStateRealArm grasping_state) const;
    

    
private:
    mutable ros::NodeHandle grasping_n; 
    mutable ros::ServiceClient sim_start_client;
    mutable ros::ServiceClient sim_stop_client;
    mutable ros::ServiceClient sim_get_object_handle_client;
    mutable ros::ServiceClient sim_get_object_pose_client;
    mutable ros::ServiceClient sim_set_object_pose_client;
    mutable ros::ServiceClient sim_read_force_sensor_client;
    mutable ros::ServiceClient sim_get_joint_state_client;
    mutable ros::ServiceClient sim_set_joint_position_client;
    mutable ros::ServiceClient sim_set_integer_signal_client;
    //mutable ros::ServiceClient sim_get_integer_signal_client;
    mutable ros::ServiceClient sim_clear_integer_signal_client;
    
    
    int mico_target_handle;
    int mico_tip_handle;
    int target_object_handle;
    int force_sensor_handles[48];
    int finger_joint_handles[4];
    int arm_joint_handles[6];
    
    double joint_angles_initial_position[20][16][10];
    

    
    
    //bool IsValidPick(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs) const ;
    //void CheckAndUpdateGripperBounds(GraspingStateRealArm& grasping_state, int action) const;
    //void GetDefaultPickState(GraspingStateRealArm& grasping_state) const;
    //void GetRewardBasedOnGraspStability(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs, double& reward) const;
    //bool CheckTouch(double current_sensor_values[], int on_bits[], int size = 2) const;
    
    bool IsReachableState(GraspingStateRealArm grasping_state, geometry_msgs::PoseStamped mico_target_pose) const ;
    void GetStateFromVrep(GraspingStateRealArm& state) const;
    void GetTouchSensorReadingFromVrep(double touch_sensor_reading[48]) const;
    bool TakeStepInVrep(int action_offset, int step_no, bool& alreadyTouching, GraspingStateRealArm& grasping_state, GraspingObservation& grasping_obs, double& reward) const;
    void OpenCloseGripperInVrep(int action_offset, GraspingStateRealArm& grasping_state, GraspingObservation& grasping_obs, double& reward) const;
    void PickActionInVrep(int action_offset, GraspingStateRealArm& grasping_state, GraspingObservation& grasping_obs, double& reward) const;
    void SetObjectPose(geometry_msgs::PoseStamped object_pose, int handle, int relativeHandle = -1) const;
    void SetMicoTargetPose(geometry_msgs::PoseStamped micoTargetPose) const;
    void SetGripperPose(int i, int j, geometry_msgs::PoseStamped micoTargetPose) const;
    void SetGripperPose(int i, int j) const;
    void GetNextStateAndObservation(GraspingStateRealArm& grasping_state, GraspingObservation& grasping_obs, geometry_msgs::PoseStamped micoTargetPose) const;
    void WaitForArmToStabilize() const;
    void WaitForStability(std::string signal_name, std::string topic_name, int signal_on_value, int signal_off_value) const;
    

};

#endif	/* VREPINTERFACE_H */
