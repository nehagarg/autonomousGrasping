/* 
 * File:   RobotInterface.h
 * Author: neha
 *
 * Created on May 5, 2015, 2:11 PM
 */

#ifndef ROBOTINTERFACE_H
#define	ROBOTINTERFACE_H

#include "GraspingStateRealArm.h"
#include "GraspingObservation.h"
#include "simulation_data_reader.h"

class RobotInterface {
public:
    RobotInterface();
    
    RobotInterface(const RobotInterface& orig);
    virtual ~RobotInterface();
    
    void PrintState(const GraspingStateRealArm& state, std::ostream& out = std::cout) const;

    void PrintAction(int action, std::ostream& out = std::cout) const;
    
    void PrintObs(const GraspingStateRealArm& state, GraspingObservation& obs, std::ostream& out = std::cout) const;
    void PrintObs(GraspingObservation& obs, std::ostream& out = std::cout) const;
    
    virtual void CreateStartState(GraspingStateRealArm& initial_state, std::string type = "DEFAULT") const = 0;
    double ObsProb(GraspingObservation grasping_obs, const GraspingStateRealArm& grasping_state, int action) const;
    bool Step(GraspingStateRealArm& state, double random_num, int action,
        double& reward, GraspingObservation& obs, bool debug = false) const;
    virtual bool StepActual(GraspingStateRealArm& state, double random_num, int action,
        double& reward, GraspingObservation& obs) const = 0;
    virtual bool IsValidState(GraspingStateRealArm grasping_state) const = 0;
    
    /*enum { //action for amazon shelf
        A_INCREASE_X = 0 ,
        A_DECREASE_X = 4 ,
        A_INCREASE_Y = 8 ,
        A_DECREASE_Y = 12,
        A_CLOSE = 16 ,
        A_OPEN = 17,
        A_PICK = 18
    };
    */
    
    enum { //action
        A_INCREASE_X = 0 ,
        A_DECREASE_X = 2 ,
        A_INCREASE_Y = 4 ,
        A_DECREASE_Y = 6,
        A_CLOSE = 8 ,
        A_OPEN = 9,
        A_PICK = 10
    };
    
    //For data collected
    double min_x_i = 0.3379; //range for gripper movement
    double max_x_i = 0.5279;  // range for gripper movement
    double min_y_i = 0.0816; // range for gripper movement
    double max_y_i = 0.2316; // range for gripper movement 
    double gripper_in_x_i = 0.3779;
    //double gripper_out_y_diff = 0.02; //for amazon shelf
    double gripper_out_y_diff = 0.0;    //for open table
    
    double min_x_o = 0.4586; //range for object location
    double max_x_o = 0.5517;  // range for object location
    double min_y_o = 0.0829; // range for object location
    double max_y_o = 0.2295; // range for object location
    
   // double min_z_o = 1.6900 ;//below this means object has fallen down //for amazon shelf
    double min_z_o = 1.1200 ; //for objects on table
    double pick_z_diff = 0.06;
    double pick_x_val = 0.3079;
    double pick_y_val = 0.1516;

    
protected:
    double epsilon = 0.01; //Smallest step value
    //double epsilon_multiplier = 2; //for step increments in amazon shelf
    double epsilon_multiplier = 8; //for open table
    
    double touch_sensor_mean[48];
    double touch_sensor_std[48];
    double touch_sensor_max[48];
    double touch_sensor_mean_closed_without_object[48];
    double touch_sensor_mean_closed_with_object[48];
    mutable std::vector<SimulationData> simulationDataCollectionWithObject[19];
    mutable std::vector<int> simulationDataIndexWithObject[19];
    mutable std::vector<SimulationData> simulationDataCollectionWithoutObject[19];
    mutable std::vector<int> simulationDataIndexWithoutObject[19];
    
    double get_action_range(int action, int action_type) const ;
    void GetObsFromData(GraspingStateRealArm grasping_state, GraspingObservation& grasping_obs, double random_num, int action, bool debug = false) const; //state is the state reached after performing action
    int GetGripperStatus(double finger_join_state[]) const;
    void GetObsUsingDefaultFunction(GraspingStateRealArm grasping_state, GraspingObservation& grasping_obs, bool debug = false) const;
    void GetNextStateAndObsFromData(GraspingStateRealArm current_grasping_state, GraspingStateRealArm& next_grasping_state, GraspingObservation& grasping_obs, int action, bool debug = false) const;
    void GetNextStateAndObsUsingDefaulFunction(GraspingStateRealArm& next_grasping_state, GraspingObservation& grasping_obs, int action, bool debug = false) const;
    void GetReward(GraspingStateRealArm initial_grasping_state, GraspingStateRealArm grasping_state, GraspingObservation grasping_obs, int action, double& reward) const;
    void ConvertObs48ToObs2(double current_sensor_values[], double on_bits[]) const;
    void UpdateNextStateValuesBasedAfterStep(GraspingStateRealArm& grasping_state, GraspingObservation grasping_obs, double reward, int action) const;
    
    virtual void GetRewardBasedOnGraspStability(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs, double& reward) const = 0;
    virtual bool CheckTouch(double current_sensor_values[], int on_bits[], int size = 2) const = 0;
    virtual bool IsValidPick(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs) const = 0;
    virtual void CheckAndUpdateGripperBounds(GraspingStateRealArm& grasping_state, int action) const = 0;
    virtual void GetDefaultPickState(GraspingStateRealArm& grasping_state) const = 0;
};




#endif	/* ROBOTINTERFACE_H */

