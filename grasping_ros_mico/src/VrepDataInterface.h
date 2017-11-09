/* 
 * File:   VrepDataInterface.h
 * Author: neha
 *
 * Created on November 11, 2016, 6:29 PM
 */

#ifndef VREPDATAINTERFACE_H
#define	VREPDATAINTERFACE_H

#include "RobotInterface.h"

class VrepDataInterface  : public RobotInterface{
public:
    VrepDataInterface(int start_state_index_ = -1);
    VrepDataInterface(const VrepDataInterface& orig);
    virtual ~VrepDataInterface();
    
    int start_state_index;
    void CheckAndUpdateGripperBounds(GraspingStateRealArm& grasping_state, int action) const;

    bool CheckTouch(double current_sensor_values[], int on_bits[], int size = 2) const;
    

    virtual void CreateStartState(GraspingStateRealArm& initial_state, std::string type) const;
    virtual std::map<int,double> GetBeliefObjectProbability(std::vector<int> belief_object_ids) const
    {
        return RobotInterface::GetBeliefObjectProbability(belief_object_ids);
    }

    void GetDefaultPickState(GraspingStateRealArm& grasping_state) const;
    

    void GetRewardBasedOnGraspStability(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs, double& reward) const;


    bool IsValidPick(GraspingStateRealArm grasping_state, GraspingObservation grasping_obs) const;
    

    bool IsValidState(GraspingStateRealArm grasping_state) const;


    virtual bool StepActual(GraspingStateRealArm& state, double random_num, int action, double& reward, GraspingObservation& obs) const;

    std::vector<GraspingStateRealArm> InitialStartStateParticles(const GraspingStateRealArm start) const;



private:

};


#endif	/* VREPDATAINTERFACE_H */

