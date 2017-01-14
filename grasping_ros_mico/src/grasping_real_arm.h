/* 
 * File:   grasping.h
 * Author: neha
 *
 * Created on September 3, 2014, 10:45 AM
 */

#ifndef GRASPING_REAL_ARM_H
#define	GRASPING_REAL_ARM_H

#include <despot/core/pomdp.h> 
#include "history_with_reward.h"
#include "LearningModel.h"


#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Vector3.h"
#include "sensor_msgs/JointState.h"


#include "VrepInterface.h"
#include "RealArmInterface.h"
#include "GraspingStateRealArm.h"
#include "GraspingObservation.h"









class GraspingRealArm : public LearningModel {
public:

    GraspingRealArm(const GraspingRealArm& orig);
    GraspingRealArm(int start_state_index_, int interfaceType = 1);
    GraspingRealArm(int start_state_index_, VrepInterface* roboInterface_);
    GraspingRealArm(std::string dataFileName, int start_state_index_);
    
    virtual ~GraspingRealArm();

    int NumActions() const {
        return robotInterface->A_PICK + 1;
    };
    
     /* Deterministic simulative model.*/
    bool Step(State& state, double random_num, int action,
        double& reward, uint64_t& obs) const {
        //std::cout << "Step: This should not have been printed";
        ObservationClass observation;
        bool isTerminal = Step(state, random_num, action, reward, observation);
        obs = observation.GetHash();
        //assert(false);
        return isTerminal;}
    ;
    bool StepActual(State& state, double random_num, int action,
        double& reward, uint64_t& obs) const { //Function for actual step in simulation or real robot
                //std::cout << "Step: This should not have been printed";
        ObservationClass observation;
        bool isTerminal = StepActual(state, random_num, action, reward, observation);
        obs = observation.GetHash();
        //assert(false);
        return isTerminal;
        
    }
     bool Step(State& state, double random_num, int action,
        double& reward, ObservationClass& obs) const;
    bool StepActual(State& state, double random_num, int action,
        double& reward, ObservationClass& obs) const; //Function for actual step in simulation or real robot
    
    /* Functions related to beliefs and starting states.*/
    double ObsProb(uint64_t obs, const State& state, int action) const {
        //std::cout << "ObsProb: This should not have been printed"; 
        ObservationClass observation;
        observation.SetIntObs(obs);
        return ObsProb(observation, state, action);
        //return false;
    };
    double ObsProb(ObservationClass obs, const State& state, int action) const;
    Belief* InitialBelief(const State* start, std::string type = "DEFAULT") const;
    State* CreateStartState(std::string type = "DEFAULT") const;
 
    /* Bound-related functions.*/
    double GetMaxReward() const { return reward_max;}
    ValuedAction GetMinRewardAction() const {
        return ValuedAction(robotInterface->A_OPEN, -1);
         
                
    };
    ScenarioLowerBound* CreateScenarioLowerBound(std::string name = "DEFAULT", std::string particle_bound_name = "DEFAULT") const;
 
    /* Memory management.*/
    State* Allocate(int state_id, double weight) const {
        //num_active_particles ++;
        GraspingStateRealArm* state = memory_pool_.Allocate();
        state->state_id = state_id;
        state->weight = weight;
        return state;
    };
    State* Copy(const State* particle) const {
        //num_active_particles ++;
       GraspingStateRealArm* state = memory_pool_.Allocate();
        *state = *static_cast<const GraspingStateRealArm*>(particle);
        state->SetAllocated();
        return state;
    };
    void Free(State* particle) const {
        //num_active_particles --;
        memory_pool_.Free(static_cast<GraspingStateRealArm*>(particle));
    };
    
    int NumActiveParticles() const {
	return memory_pool_.num_allocated();
    }
    
    /**printing functions*/

    void PrintState(const State& state, std::ostream& out = std::cout) const;

    void PrintAction(int action, std::ostream& out = std::cout) const;
    
    void PrintObs(const State& state, ObservationClass& obs, std::ostream& out = std::cout) const;
    void PrintObs(ObservationClass& obs, std::ostream& out = std::cout) const;
    void PrintObs(const State& state, uint64_t obs, std::ostream& out = std::cout) const {
        ObservationClass observation;
        observation.SetIntObs(obs);
        PrintObs(state,observation,out);
    }
    void PrintBelief(const Belief& belief, std::ostream& out = std::cout) const;
    void DisplayBeliefs(ParticleBelief* belief, std::ostream& ostr) const;
    void DisplayState(const State& state, std::ostream& ostr) const;
    
    int  num_sampled_objects = 27;
    std::string learning_data_file_name;

    mutable MemoryPool<GraspingStateRealArm> memory_pool_;
    double reward_max = 20;
    double step_cost = -1;
    int start_state_index = -1;
       
    RobotInterface* robotInterface;
    
    std::hash<std::string> obsHash;
    mutable std::map<uint64_t, GraspingObservation> obsHashMap;
    mutable GraspingStateRealArm initial_state;
    
    //vector<HistoryWithReward*> LearningData() const;
    //ObservationClass GetInitialObs() const;
    //int GetActionIdFromString(string dataline ) const;
    std::vector<State*> InitialBeliefParticles(const State* start, std::string type="DEFAULT") const;
    
    mutable ros::NodeHandle grasping_display_n; 
    // data used for this graspcup example
    mutable ros::Publisher pub_belief;
    mutable ros::Publisher pub_gripper;
    
    
    //Learning model
    std::vector<HistoryWithReward*> LearningData() const {
        std::vector<HistoryWithReward*> ans;
        return ans;}
    //Not used anymore should be removed
    uint64_t GetInitialObs() const {return 0;} ; //Not used anymore should be removed
    double GetDistance(int action1, uint64_t obs1, int action2, uint64_t obs2) const {return 0;} //Not used anymore should be removed
    
    void PrintObs(uint64_t obs, std::ostream& out = std::cout) const {
        ObservationClass observation;
        observation.SetIntObs(obs);
        PrintObs(observation,out);
    }; 
    void GenerateAdaboostTestFile(uint64_t obs, History h) const {}//Not used anymore should be removed
    int GetStartStateIndex() const {
        return start_state_index;
    }
        
};



#endif	/* GRASPING_REAL_ARM_H */
